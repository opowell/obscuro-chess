// ---------------------------------------------------------------------------
// Training data for a distilled leaf evaluator.
//
//   node scripts/collect-evals.mjs --sessions <corpus> [--games N] [--depth 7]
//                                  [--worlds 8] [--out evals.jsonl]
//
// Dumps (position, per-move Stockfish score) pairs in the SAME DISTRIBUTION THE
// LEAF EVALUATOR SEES, which is the only reason this script exists rather than
// "run Stockfish over a PGN". The evaluator is never asked about the real board:
// it is asked about SAMPLED BELIEF WORLDS — boards that are consistent with one
// player's observations and mostly not the position actually on the table. Train
// on real games and you fit a distribution the net is never queried on, and the
// mismatch will not show up as a training error, only as a weaker engine.
//
// So: replay each corpus game, and at every turn of the seat under study draw
// belief worlds exactly as the search does (FogChess.sampleWorlds, which honours
// the shipped α), then ask the engine for one MultiPV sweep per world. That
// sweep is precisely what scoreChildren consumes, so one call yields ~30
// labelled children for the price of one.
//
// Output is JSONL, one object per position: {fen, side, ply, n, moves:[[uci,cp]]}
// — flushed as it goes, because these runs are long and a dump written only at
// the end is a dump you lose.
// ---------------------------------------------------------------------------

import { existsSync, appendFileSync, writeFileSync } from 'node:fs';
import { loadCorpus, describeCorpus } from '../src/corpus.js';
import { FogChess } from '../src/FogChess.js';
import { toFEN } from '../src/fen.js';
import { multiPV, available, quit as stockfishQuit, setFreshHash, setAutoRecycle, recycleEngine }
  from '../src/stockfish.js';
import { applyCliSettings, maybePrintConfig, makeArgReader } from '../src/cli.js';

const { rest: argv, printConfig } = applyCliSettings();
await maybePrintConfig(printConfig);
const arg = makeArgReader(argv);

const SESSIONS = arg('sessions', null);
const maxGames = Number(arg('games', '40'));
const DEPTH = Number(arg('depth', '7'));
const WORLDS = Number(arg('worlds', '8'));
const OUT = arg('out', 'evals.jsonl');
const maxPlies = Number(arg('max-plies', '60'));

// Same determinism settings every measurement here needs: without fresh hash the
// engine's answer depends on what it looked at before, so identical positions
// collected at different moments would carry different labels — noise injected
// straight into the training target. See stockfish.js.
setFreshHash(true);
setAutoRecycle(false);

if (!SESSIONS || !existsSync(SESSIONS)) {
  console.error('--sessions <dir|zip|pgn|json> is required'); process.exit(1);
}
if (!(await available())) { console.error('stockfish unavailable'); process.exit(1); }

const { games, stats } = loadCorpus(SESSIONS, { maxGames, minPlies: 20 });
console.log(describeCorpus(games, stats));
console.log(`depth ${DEPTH}, ${WORLDS} worlds/turn -> ${OUT}`);

writeFileSync(OUT, '');
const seen = new Set();               // one label per position, however often drawn
let rows = 0, calls = 0, dupes = 0, empty = 0;
const t0 = Date.now();

let seed = Number(arg('seed', '20260816'));
const rng = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

for (let g = 0; g < games.length; g++) {
  const sess = games[g].sess;
  const players = JSON.parse(JSON.stringify(sess.params.players));
  let state = FogChess.createInitialState(players, { ...sess.params.config, aiTimeMs: null });
  const log = sess.log ?? [];
  await recycleEngine();              // caller-owned boundary, as everywhere else

  for (let i = 0; i < Math.min(log.length, maxPlies); i++) {
    const pa = log[i].playerActions?.[0];
    if (!pa?.action) break;
    const seat = pa.playerId;
    const obs = FogChess.getVisibleState(state, seat);

    // The search's own draw, so the training distribution IS the query
    // distribution. With fog off this returns [] and there is nothing hidden to
    // learn from, so fall back to the single real board.
    const worlds = FogChess.sampleWorlds(obs, seat, WORLDS, rng);
    const boards = worlds.length ? worlds : [obs];

    for (const w of boards) {
      const side = seat === 'white' ? 'w' : 'b';
      const fen = toFEN(w.board, w.gameSpecific ?? obs.gameSpecific, side, w.turnNumber ?? obs.turnNumber ?? 1);
      if (seen.has(fen)) { dupes++; continue; }
      seen.add(fen);
      const legal = FogChess.getLegalActions({ ...w, activePlayers: [seat] }, seat);
      if (!legal.length) continue;
      let pv = null;
      try { pv = await multiPV(fen, { multipv: Math.max(legal.length, 1), depth: DEPTH }); }
      catch { pv = null; }
      calls++;
      if (!pv || !pv.length) { empty++; continue; }
      const moves = pv.filter(x => typeof x.cp === 'number').map(x => [x.move, x.cp]);
      if (!moves.length) { empty++; continue; }
      appendFileSync(OUT, JSON.stringify({ fen, side, ply: i, n: moves.length, moves }) + '\n');
      rows++;
    }
    FogChess.onActionCommitted?.(obs, seat, pa.action);
    state = FogChess.applyActions(state, [{ playerId: seat, action: pa.action }]);
  }
  const s = ((Date.now() - t0) / 1000).toFixed(0);
  console.log(`  game ${g + 1}/${games.length}: ${rows} positions, ${s}s` +
    `  (${dupes} dupes skipped, ${empty} engine-empty)`);
}

console.log(`\n${rows} positions, ~${rows * 30} labelled children, ${empty} empty (${(100 * empty / Math.max(1, calls)).toFixed(1)}%)`);
await stockfishQuit?.();
process.exit(0);
