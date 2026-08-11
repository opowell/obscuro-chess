// ---------------------------------------------------------------------------
// A/B the belief by MOVE QUALITY, not by win/loss.
//
//   node scripts/move-quality.mjs --arm alpha            # α=1 vs α=0
//   node scripts/move-quality.mjs --arm null             # THE CONTROL. run it.
//   node scripts/move-quality.mjs --arm prior            # fitted π vs uniform π
//   node scripts/move-quality.mjs --alphas 0.5,0 --games 8 --ref-depth 14
//   node scripts/move-quality.mjs --arm temper --min-plies 0   # take every game
//
// WHY THIS EXISTS. `strength-belief.mjs` plays whole games and counts wins, and
// it cannot resolve anything: White scores ~57.5% in the paper's own 10,000-game
// self-play sample, and far more lopsidedly in ours, so a seat-swapped pair
// contributes 1-1 and only the rare upset carries signal. Sixteen games resolve
// nothing about a belief change. This harness replaces the outcome with a
// per-move measurement on the SAME positions, which is the standard fix:
//
//   • PAIRED. Both arms are asked the same question — identical recorded
//     position, identical fog, identical legal-action list — so the comparison
//     is a paired difference and the position-to-position variance (which is
//     enormous, and which the win/loss harness pays in full) cancels.
//   • BOUNDED. The score is centipawn loss against a deep reference search on
//     the TRUE board, clamped by the same LEAF_CLAMP the search uses, with a
//     hung king pinned at KING_HANG. No single position can dominate the mean.
//   • DENSE. Every ply is a datum. One 60-ply game yields ~30 per seat, where
//     the win/loss harness yields one bit per game.
//
// WHAT IT MEASURES, AND WHAT IT DOES NOT. "cp loss vs the true board" is a
// TACTICAL SOUNDNESS proxy, not the fog objective. A move that is −40 cp on the
// true board but scouts a whole file may be correct in fog and is scored as a
// mistake here; conversely a belief that gambles correctly is not rewarded.
// Information value is exactly what this metric is blind to. It is still the
// right first instrument, because a belief change that is worth anything should
// move tactical soundness in a measurable direction, and because it needs ~100×
// less compute than resolving the same question with games. Do not quote it as
// "strength" — quote it as "cp lost per move against perfect information".
//
// PROTOCOL NOTES (all three are scar tissue from OBSCURO-MOVE-PRIOR-PLAN.md):
//  1. A FRESH agent per arm per game. The exported singleton's `_carry` map holds
//     the KLUSS search tree per colour and is never reset — reusing it makes
//     results order-dependent, which once produced a completely fictional 0/16.
//  2. A FRESH `players` array per arm. Both belief stores are WeakMaps keyed by
//     that array's identity, so sharing it would share trackers between arms.
//  3. The NULL CONTROL is not optional. `--arm null` runs α=0 against α=0; if it
//     does not come back ~0 ± noise, the harness is broken and every other
//     number it printed is fiction.
// ---------------------------------------------------------------------------

import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadCorpus, describeCorpus } from '../src/corpus.js';
import { FogChess } from '../src/FogChess.js';
import { ChessObscuroAgent, makeChessLeafEval, getLeafEvalStats, resetLeafEvalStats, setGame } from '../src/ObscuroAgent.js';
import {
  setBeliefSampleAlphaForSeat, setMovePriorForSeat, setBeliefReachWeightingForSeat,
} from '../src/exactBelief.js';
import { makeMovePrior, UNIFORM_PRIOR, FITTED_WEIGHTS } from '../src/movePrior.js';
import { quit as stockfishQuit, setFreshHash } from '../src/stockfish.js';
import { applyCliSettings, maybePrintConfig, makeArgReader } from '../src/cli.js';

const HERE = dirname(fileURLToPath(import.meta.url));
// The recorded games to replay. Defaults to the three fixtures the test suite
// uses, so the script runs from a bare clone; that is enough to smoke-test the
// pipeline and nowhere near enough to draw a conclusion from. Point --sessions
// at a real corpus for that (battle-simulator keeps its recorded games in its
// own untracked sessions/ directory).
const { rest: argv, printConfig } = applyCliSettings();
await maybePrintConfig(printConfig);
const arg = makeArgReader(argv);
const SESSIONS = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
const armName = arg('arm', 'alpha');
const maxGames = Number(arg('games', '6'));
// See the loadCorpus call below for why the floor is 20 and not the loader's 10.
// `--min-plies 0` keeps every game the corpus offers.
const minPlies = Number(arg('min-plies', '20'));
const maxPlies = Number(arg('max-plies', '60'));
const refDepth = Number(arg('ref-depth', '12'));

// DETERMINISM IS THE WHOLE INSTRUMENT. The first version of this harness ran the
// agent on a wall clock (`aiTimeMs`), and the null control — two arms configured
// IDENTICALLY — picked the same move only 20% of the time: whichever arm ran
// second inherited a warmer Stockfish cache, reached deeper leaves in the same
// milliseconds, and played differently. That is a paired design in name only,
// and it put the noise floor (±50 cp) above any effect worth finding.
//
// So the budget here is structural, not temporal: power mode fixes the leaf
// depth, `timeBudgetMs: 0` removes the wall clock from the search loop entirely
// (`budgetMs && …` short-circuits every deadline check), and the tree is bounded
// by round/infoset counts instead. Same seed + same knobs ⇒ same moves, so the
// null control collapses to ~100% agreement and every difference the α arm shows
// is α. Slower per move than a clock; the point is that the number means
// something.
const dial = Number(arg('dial', '30'));            // power dial → leaf depth 2..7
const knobs = {
  timeBudgetMs: 0,                                  // no wall clock at all
  particles: Number(arg('worlds', '16')),
  maxRounds: Number(arg('rounds', '6')),
  maxInfosets: Number(arg('infosets', '1200')),
  expandPerRound: Number(arg('expand', '10')),
  cfrPerRound: 6,
  finalCfr: 50,
};
// WINSORISE the per-position cp loss. Raw cp loss has a median around 47 and a
// mean around 230: a handful of blunders set the standard error, which is why a
// 3-game run once reported z = 2.40 AGAINST a change that a 37-game run put
// mildly in favour. Clipping bounds each position's leverage while keeping
// magnitude — strictly more informative than the sign test, which discards it.
// --clip 0 restores the raw mean.
const CLIP = Number(arg('clip', '300'));
const clipLoss = x => (CLIP > 0 ? Math.min(x, CLIP) : x);
const seed0 = Number(arg('seed', '12345'));
// THE PAIRED DESIGN REQUIRES THIS. `go depth N` is not a pure function of the
// position — Stockfish carries its transposition table between searches — and
// the cache only hides that until it fills. Past `CACHE_MAX` entries a run of
// this size thrashes, positions get recomputed against a different table, and
// two IDENTICAL arms drift apart: measured at 39% of moves disagreeing, on a
// corpus where half the divergence sat on positions with a single possible
// board. `--fresh-hash 0` restores the old (fast, unsound) behaviour.
setFreshHash(arg('fresh-hash', '1') !== '0');
// --verbose prints per-ply cost, which is how the ms/move mystery got solved.
const VERBOSE = argv.includes('--verbose');
const seatArg = arg('seat', 'both');
const seats = seatArg === 'both' ? ['white', 'black'] : [seatArg];

// An arm is a (α, π) pair. `null` is the control: both arms identical.
const ARMS = {
  alpha: { a: { label: 'α=1', alpha: 1 }, b: { label: 'α=0', alpha: 0 } },
  half: { a: { label: 'α=0.5', alpha: 0.5 }, b: { label: 'α=0', alpha: 0 } },
  null: { a: { label: 'control-A', alpha: 0 }, b: { label: 'control-B', alpha: 0 } },
  // THE ONE THAT MATTERS: does the posterior change play at all once it is
  // allowed to? `reach: true` weights each sampled world's root reach by its
  // importance weight; `false` is the flat 1/N the search used until 2026-08-02.
  reach: {
    a: { label: 'posterior reach β=1', alpha: 0, reach: 1 },
    b: { label: 'uniform reach (shipped)', alpha: 0, reach: 0 },
  },
  // Tempered: half the correction, roughly double the effective sample.
  temper: {
    a: { label: 'tempered reach β=0.5', alpha: 0, reach: 0.5 },
    b: { label: 'uniform reach (shipped)', alpha: 0, reach: 0 },
  },
  // With α=0 the prior cannot reach the world draw at all, so this arm is only
  // meaningful together with α>0 — it is here to make that point measurable
  // rather than argued.
  prior: {
    a: { label: 'fitted π, α=1', alpha: 1, prior: makeMovePrior(FITTED_WEIGHTS) },
    b: { label: 'uniform π, α=1', alpha: 1, prior: UNIFORM_PRIOR },
  },
};
const arm = ARMS[armName];
if (!arm) throw new Error(`--arm must be one of ${Object.keys(ARMS).join('|')}`);
if (argv.includes('--alphas')) {
  const [x, y] = arg('alphas', '1,0').split(',').map(Number);
  arm.a = { label: `α=${x}`, alpha: x };
  arm.b = { label: `α=${y}`, alpha: y };
}

// Deterministic RNG so a rerun of the same arm reproduces exactly.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- games -------------------------------------------------------------------

if (!existsSync(SESSIONS)) {
  console.error(`No corpus at ${SESSIONS} — pass --sessions <dir|zip|pgn|json>.`);
  process.exit(1);
}
// DEFAULT 20 plies, not the loader's default 10: this harness measures move
// quality against a deep reference search, and an opening-only game contributes
// almost nothing but costs a full Stockfish pass per ply. It is a default and
// not a law — `--min-plies 0` takes everything — but changing it changes the
// corpus, so a run at a different floor is NOT comparable with the recorded
// numbers in exactBelief.js, which were all measured at 20.
//
// It matters more here than the ply cost suggests: a belief knob can only act
// through what is hidden, and |P| grows with the game. Short games are the part
// of the corpus where every arm is closest to agreeing, so admitting them adds
// positions faster than it adds signal.
const { games, stats: corpusStats } = loadCorpus(SESSIONS, { maxGames, minPlies });
if (!games.length) {
  console.error(`No chess fog games in ${SESSIONS}.\n  ${describeCorpus(games, corpusStats)}`);
  process.exit(1);
}
console.log(describeCorpus(games, corpusStats));

// Print how much of this run was actually evaluated by the engine. A run with
// fallback leaves is a run where Stockfish timed out and the static evaluator
// stood in — its numbers are not comparable with a clean run's. See
// ObscuroAgent.getLeafEvalStats.
function reportLeafHealth() {
  const st = getLeafEvalStats();
  const total = st.engineLeaves + st.fallbackLeaves;
  const pct = total ? (100 * st.fallbackLeaves / total) : 0;
  console.log(`leaf evaluations: ${total} (${st.calls} engine calls), ` +
    `static-eval fallbacks ${st.fallbackLeaves} (${pct.toFixed(2)}%), ` +
    `engine-refused nodes ${st.refusedNodes}, truncated rungs ${st.truncated}`);
  console.log(`  fallback causes by node: engine-said-nothing ${st.pvNullNodes}, ` +
    `fewer-lines-than-asked ${st.pvShortNodes}, lines-but-not-our-moves ${st.unmappedNodes}` +
    (st.engineUnavailable ? `, engine-unavailable ${st.engineUnavailable}` : ''));
  if (pct > 0.5) {
    console.log('  ^ MEASUREMENT DEGRADED. Stockfish was timing out — almost always because');
    console.log('    something else heavy was running on this machine. Re-run it alone.');
  }
}

const actionKey = a => (a.type === 'castle' ? `O-${a.side}` : `${a.from}${a.to}${a.payload?.promote ?? ''}`);

// THE AGENT MUST NOT COMMIT ITS OWN PICK HERE, and this cost every measurement
// this script has ever produced.
//
// ObscuroAgent.chooseAction ends by calling game.onActionCommitted with the move
// it chose (vendor/obscuro/src/ObscuroAgent.js), which is right in a real game:
// the move it chose is the move that gets played. In THIS harness the pick is
// measured and thrown away, and the RECORDED move is what happens — so the
// belief was being advanced by our own move twice per turn, once with a move
// that was never played. The upstream comment says what that does: "committing
// an action other than the one actually played silently corrupts the belief
// (fatally so for the exact tracker)". It gave up on ply 2 of every replay, and
// every arm then ran on the heuristic particle fallback instead of P — which is
// not the thing α and β are knobs on at all.
//
// So the agent gets a game whose commit hook does nothing, and replayArm commits
// the recorded move itself, once. The `|P| exact` count in the report below is
// the canary that caught this and the reason it is printed on every run: an arm
// where exactness is lost early is an arm measuring the fallback belief.
const REPLAY_GAME = { ...FogChess, onActionCommitted() {} };
setGame(REPLAY_GAME);

/**
 * Replay one recorded game from one seat, asking the agent for a move at each of
 * that seat's turns but ALWAYS committing the RECORDED move. The agent's choice
 * is measured, never played: both arms must walk the identical position stream,
 * and a diverging game would compare two different sets of positions.
 */
async function replayArm(sess, seat, spec, seed) {
  const { alpha, prior, sfDepth, rounds, reach } = spec;
  // Fresh identity → fresh belief trackers (see protocol note 2). The α is read
  // in the ExactBelief constructor, so it must be set BEFORE the first sample.
  const players = JSON.parse(JSON.stringify(sess.params.players));
  setBeliefSampleAlphaForSeat(seat, alpha ?? 0);
  setMovePriorForSeat(seat, prior ?? null);
  setBeliefReachWeightingForSeat(seat, reach ?? null);
  const agent = new ChessObscuroAgent({
    rng: mulberry32(seed), ...knobs,
    ...(sfDepth ? { sfDepth } : {}), ...(rounds ? { maxRounds: rounds } : {}),
  });
  let searchMs = 0;

  let state = FogChess.createInitialState(players, { ...sess.params.config, aiTimeMs: null, difficulty: dial });
  const picks = [];
  const log = sess.log ?? [];
  for (let i = 0; i < Math.min(log.length, maxPlies); i++) {
    const pa = log[i].playerActions?.[0];
    if (!pa?.action) break;
    if (pa.playerId === seat) {
      const obs = FogChess.getVisibleState(state, seat);
      // The engine computes legal actions on the TRUE state (GameEngine.js:168),
      // so a move blocked by an unseen piece is never offered. Match that, or the
      // agent gets an action set production would never hand it.
      const legal = FogChess.getLegalActions({ ...state, activePlayers: [seat] }, seat);
      const tm = Date.now();
      const chosen = legal.length > 1 ? await agent.chooseAction(obs, legal) : legal[0];
      searchMs += Date.now() - tm;
      // |P| AT THIS DECISION — how much the seat did not know when it moved.
      // Read AFTER chooseAction on purpose: beliefPopulation prepares the same
      // trackers sampleWorlds does and is idempotent within a turn (turnKey), so
      // reading it second is a pure observation and cannot perturb the search
      // that produced `chosen`. `total` is null when exact tracking was lost
      // (P outgrew CAP, or the time guard tripped) — recorded as null, not 0,
      // and excluded from the fit rather than silently entered as a small |P|.
      const pop = FogChess.beliefPopulation(obs, seat);
      if (VERBOSE) process.stdout.write(`      ply ${i} search ${Date.now() - tm} ms (worlds ${agent.lastAnalysis?.worlds}, |P| ${pop?.total ?? 'lost'})\n`);
      picks.push({ ply: i, key: chosen ? actionKey(chosen) : null, pSize: pop?.exact ? pop.total : null });
      FogChess.onActionCommitted(obs, seat, pa.action);
    }
    state = FogChess.applyActions(state, [pa]);
  }
  setBeliefSampleAlphaForSeat(seat, null);
  setMovePriorForSeat(seat, null);
  setBeliefReachWeightingForSeat(seat, null);
  return { picks, searchMs };
}

/**
 * Reference scores for every legal move at one of `seat`'s turns, on the TRUE
 * board. Same evaluator the search uses at its leaves (Stockfish MultiPV, hung
 * kings pinned at −KING_HANG, clamped) but at a depth the search never affords.
 * Identical for both arms, so it is computed once per position and shared.
 */
const refEval = makeChessLeafEval(refDepth, 0);
async function referenceAt(state, seat) {
  const legal = FogChess.getLegalActions({ ...state, activePlayers: [seat] }, seat);
  if (legal.length <= 1) return null;
  const childStates = legal.map(a => FogChess.applyActions(state, [{ playerId: seat, action: a }]));
  const scores = await refEval(state, seat, legal, childStates);
  if (!scores) return null;
  const byKey = new Map();
  let best = -Infinity;
  legal.forEach((a, j) => {
    const s = scores[j] ?? -Infinity;
    byKey.set(actionKey(a), s);
    if (s > best) best = s;
  });
  return { byKey, best, n: legal.length };
}

/** Walk the game once with no agent, collecting the reference at each seat turn. */
async function referencesFor(sess, seat) {
  const players = JSON.parse(JSON.stringify(sess.params.players));
  let state = FogChess.createInitialState(players, sess.params.config);
  const refs = new Map();
  const log = sess.log ?? [];
  for (let i = 0; i < Math.min(log.length, maxPlies); i++) {
    const pa = log[i].playerActions?.[0];
    if (!pa?.action) break;
    if (pa.playerId === seat) {
      const tr = Date.now();
      const r = await referenceAt(state, seat);
      if (VERBOSE) process.stdout.write(`      ply ${i} REFERENCE ${Date.now() - tr} ms (${r?.n ?? 0} moves)\n`);
      if (r) refs.set(i, r);
    }
    state = FogChess.applyActions(state, [pa]);
  }
  return refs;
}

// --- grid mode: the leaf-depth / tree-size frontier ---------------------------
//
// `--grid 1:24,2:12,4:6,7:3` runs each `sfDepth:maxRounds` configuration as a
// single arm over the same positions and reports cp loss AND ms/move, so the two
// can be read against each other. This is the question the whole search-scale
// discussion turns on: 99% of a move is Stockfish leaf calls, they cost ~4 ms at
// depth 1 and ~200 ms at depth 7, and the paper spends the same budget on depth-1
// leaves and a tree ~100× larger. Whether that trade is right HERE — with a JS
// CFR and a WASM engine — is measurable, and this measures it. Read the output as
// a frontier: the configuration with the lowest cp loss per millisecond wins.
if (argv.includes('--grid')) {
  const configs = arg('grid', '1:24,2:12,4:6,7:3').split(',').map(s => {
    const [d, r] = s.split(':').map(Number);
    return { label: `depth ${d}, ${r} rounds`, sfDepth: d, rounds: r, alpha: 0 };
  });
  const acc = configs.map(() => ({ loss: [], ms: 0, moves: 0, top: 0 }));
  const tg = Date.now();
  for (let g = 0; g < games.length; g++) {
    for (const seat of seats) {
      const refs = await referencesFor(games[g].sess, seat);
      if (!refs.size) continue;
      // The reference sweep is a deliberate deep search; only the ARMS' leaf
      // health says whether the measurement itself was degraded.
      resetLeafEvalStats();
      const seed = seed0 + g * 977 + (seat === 'white' ? 0 : 1);
      for (let c = 0; c < configs.length; c++) {
        const { picks, searchMs } = await replayArm(games[g].sess, seat, configs[c], seed);
        acc[c].ms += searchMs;
        for (const { ply, key } of picks) {
          const ref = refs.get(ply);
          if (!ref || !key) continue;
          const s = ref.byKey.get(key);
          if (s === undefined) continue;
          acc[c].loss.push(ref.best - s);
          if (ref.best - s === 0) acc[c].top++;
          acc[c].moves++;
        }
      }
      process.stdout.write(`  game ${g + 1}/${games.length} ${seat}: ${((Date.now() - tg) / 1000).toFixed(0)}s\n`);
    }
  }
  const mn = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
  const md = xs => { const s = [...xs].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
  console.log(`\ngrid over ${games.length} games, seats ${seats.join('+')}, worlds ${knobs.particles}, reference depth ${refDepth}`);
  console.log('config'.padEnd(22) + 'moves  mean cp  median cp  ms/move  best-move%');
  for (let c = 0; c < configs.length; c++) {
    const a = acc[c];
    console.log(configs[c].label.padEnd(22) +
      String(a.moves).padStart(5) +
      mn(a.loss).toFixed(1).padStart(9) +
      md(a.loss).toFixed(1).padStart(11) +
      (a.ms / Math.max(1, a.moves)).toFixed(0).padStart(9) +
      (100 * a.top / Math.max(1, a.moves)).toFixed(1).padStart(11));
  }
  console.log('\nLower cp loss at lower ms/move is strictly better; anything else is a trade.');
  reportLeafHealth();
  await stockfishQuit?.();
  process.exit(0);
}

// --- run ---------------------------------------------------------------------

const stats = { a: [], b: [], diffs: [], aTop: 0, bTop: 0, n: 0, same: 0, rows: [], pMismatch: 0 };
const blankHealth = () => ({ engineLeaves: 0, fallbackLeaves: 0, pvNullNodes: 0, pvShortNodes: 0, unmappedNodes: 0, refusedNodes: 0 });
const healthA = blankHealth(), healthB = blankHealth();
const accStats = (into, st) => { for (const k of Object.keys(into)) into[k] += st[k] ?? 0; };
const t0 = Date.now();

for (let g = 0; g < games.length; g++) {
  const { sess, file } = games[g];
  for (const seat of seats) {
    const refs = await referencesFor(sess, seat);
    if (!refs.size) continue;
    resetLeafEvalStats();
    // Same seed for both arms: common random numbers, so the streams start
    // identical and only α/π can pull them apart.
    const seed = seed0 + g * 977 + (seat === 'white' ? 0 : 1);
    // Leaf health PER ARM: a shared counter hides the case where one arm's
    // configuration is what degrades the evaluator, which is exactly the case
    // that would invalidate the comparison.
    resetLeafEvalStats();
    const A = (await replayArm(sess, seat, arm.a, seed)).picks;
    accStats(healthA, getLeafEvalStats());
    resetLeafEvalStats();
    const B = (await replayArm(sess, seat, arm.b, seed)).picks;
    accStats(healthB, getLeafEvalStats());
    const byPlyB = new Map(B.map(p => [p.ply, p]));
    for (const { ply, key: ka, pSize: pa } of A) {
      const ref = refs.get(ply);
      const b = byPlyB.get(ply);
      const kb = b?.key;
      if (!ref || !ka || !kb) continue;
      const sa = ref.byKey.get(ka), sb = ref.byKey.get(kb);
      if (sa === undefined || sb === undefined) continue;
      const la = clipLoss(ref.best - sa), lb = clipLoss(ref.best - sb);
      stats.a.push(la); stats.b.push(lb); stats.diffs.push(la - lb);
      if (la === 0) stats.aTop++;
      if (lb === 0) stats.bTop++;
      if (ka === kb) stats.same++;
      stats.n++;
      // |P| is a property of the OBSERVATION HISTORY, not of α or β — both arms
      // see the identical position stream, so the two readings must agree. They
      // can only diverge if one arm lost exactness where the other did not
      // (the CAP/time-guard boundary); counted rather than averaged away, since
      // a run with many mismatches is measuring two different belief regimes.
      if (pa != null && b.pSize != null && pa !== b.pSize) stats.pMismatch++;
      stats.rows.push({ game: file, seat, ply, pSize: pa, pSizeB: b.pSize, la, lb, diff: la - lb, same: ka === kb });
    }
    process.stdout.write(`  ${file.slice(0, 8)} ${seat}: ${stats.n} positions, ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
  }
}

// --- report ------------------------------------------------------------------

const mean = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const median = xs => { const s = [...xs].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
const stderr = xs => {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1) / xs.length);
};

const d = mean(stats.diffs), se = stderr(stats.diffs);
const wins = stats.diffs.filter(x => x < 0).length, losses = stats.diffs.filter(x => x > 0).length;

console.log(`\narm ${armName}: ${arm.a.label} (A) vs ${arm.b.label} (B)`);
console.log(`clip ${CLIP > 0 ? CLIP + ' cp (winsorised)' : 'off (raw mean)'}`);
console.log(`${games.length} games, seats ${seats.join('+')}, dial ${dial} (leaf depth ${Math.max(1, Math.round(2 + dial/100*5))}), worlds ${knobs.particles}, rounds ${knobs.maxRounds}, reference depth ${refDepth}, seed ${seed0}`);
console.log(`positions: ${stats.n}   identical move chosen: ${stats.same} (${(100 * stats.same / stats.n).toFixed(1)}%)`);
console.log(`mean cp loss   A ${mean(stats.a).toFixed(1)}   B ${mean(stats.b).toFixed(1)}`);
console.log(`median cp loss A ${median(stats.a).toFixed(1)}   B ${median(stats.b).toFixed(1)}`);
console.log(`reference-best move played:  A ${(100 * stats.aTop / stats.n).toFixed(1)}%   B ${(100 * stats.bTop / stats.n).toFixed(1)}%`);
console.log(`PAIRED DIFFERENCE (A − B, negative favours A): ${d.toFixed(2)} ± ${se.toFixed(2)} cp  (z = ${(d / se).toFixed(2)})`);
console.log(`positions where A lost less: ${wins}, where B lost less: ${losses}, tied: ${stats.n - wins - losses}`);

// THE SIGN TEST IS THE STATISTIC TO READ, not the mean. cp loss has a brutal
// tail — a single blundered queen is 900 while the median position is ~75 — so
// the mean's standard error is dominated by a handful of positions and barely
// improves with more data (going 220 → 1345 positions moved SE 8.1 → 17.6,
// because the extra positions were late-game ones with bigger swings). The
// count of which arm lost less is bounded per position and cannot be swamped
// that way. Report both; believe this one.
const dec = wins + losses;
if (dec > 0) {
  const z = (wins - dec / 2) / Math.sqrt(dec * 0.25);
  console.log(`SIGN TEST over the ${dec} decisive positions: A better ${(100 * wins / dec).toFixed(1)}%  (z = ${z.toFixed(2)})`);
}
// --- does the effect depend on how much is hidden? ---------------------------
//
// THE HYPOTHESIS THIS TESTS. Every arm above pools two regimes: turns where the
// board is effectively known (|P| small, often 1) and turns where it is not
// (|P| in the thousands). A belief change CANNOT move a decision in the first
// regime — there is nothing to be uncertain about — so those positions enter the
// paired mean as structural zeros and dilute whatever signal the second regime
// carries. An aggregate null is therefore consistent with a real effect that
// only exists where |P| is large, and the aggregate cannot tell the two apart.
//
// So |P| is RECORDED PER POSITION and the paired difference is REGRESSED on it,
// rather than the positions being cut into strata: binning throws away the
// ordering, needs edges chosen by the person hoping for a result, and estimates
// each bin from a fraction of the data. The slope uses every position.
//
// Read the SLOPE, not the intercept. Negative slope = arm A gains as uncertainty
// grows, which is the shape "the belief matters where the belief matters".
//
// TWO FITS, because |P| spans ~5 orders of magnitude and a raw-|P| fit is a fit
// to its few largest values: the raw fit is the one asked for and the log10 fit
// is the one whose leverage is spread evenly. If they disagree, believe neither
// and look at the decile diagnostic underneath them.
//
// SEs ARE HETEROSKEDASTICITY-ROBUST (HC1), which is not pedantry here: the
// hypothesis IS that the diff's variance grows with |P|, so classical OLS SEs
// are wrong under exactly the alternative being tested.
function ols(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  const mx = mean(xs), my = mean(ys);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  if (!(sxx > 0)) return null;
  const slope = sxy / sxx, intercept = my - slope * mx;
  // HC1 sandwich: Var(b) = (Σ dx²e²) / Sxx² · n/(n-2).
  let meat = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    meat += (xs[i] - mx) ** 2 * e * e;
  }
  const se = Math.sqrt((meat / (sxx * sxx)) * (n / (n - 2)));
  return { n, slope, intercept, se, t: slope / se, r: sxy / Math.sqrt(sxx * syy) };
}
const fmtFit = (label, fit, unit) => {
  if (!fit) { console.log(`  ${label.padEnd(34)} (too few points)`); return; }
  console.log(`  ${label.padEnd(34)} ${fit.slope.toFixed(3).padStart(9)} ± ${fit.se.toFixed(3).padStart(7)} ${unit}` +
    `   t = ${fit.t.toFixed(2).padStart(5)}   r = ${fit.r.toFixed(3)}   n = ${fit.n}`);
};

const withP = stats.rows.filter(r => r.pSize != null);
const lostExact = stats.rows.length - withP.length;
console.log(`\n=== paired difference vs |P| (the belief's own size at each decision) ===`);
console.log(`positions with exact |P|: ${withP.length} of ${stats.rows.length}` +
  (lostExact ? `  (${lostExact} excluded: exact tracking lost)` : '') +
  (stats.pMismatch ? `  [!] ${stats.pMismatch} positions where the two arms' |P| disagreed` : ''));
// A high loss rate is not a cosmetic gap in the fit — it means the arms spent
// those positions on the HEURISTIC particle belief, which α and β do not even
// apply to, so the run is partly an A/B of two identical configurations. It is
// also right-censoring: exactness is lost at the LARGEST |P|, i.e. precisely the
// positions this regression is about, which biases the slope toward zero.
// Raise the cap and the guard together to buy coverage back:
//   --set chess.EXACT_BELIEF_CAP=1500000 --set chess.EXACT_BELIEF_TIME_GUARD_MS=30000
if (lostExact / Math.max(1, stats.rows.length) > 0.2) {
  console.log(`  ^ ${(100 * lostExact / stats.rows.length).toFixed(0)}% OF POSITIONS RAN ON THE FALLBACK BELIEF, not P.`);
  console.log('    α and β are knobs on the exact belief, so those positions cannot show an effect,');
  console.log('    and they are the high-|P| ones. Re-run with a larger chess.EXACT_BELIEF_CAP.');
}
if (withP.length >= 3) {
  const ps = withP.map(r => r.pSize).sort((x, y) => x - y);
  const q = f => ps[Math.min(ps.length - 1, Math.floor(f * ps.length))];
  console.log(`|P|: min ${ps[0]}  p25 ${q(0.25)}  median ${q(0.5)}  p75 ${q(0.75)}  p95 ${q(0.95)}  max ${ps[ps.length - 1]}` +
    `   |P|=1 (nothing hidden): ${ps.filter(x => x === 1).length}`);

  const x = withP.map(r => r.pSize), lx = withP.map(r => Math.log10(r.pSize)), y = withP.map(r => r.diff);
  const raw = ols(x, y), lg = ols(lx, y);
  console.log(`\ncp-loss difference (A − B; a NEGATIVE slope means A gains as uncertainty grows):`);
  // Raw slope is cp per world, which prints as zeros — scaled to per-1000 so the
  // number is readable without changing the fit.
  fmtFit('slope on raw |P| (per 1000 worlds)', raw && { ...raw, slope: raw.slope * 1000, se: raw.se * 1000 }, 'cp');
  fmtFit('slope on log10|P| (per decade)', lg, 'cp');

  // The sign counterpart. The script's own header explains why the mean is the
  // weaker statistic here (one blundered queen is 900 cp), and the same tail
  // dominates a mean-based slope. This is the sign test made continuous in |P|:
  // a linear probability model of "did A lose less", over decisive positions only
  // (ties are structural zeros and would just flatten it toward 0.5).
  const dec2 = withP.filter(r => r.diff !== 0);
  if (dec2.length >= 3) {
    const sx = dec2.map(r => r.pSize), slx = dec2.map(r => Math.log10(r.pSize));
    const sy = dec2.map(r => (r.diff < 0 ? 1 : 0));
    const sraw = ols(sx, sy), slg = ols(slx, sy);
    console.log(`\nP(A lost less) over the ${dec2.length} decisive positions — 0.5 is no effect:`);
    fmtFit('slope on raw |P| (per 1000 worlds)', sraw && { ...sraw, slope: sraw.slope * 1000, se: sraw.se * 1000 }, '   ');
    fmtFit('slope on log10|P| (per decade)', slg, '   ');
    if (slg) console.log(`  fitted P(A better) at |P|=1: ${(slg.intercept).toFixed(3)}   at |P|=10k: ${(slg.intercept + 4 * slg.slope).toFixed(3)}`);
  }

  // LINEARITY CHECK ONLY — the fits above are on raw |P|, not on these bins.
  // Deciles are here so a curved or single-point-driven relationship is visible
  // rather than being reported as a slope.
  console.log(`\nlinearity check (deciles of |P|; the fits above do NOT use these):`);
  const byP = [...withP].sort((a, b) => a.pSize - b.pSize);
  const per = Math.ceil(byP.length / 10);
  console.log('  |P| range'.padEnd(24) + 'n     mean Δcp   disagreed   A better');
  for (let i = 0; i < byP.length; i += per) {
    const b = byP.slice(i, i + per);
    const d2 = b.filter(r => r.diff !== 0);
    console.log(`  ${b[0].pSize}–${b[b.length - 1].pSize}`.padEnd(24) +
      String(b.length).padStart(4) +
      mean(b.map(r => r.diff)).toFixed(1).padStart(11) +
      `${(100 * d2.length / b.length).toFixed(0)}%`.padStart(12) +
      (d2.length ? `${(100 * d2.filter(r => r.diff < 0).length / d2.length).toFixed(0)}%` : '—').padStart(10));
  }
}

// Per-position rows, so this run can be re-analysed (or pooled with another
// arm's) without paying for the reference sweep again — which is most of the
// wall clock.
const dumpPath = arg('dump', null);
if (dumpPath) {
  const { writeFileSync } = await import('node:fs');
  const head = 'arm,a_label,b_label,game,seat,ply,p_size,p_size_b,loss_a,loss_b,diff,same\n';
  const body = stats.rows.map(r => [
    armName, JSON.stringify(arm.a.label), JSON.stringify(arm.b.label), JSON.stringify(r.game), r.seat, r.ply,
    r.pSize ?? '', r.pSizeB ?? '', r.la, r.lb, r.diff, r.same ? 1 : 0,
  ].join(',')).join('\n');
  writeFileSync(dumpPath, head + body + '\n');
  console.log(`\nper-position rows → ${dumpPath}`);
}

console.log(`\nOnly the ${stats.n - stats.same} positions where the arms disagreed can carry signal.`);
for (const [label, h] of [[arm.a.label, healthA], [arm.b.label, healthB]]) {
  const tot = h.engineLeaves + h.fallbackLeaves;
  console.log(`leaf health [${label}]: ${tot} leaves, static-eval fallbacks ` +
    `${h.fallbackLeaves} (${tot ? (100 * h.fallbackLeaves / tot).toFixed(2) : '0'}%) — ` +
    `nothing ${h.pvNullNodes}, short ${h.pvShortNodes}, unmapped ${h.unmappedNodes}, refused ${h.refusedNodes}`);
}
if (armName !== 'null') {
  console.log('Run `--arm null` before believing any of this: the control must come back ~0.');
}

await stockfishQuit?.();
