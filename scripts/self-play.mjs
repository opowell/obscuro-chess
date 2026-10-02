// ---------------------------------------------------------------------------
// Seat-swapped self-play between two agent configurations, in parallel.
//
//   node scripts/self-play.mjs --arms fog,wdl [--pairs 50] [--dial 30] [--max-turns 200] [--workers N]
//
// Each pair is two games from the same seed, A as white then A as black: under
// fog white wins most games whatever the arms do (strength-belief.mjs measured
// 10–11 of 12), so only whole pairs mean anything. Reported: A's score over all
// games, and the pairs that were NOT split 1–1, which is where the signal lives.
//
// ARMS are agent option overlays (see ARMS). The search is the deterministic one
// move-quality.mjs uses (fixed rounds and tree size, no clock), with the belief's
// time guard raised so nothing depends on machine load.
// ---------------------------------------------------------------------------

import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

import { FOG_MODEL_ARMS } from './fog-model-arms.mjs';

// Arms combine with '+' (options merged), e.g. --arms bal,bal+pure.
const ARMS = {
  // Leaf values as shipped: expected result under fog (fogValueModel.js).
  fog: {},
  // Stockfish's full-information win/draw/loss, the value model of 2026-10-01/02.
  wdl: { leafScale: 'wdl' },
  // The safety test and fresh-world alternate value before 2026-10-02: p_max
  // alone, and ṽ(h) from the world's best child. Almost never mixes.
  pure: { freshAlt: 'bestChild', marginSafe: false },
  // Only the fresh-world alternate value as before (the margin test kept).
  bestchild: { freshAlt: 'bestChild' },
  ...FOG_MODEL_ARMS,
};

function armOpts(name) {
  return Object.assign({}, ...name.split('+').map(p => {
    if (!ARMS[p]) throw new Error(`unknown arm ${p}; known: ${Object.keys(ARMS).join(', ')} (combine with +)`);
    return ARMS[p];
  }));
}

async function main() {
  const { makeArgReader } = await import('../src/cli.js');
  const arg = makeArgReader(process.argv.slice(2));
  const [A, B] = arg('arms', 'fog,wdl').split(',');
  for (const a of [A, B]) armOpts(a);   // throws on an unknown arm
  const pairs = Number(arg('pairs', '50'));
  const dial = Number(arg('dial', '30'));
  const maxTurns = Number(arg('max-turns', '200'));
  const workers = Number(arg('workers', String(Math.max(1, availableParallelism() - 1))));

  const tasks = [];
  for (let p = 0; p < pairs; p++) for (const aIsWhite of [true, false]) tasks.push({ id: tasks.length, pair: p, aIsWhite });
  const results = [];
  const t0 = Date.now();
  let next = 0, done = 0, lastReport = 0;
  await new Promise((resolve, reject) => {
    const pool = Array.from({ length: Math.min(workers, tasks.length) }, () =>
      new Worker(fileURLToPath(import.meta.url), { workerData: { A, B, dial, maxTurns } }));
    const feed = w => { if (next < tasks.length) w.postMessage(tasks[next++]); else w.terminate(); };
    for (const w of pool) {
      w.on('error', reject);
      w.on('message', r => {
        results.push(r);
        done++;
        const now = Date.now();
        if (done === tasks.length || now - lastReport > 60000) {
          lastReport = now;
          const min = (now - t0) / 60000;
          console.log(`  ${done}/${tasks.length} games, ${min.toFixed(1)} min` +
            (done < tasks.length ? `, ~${(min / done * (tasks.length - done)).toFixed(0)} min to go` : ''));
        }
        if (done === tasks.length) resolve();
        feed(w);
      });
      feed(w);
    }
  });

  // A's score per game: 1 win, ½ draw or unfinished, 0 loss.
  const score = r => (r.error ? null : r.winner == null ? 0.5 : (r.winner === 'white') === r.aIsWhite ? 1 : 0);
  const ok = results.filter(r => !r.error);
  const s = ok.map(score);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  const se = Math.sqrt(s.reduce((a, x) => a + (x - mean) ** 2, 0) / (s.length - 1) / s.length);
  const whiteWins = ok.filter(r => r.winner === 'white').length;
  const byPair = new Map();
  for (const r of ok) { const a = byPair.get(r.pair) ?? []; a.push(score(r)); byPair.set(r.pair, a); }
  let aBoth = 0, bBoth = 0, split = 0;
  for (const v of byPair.values()) { if (v.length < 2) continue; const t = v[0] + v[1]; if (t === 2) aBoth++; else if (t === 0) bBoth++; else split++; }
  console.log(`\n${A} vs ${B}, dial ${dial}: ${ok.length} games (${results.length - ok.length} errors), white won ${whiteWins}`);
  console.log(`  ${A} scored ${(100 * mean).toFixed(1)}% ± ${(100 * se).toFixed(1)}`);
  console.log(`  pairs: ${A} won both ${aBoth}, ${B} won both ${bBoth}, split ${split}`);
  for (const [side, arm] of [['A', A], ['B', B]]) {
    const fog = ok.reduce((a, r) => a + (r.tally?.[side].fog ?? 0), 0), safe = ok.reduce((a, r) => a + (r.tally?.[side].safe ?? 0), 0);
    console.log(`  ${arm}: allowed to mix on ${safe} of ${fog} fog decisions (${fog ? (100 * safe / fog).toFixed(1) : '—'}%)`);
  }
  for (const r of results.filter(x => x.error).slice(0, 3)) console.log('  error:', r.error.split('\n')[0]);
  process.exit(0);
}

async function worker() {
  const { FogChess } = await import('../src/FogChess.js');
  const { ChessObscuroAgent } = await import('../src/ObscuroAgent.js');
  const { playMatch } = await import('../src/playMatch.js');
  const { setOverrides } = await import('../src/config.js');
  const { setFreshHash, setAutoRecycle, recycleEngine, setCacheDir } = await import('../src/stockfish.js');
  const dir = join(HERE, '..', 'vendor', 'stockfish', 'move-quality');
  mkdirSync(dir, { recursive: true });
  setCacheDir(dir);
  setFreshHash(true);
  setAutoRecycle(false);
  setOverrides({ chess: { EXACT_BELIEF_TIME_GUARD_MS: 3600000 } });
  const { A, B, dial, maxTurns } = workerData;
  const knobs = { timeBudgetMs: 0, particles: 16, maxRounds: 6, maxInfosets: 1200, expandPerRound: 10, cfrPerRound: 6, finalCfr: 50 };
  const mulberry32 = a => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  parentPort.on('message', async ({ id, pair, aIsWhite }) => {
    try {
      await recycleEngine();
      // Same seed for both games of a pair, and per seat, so the pair differs
      // only in which arm sits where.
      // Each agent also tallies its fog decisions and how many purification was
      // allowed to mix (the gadget reported safe), off its analysis record.
      const tally = { A: { fog: 0, safe: 0 }, B: { fog: 0, safe: 0 } };
      const make = (arm, seat, side) => {
        const agent = new ChessObscuroAgent({ rng: mulberry32(7919 * pair + (seat === 'white' ? 1 : 2)), ...knobs, ...armOpts(arm) });
        const choose = agent.chooseAction.bind(agent);
        agent.chooseAction = async (...args) => {
          agent.lastAnalysis = null;
          const a = await choose(...args);
          const an = agent.lastAnalysis;
          if (an?.worlds > 1) { tally[side].fog++; if (an.safe) tally[side].safe++; }
          return a;
        };
        return agent;
      };
      const agents = aIsWhite ? { white: make(A, 'white', 'A'), black: make(B, 'black', 'B') } : { white: make(B, 'white', 'B'), black: make(A, 'black', 'A') };
      // A fresh players array per game: belief trackers are keyed by its identity.
      const players = [{ id: 'white', name: 'White' }, { id: 'black', name: 'Black' }];
      const { result, plies } = await playMatch(agents, { game: FogChess, players, maxTurns, config: { difficulty: dial, aiTimeMs: null } });
      parentPort.postMessage({ id, pair, aIsWhite, winner: result?.winnerId ?? null, plies, tally });
    } catch (e) {
      parentPort.postMessage({ id, pair, aIsWhite, error: String(e?.stack ?? e) });
    }
  });
}

if (isMainThread) await main(); else await worker();
