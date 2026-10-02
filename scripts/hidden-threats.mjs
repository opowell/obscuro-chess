// ---------------------------------------------------------------------------
// Does the agent defend against threats it cannot see?
//
//   node scripts/hidden-threats.mjs --sessions <corpus> [--arms base,mix] [--probes 200]
//        [--dial 30] [--workers N] [--dump file.csv]
//
// THE PROBES ARE MINED, NOT CHOSEN. Every position in the corpus where the side to
// move has its king capturable by a piece it cannot see (the threat is on the
// true board, not on the side's own view), and at least one legal move removes
// the threat. One probe per game, spread over the corpus. Human players in the
// Chess.com crawl leave the king capturable in 35% of them; that is printed as the
// baseline.
//
// Each probe is replayed from the start exactly as in play: the belief advanced
// at every turn with the RECORDED moves, the agent searching once at the side's
// previous turn (so the next search carries that tree over, as a real game does)
// and then deciding at the probe. Scored on the true board:
//
//   hung     the chosen move leaves the king capturable — the blunder
//   defend   how much of the search's root strategy (before purification) is on
//            moves that leave the king safe — what the search WANTED to do
//
// ARMS, each a settings overlay applied in the worker (see ARMS below):
//   base    the shipped search
//   mix     always sample from the purified mixed strategy when there is more
//           than one world (safePmaxThreshold above any pmax)
//   pure    the safety test and fresh-world alternate value before 2026-10-02
//           (p_max alone; ṽ(h) from the best child), which almost never mixed
//   bestchild only the fresh-world alternate value as before
//   fog0, balseen  other fog value models (scripts/fog-model-arms.mjs)
//
// Also reported per arm: the belief's posterior mass on positions where the king
// is capturable, the search's root reach on such worlds, and the chosen move's
// expected-score loss on the true board against the engine at depth 12.
//
// Results, 2026-10-02 (300 probes, dial 30): see NEXT STEPS.md.
//
// WHAT THIS DOES NOT MEASURE: what defending costs where there was no threat.
// That needs games (strength-belief.mjs), and a fix should be run through both.
// ---------------------------------------------------------------------------

import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FogChess } from '../src/FogChess.js';
import { isAttackedBy } from '../src/board.js';
import { FOG_MODEL_ARMS } from './fog-model-arms.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

// A settings overlay per arm. Applied with setOverrides inside the worker, so
// each worker thread holds exactly one arm's settings while it runs a probe.
// `agent` holds agent options rather than settings (fog value models, below).
const ARMS = {
  base: {},
  mix: { search: { SEARCH_DEFAULTS: { safePmaxThreshold: 2 } } },
  pure: { agent: { freshAlt: 'bestChild', marginSafe: false } },
  bestchild: { agent: { freshAlt: 'bestChild' } },
  ...Object.fromEntries(Object.entries(FOG_MODEL_ARMS).map(([k, v]) => [k, { agent: v }])),
};


// 'a+b' combines arms: settings merged one level deep, agent options merged.
function armOverlay(name) {
  const out = {};
  for (const part of name.split('+')) {
    const o = ARMS[part];
    if (!o) throw new Error(`unknown arm ${part}; known: ${Object.keys(ARMS).join(', ')} (combine with +)`);
    for (const [k, v] of Object.entries(o)) out[k] = { ...(out[k] ?? {}), ...v };
  }
  return out;
}

async function main() {
  const { makeArgReader } = await import('../src/cli.js');
  const argv = process.argv.slice(2);
  const arg = makeArgReader(argv);
  const sessions = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
  const arms = arg('arms', 'base,mix').split(',');
  for (const a of arms) armOverlay(a);   // throws on an unknown arm
  const nProbes = Number(arg('probes', '200'));
  const dial = Number(arg('dial', '30'));
  const workers = Number(arg('workers', String(Math.max(1, availableParallelism() - 1))));
  const dumpPath = arg('dump', null);

  const probes = await mine(sessions, nProbes);
  console.log(`${probes.length} probes; humans left the king capturable in ` +
    `${probes.filter(p => p.humanHung).length} (${(100 * probes.filter(p => p.humanHung).length / probes.length).toFixed(1)}%)`);

  const tasks = [];
  for (const p of probes) for (const arm of arms) tasks.push({ id: tasks.length, arm, ...p });
  const results = new Array(tasks.length);
  const t0 = Date.now();
  let next = 0, done = 0, lastReport = 0;
  await new Promise((resolve, reject) => {
    const pool = Array.from({ length: Math.min(workers, tasks.length) }, () =>
      new Worker(fileURLToPath(import.meta.url), { workerData: { sessions, dial } }));
    const feed = w => { if (next < tasks.length) w.postMessage(tasks[next++]); else w.terminate(); };
    for (const w of pool) {
      w.on('error', reject);
      w.on('message', r => {
        results[r.id] = r;
        done++;
        const now = Date.now();
        if (done === tasks.length || now - lastReport > 30000) {
          lastReport = now;
          const min = (now - t0) / 60000;
          console.log(`  ${done}/${tasks.length} decisions, ${min.toFixed(1)} min` +
            (done < tasks.length ? `, ~${(min / done * (tasks.length - done)).toFixed(0)} min to go` : ''));
        }
        if (done === tasks.length) resolve();
        feed(w);
      });
      feed(w);
    }
  });

  console.log(`\ndial ${dial}; ${probes.length} probes per arm`);
  const byArm = new Map(arms.map(a => [a, results.filter(r => r.arm === a)]));
  for (const [a, rs] of byArm) {
    const ok = rs.filter(r => !r.error);
    const hung = ok.filter(r => r.hung).length;
    const mean = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
    console.log(`  ${a.padEnd(6)} hung ${hung}/${ok.length} (${(100 * hung / ok.length).toFixed(1)}%)  ` +
      `mean strategy on safe moves ${(100 * mean(ok.map(r => r.defend))).toFixed(1)}%  ` +
      `pmax=1 in ${ok.filter(r => r.pmax >= 0.999).length}  safe to mix in ${ok.filter(r => r.safe).length}  errors ${rs.length - ok.length}`);
    console.log(`         true-board expected-score loss per decision: mean ${mean(ok.map(r => r.loss)).toFixed(2)} points; ` +
      `when hung: mean ${mean(ok.filter(r => r.hung).map(r => r.loss)).toFixed(1)}; ` +
      `best defence worth ≥ 10%: ${ok.filter(r => r.bestSafe >= 10).length} probes, hung in ${ok.filter(r => r.bestSafe >= 10 && r.hung).length}`);
    const known = ok.filter(r => r.beliefThreat != null);
    const band = (lo, hi) => known.filter(r => r.beliefThreat >= lo && r.beliefThreat < hi);
    console.log(`         belief's mass on a threat: mean ${(100 * mean(known.map(r => r.beliefThreat))).toFixed(1)}%, ` +
      `search reach on threat worlds: mean ${(100 * mean(ok.map(r => r.searchThreat))).toFixed(1)}%`);
    for (const [lo, hi] of [[0, 0.05], [0.05, 0.2], [0.2, 0.5], [0.5, 1.01]]) {
      const b = band(lo, hi);
      if (b.length) console.log(`         belief threat ${(100 * lo).toFixed(0)}–${Math.min(100, 100 * hi).toFixed(0)}%: ${b.length} probes, hung ${(100 * b.filter(r => r.hung).length / b.length).toFixed(0)}%, ` +
        `strategy on safe moves ${(100 * mean(b.map(r => r.defend))).toFixed(0)}%`);
    }
  }
  // Paired: same probes, same seeds.
  for (const b of arms.slice(1)) {
    const a = arms[0];
    const A = new Map(byArm.get(a).map(r => [r.key, r]));
    let onlyA = 0, onlyB = 0; const d = [];
    for (const r of byArm.get(b)) {
      const q = A.get(r.key); if (!q || q.error || r.error) continue;
      if (q.hung && !r.hung) onlyA++; if (r.hung && !q.hung) onlyB++;
      d.push(r.loss - q.loss);
    }
    const m = d.reduce((x, y) => x + y, 0) / d.length;
    const se = Math.sqrt(d.reduce((x, y) => x + (y - m) ** 2, 0) / (d.length - 1) / d.length);
    console.log(`  paired ${b} vs ${a}: hung under ${a} only ${onlyA}, under ${b} only ${onlyB}; ` +
      `expected-score loss ${b} − ${a} ${m.toFixed(2)} ± ${se.toFixed(2)} points`);
  }
  if (dumpPath) {
    writeFileSync(dumpPath, 'arm,game,ply,seat,hung,defend,pmax,human_hung,belief_threat,search_threat,p_size\n' +
      results.map(r => [r.arm, r.gi, r.ply, r.seat, r.hung ? 1 : 0, r.defend?.toFixed(4), r.pmax, r.humanHung ? 1 : 0,
        r.beliefThreat?.toFixed(4) ?? '', r.searchThreat?.toFixed(4), r.pSize ?? ''].join(',')).join('\n') + '\n');
  }
  process.exit(0);
}

// The probes: one hidden, defensible king threat per game, from games spread
// evenly over the corpus.
async function mine(sessions, n) {
  const { loadCorpus } = await import('../src/corpus.js');
  const { games } = loadCorpus(sessions, {});
  const out = [];
  // Walk the corpus with a stride coprime to its length, so the games examined
  // are spread over it, until n probes are found or every game has been seen.
  const L = games.length;
  let stride = Math.max(1, Math.floor(L / Math.max(1, n * 3)));
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  while (gcd(stride, L) !== 1) stride++;
  for (let k = 0, gi = 0; k < L && out.length < n; k++, gi = (gi + stride) % L) {
    const p = firstThreat(games[gi].sess);
    if (p) out.push({ gi, key: `${gi}|${p.ply}`, ...p });
  }
  return out;
}

const kingSq = (b, c) => Object.keys(b).find(s => b[s]?.ownerId === c && b[s].type === 'king');
const other = c => (c === 'white' ? 'black' : 'white');
function kingHangs(st, me, action) {
  const n = FogChess.applyActions(st, [{ playerId: me, action }]);
  const k = kingSq(n.board, me);
  return !k || isAttackedBy(n.board, k, other(me));
}

function firstThreat(sess) {
  let st = FogChess.createInitialState(sess.params.players, sess.params.config);
  const log = sess.log ?? [];
  for (let i = 0; i < log.length; i++) {
    const pa = log[i].playerActions?.[0];
    if (!pa?.action || FogChess.getResult?.(st)) return null;
    const me = pa.playerId, k = kingSq(st.board, me);
    // A probe needs a previous turn of the same seat to search at (ply ≥ 2).
    if (i >= 2 && k && isAttackedBy(st.board, k, other(me))
        && !isAttackedBy(FogChess.getVisibleState(st, me).board, k, other(me))) {
      const legal = FogChess.getLegalActions({ ...st, activePlayers: [me] }, me);
      if (legal.some(a => !kingHangs(st, me, a))) {
        return { ply: i, seat: me, humanHung: kingHangs(st, me, pa.action) };
      }
    }
    try { st = FogChess.applyActions(st, [pa]); } catch { return null; }
  }
  return null;
}

async function worker() {
  const { loadCorpus } = await import('../src/corpus.js');
  const OA = await import('../src/ObscuroAgent.js');
  const { ChessObscuroAgent, setGame, makeChessLeafEval } = OA;
  // The true-board yardstick: every legal move valued by the engine at depth 12
  // as an expected result in [−1, +1], whatever scale the arm searches on.
  const refEval = makeChessLeafEval(12, 0);
  const reference = async (st, seat, legal) => {
    OA.setLeafValue?.('wdl');
    try {
      const children = legal.map(a => FogChess.applyActions(st, [{ playerId: seat, action: a }]));
      return await refEval(st, seat, legal, children);
    } finally { OA.setLeafValue?.(null); }
  };
  const { getExactBelief, toBoardObject } = await import('../src/exactBelief.js');
  const { setOverrides, resetSettings } = await import('../src/config.js');
  const { setFreshHash, setAutoRecycle, recycleEngine, setCacheDir } = await import('../src/stockfish.js');
  // Same determinism protocol as move-quality.mjs: fresh hash, a respawned engine
  // per decision, and a cache of its own that nothing without fresh hash writes to.
  const cacheDir = join(HERE, '..', 'vendor', 'stockfish', 'move-quality');
  mkdirSync(cacheDir, { recursive: true });
  setCacheDir(cacheDir);
  setFreshHash(true);
  setAutoRecycle(false);
  // The agent must not commit its own pick: the RECORDED move is what happens
  // (see move-quality.mjs, REPLAY_GAME).
  const REPLAY_GAME = { ...FogChess, onActionCommitted() {} };
  setGame(REPLAY_GAME);
  const { games } = loadCorpus(workerData.sessions, {});
  const knobs = { timeBudgetMs: 0, particles: 16, maxRounds: 6, maxInfosets: 1200, expandPerRound: 10, cfrPerRound: 6, finalCfr: 50 };
  const mulberry32 = a => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

  parentPort.on('message', async (task) => {
    const { id, arm, gi, ply, seat, key, humanHung } = task;
    try {
      resetSettings();
      // The belief must not depend on the clock (see move-quality.mjs).
      const { agent: agentOpts, ...overlay } = armOverlay(arm);
      setOverrides({ ...overlay, chess: { ...(overlay.chess ?? {}), EXACT_BELIEF_TIME_GUARD_MS: 3600000 } });
      await recycleEngine();
      const sess = games[gi].sess;
      const players = JSON.parse(JSON.stringify(sess.params.players));
      let st = FogChess.createInitialState(players, { ...sess.params.config, aiTimeMs: null, difficulty: workerData.dial });
      const agent = new ChessObscuroAgent({ rng: mulberry32(1000 + gi), ...knobs, ...agentOpts });
      let chosen = null;
      for (let i = 0; i <= ply; i++) {
        const pa = sess.log[i].playerActions[0];
        if (pa.playerId === seat) {
          const obs = FogChess.getVisibleState(st, seat);
          const legal = FogChess.getLegalActions({ ...st, activePlayers: [seat] }, seat);
          if (i === ply || i === ply - 2) {
            const pick = await agent.chooseAction(obs, legal);
            if (i === ply) { chosen = pick; break; }
            // Carry the tree over along the move that was really played.
            const carry = agent._carry.get(seat);
            if (carry) carry.actionKey = agent._key(pa.action);
          } else {
            FogChess.beliefPopulation(obs, seat);
          }
          FogChess.onActionCommitted(obs, seat, pa.action);
          st = FogChess.applyActions(st, [pa]);
          FogChess.onActionObserved(FogChess.getVisibleState(st, seat), seat);
          continue;
        }
        st = FogChess.applyActions(st, [pa]);
      }
      const hung = kingHangs(st, seat, chosen);
      const legalNow = FogChess.getLegalActions({ ...st, activePlayers: [seat] }, seat);
      const ref = await reference(st, seat, legalNow);
      const refBest = Math.max(...ref);
      const refChosen = ref[legalNow.findIndex(a => agent._key(a) === agent._key(chosen))];
      let refBestSafe = -1;
      legalNow.forEach((a, j) => { if (!kingHangs(st, seat, a)) refBestSafe = Math.max(refBestSafe, ref[j]); });
      // The share of the root strategy on moves that leave the king safe.
      const legal = FogChess.getLegalActions({ ...st, activePlayers: [seat] }, seat);
      const safeKeys = new Set(legal.filter(a => !kingHangs(st, seat, a)).map(a => agent._key(a)));
      const res = agent.lastAnalysis;
      let defend = 0, total = 0;
      for (const c of res?.candidates ?? []) { total += c.prob; if (safeKeys.has(c.key)) defend += c.prob; }
      // Where the danger was seen, if anywhere: the belief's posterior mass on
      // positions where our king is capturable, and the share of the search's
      // root reach those positions got.
      const opp = other(seat);
      const threatened = b => { const k = kingSq(b, seat); return !k || isAttackedBy(b, k, opp); };
      const tracker = getExactBelief(FogChess.getVisibleState(st, seat), seat);
      let beliefThreat = null;
      if (tracker.size) {
        beliefThreat = 0;
        const W = tracker.weights;
        for (let i = 0; i < tracker.size; i++) if (threatened(toBoardObject(tracker.P.view(i)))) beliefThreat += W[i];
      }
      const worlds = agent._carry.get(seat)?.tree?.worlds ?? [];
      let reachThreat = 0, reachAll = 0;
      for (const w of worlds) { reachAll += w.prob; if (threatened(w.node.state.board)) reachThreat += w.prob; }
      parentPort.postMessage({ id, arm, gi, ply, seat, key, humanHung, hung, defend: total ? defend / total : 0, pmax: res?.pmax ?? null, safe: res?.safe ?? null,
        beliefThreat, searchThreat: reachAll ? reachThreat / reachAll : 0, pSize: tracker.size || null,
        // Expected-score loss of the chosen move on the true board, points of %;
        // and what the best defence was worth, as an expected score in %.
        loss: (refBest - refChosen) * 50, bestSafe: (1 + refBestSafe) * 50 });
    } catch (e) {
      parentPort.postMessage({ id, arm, gi, ply, seat, key, humanHung, error: String(e?.stack ?? e) });
    }
  });
}

if (isMainThread) await main(); else await worker();
