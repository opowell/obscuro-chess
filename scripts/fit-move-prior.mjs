// ---------------------------------------------------------------------------
// Fit π — the move prior's tables — to how people ACTUALLY move under fog.
//
//   node scripts/fit-move-prior.mjs --sessions <corpus>          # fit + cross-validate
//   node scripts/fit-move-prior.mjs --sessions <corpus> --e2e    # + belief log-loss, held out
//   node scripts/fit-move-prior.mjs --sessions <corpus> --write  # update moveTables.js
//   node scripts/fit-move-prior.mjs --sessions <corpus> --rating # + rating slopes, held out
//   options: --folds 3  --l2 1e-6,1e-5,1e-4  --iters 3000  --max-games N  --e2e-games 150  --workers N
//
// WHAT IS FITTED. Everything the score is built from (movePrior.js): a value per
// (piece type, square) from the mover's side, per captured piece type, per
// promotion type, and for castling — 395 numbers, written to src/moveTables.js.
// Until 2026-10-01 those were hand-written (the evaluator's piece-square tables
// and textbook material values, with a hand-picked king value of 1000) and only
// nine weights on top of them were fitted. Now nothing in them is chosen by hand.
// The per-term multipliers in FITTED_WEIGHTS ship at 1; they are the handle the
// rating slopes turn.
//
// THE MODEL IS A CONDITIONAL LOGIT, as before:
//
//     π(m | p) = (1 − floor) · softmax_m ( Σ_e θ_e · count_e(p, m) ) + floor / |M|
//
// where count_e is how a move touches table entry e (movePrior.js moveEntries:
// +1 where a piece arrives, −1 where it leaves, +1 for what it captures, …). The
// objective is concave in θ; it is fitted by full-batch Adam on the sparse
// design, with an L2 penalty whose strength is CHOSEN BY CROSS-VALIDATION over
// --l2, and the floor is the mixture weight that maximizes the HELD-OUT
// likelihood — fitted too, not set.
//
// WHAT IT TRAINS ON. Every ply of every recorded fog game in the corpus: the true
// position, the full fog-legal move list from `genFogMoves` (production's own
// choice set — imported, not reimplemented), and which move was played. Both
// seats, all actor types, because π models "the opponent" generically.
//
// TWO THINGS TO BE CAREFUL ABOUT, both of which have bitten this subsystem:
//
//  • MOVE log-loss is not POSITION log-loss. The belief's gate is how much mass
//    it puts on the TRUE BOARD, and the observation filter already prices much of
//    what π would say. Only `--e2e` measures the one that matters, and it is the
//    gate for --write.
//  • Everything is cross-validated by GAME (never by ply — plies inside one game
//    are anything but independent), and --write refits on everything only AFTER
//    the held-out numbers have been printed.
// ---------------------------------------------------------------------------

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { genFogMoves, fromBoardObject } from '../src/exactBelief.js';
import { FogChess } from '../src/FogChess.js';
import { replayBelief, mean } from '../src/beliefCalibration.js';
import { loadCorpus, describeCorpus, ratingSpread } from '../src/corpus.js';
import {
  moveEntries, NUM_ENTRIES, ENTRY, NUM_FEATURES, FEATURE_NAMES, makeMovePrior,
  tablesFromVector, vectorFromTables, UNIFORM_PRIOR, FITTED_WEIGHTS,
  RATING_PIVOT, RATING_SCALE, ratingZ,
} from '../src/movePrior.js';
import { MOVE_TABLES } from '../src/moveTables.js';
import { renderTables } from './move-tables-format.mjs';
import { applyCliSettings, maybePrintConfig, makeArgReader } from '../src/cli.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const { rest: argv, printConfig } = applyCliSettings();
await maybePrintConfig(printConfig);
const arg = makeArgReader(argv);
// Defaults to the three test fixtures so the script runs from a bare clone; that
// smoke-tests the pipeline and is nowhere near enough data. The crawl at the
// repo root (fow-crawl-*.json, ~3,000 Chess.com games) is the real corpus.
const SESSIONS = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
const has = name => argv.includes('--' + name);
const FOLDS = Number(arg('folds', '3'));
const L2_GRID = arg('l2', '1e-6,1e-5,1e-4').split(',').map(Number);
const ITERS = Number(arg('iters', '3000'));
const MAX_GAMES = Number(arg('max-games', '1000000'));
const E2E_GAMES = Number(arg('e2e-games', '150'));
// The belief gate's replays run in parallel, one per worker thread. Each holds
// its own belief (up to ~1 GB at |P| = 10⁶), so the count is also a memory knob.
const WORKERS = Number(arg('workers', String(Math.max(1, availableParallelism() - 1))));
// --fit-cache <file>: keep the cross-validated fits on disk, keyed by the
// corpus and the grid, so a re-run of the gate does not redo ~15 minutes of fits.
const FIT_CACHE = arg('fit-cache', null);
const RATING = has('rating');
// A floor on the gain for rating slopes: 0.01 nats is the smallest difference
// this subsystem has ever treated as real.
const MIN_RATING_GAIN = Number(arg('min-rating-gain', '0.01'));
// Table units: logits × this, so the tables read in the same units as
// FITTED_WEIGHTS.temperature (a unit, not a knob).
const UNIT = FITTED_WEIGHTS.temperature;

const PIECE_CODE = { pawn: 1, knight: 2, bishop: 3, rook: 4, queen: 5, king: 6 };
const sqToIdx = sq => (sq.charCodeAt(1) - 49) * 8 + (sq.charCodeAt(0) - 97);

// --- data --------------------------------------------------------------------

if (!existsSync(SESSIONS)) {
  console.error(`No corpus at ${SESSIONS}. Pass --sessions <path> (a directory, .zip, .pgn or .json).`);
  process.exit(1);
}
const { games, stats } = loadCorpus(SESSIONS, { maxGames: MAX_GAMES });
if (!games.length) {
  console.error(`No chess fog games in ${SESSIONS}.\n  ${describeCorpus(games, stats)}`);
  process.exit(1);
}
console.log(describeCorpus(games, stats));

function indexOfAction(moves, action) {
  const f = sqToIdx(action.from), t = sqToIdx(action.to);
  const castle = action.type === 'castle' ? (action.side === 'kingside' ? 1 : 2) : 0;
  const promo = action.payload?.promote ? PIECE_CODE[action.payload.promote] : 0;
  for (let j = 0; j < moves.length; j++) {
    const m = moves[j];
    if (m.f === f && m.t === t && m.castle === castle && m.promo === promo) return j;
  }
  return -1;
}

// The sparse design, in growable typed arrays: decision i owns candidates
// [cand[i], cand[i+1]); candidate j owns entries [ent[j], ent[j+1]).
class Grow {
  constructor(T, n = 1 << 16) { this.T = T; this.a = new T(n); this.n = 0; }
  push(x) { if (this.n === this.a.length) { const b = new this.T(this.a.length * 2); b.set(this.a); this.a = b; } this.a[this.n++] = x; }
  done() { return this.a.subarray(0, this.n); }
}
const candStart = new Grow(Int32Array), chosenG = new Grow(Int32Array), gameG = new Grow(Int32Array);
const ratingG = new Grow(Float64Array);
const entStart = new Grow(Int32Array), entIdx = new Grow(Int16Array), entVal = new Grow(Int8Array), entTerm = new Grow(Int8Array);
const ix = new Int16Array(8), vx = new Int8Array(8), tx = new Int8Array(8);
let unmatched = 0, nCand = 0;
games.forEach(({ sess, ratings }, gi) => {
  let state = FogChess.createInitialState(sess.params.players, sess.params.config);
  for (const entry of sess.log ?? []) {
    const pa = entry.playerActions?.[0];
    if (!pa?.action) break;
    const gs = state.gameSpecific;
    const sign = pa.playerId === 'white' ? 1 : -1;
    const pos = fromBoardObject(state.board, gs.castlingRights, gs.enPassantTarget);
    const moves = genFogMoves(pos, sign);
    const chosen = indexOfAction(moves, pa.action);
    if (chosen < 0) unmatched++;
    // A one-move position carries no information about preferences.
    if (chosen >= 0 && moves.length > 1) {
      candStart.push(nCand); chosenG.push(chosen); gameG.push(gi);
      ratingG.push(ratings?.[pa.playerId] ?? NaN);
      for (const m of moves) {
        entStart.push(entIdx.n);
        const k = moveEntries(pos, m, sign, ix, vx, tx);
        for (let e = 0; e < k; e++) { entIdx.push(ix[e]); entVal.push(vx[e]); entTerm.push(tx[e]); }
        nCand++;
      }
    }
    state = FogChess.applyActions(state, [pa]);
  }
});
candStart.push(nCand); entStart.push(entIdx.n);
const CAND = candStart.done(), CHOSEN = chosenG.done(), GAME = gameG.done(), RATINGS = ratingG.done();
const ENT = entStart.done(), IDX = entIdx.done(), VAL = entVal.done(), TERM = entTerm.done();
const N = CHOSEN.length;
console.log(`${games.length} fog games → ${N} decisions, ${nCand} candidate moves, ${IDX.length} table touches` +
  (unmatched ? `  (WARNING: ${unmatched} recorded actions not found in genFogMoves output)` : ''));

// --- the conditional logit over the tables -----------------------------------

const scratch = new Float64Array(512);
/** Scores of decision i's candidates under θ (logits), into `s`; returns the count. */
function scores(theta, i, s) {
  const c0 = CAND[i], c1 = CAND[i + 1];
  for (let j = c0; j < c1; j++) {
    let x = 0;
    for (let e = ENT[j]; e < ENT[j + 1]; e++) x += theta[IDX[e]] * VAL[e];
    s[j - c0] = x;
  }
  return c1 - c0;
}

/** P(chosen) per decision in `idxs`, and each decision's move count. */
function heldOut(theta, idxs) {
  const p = new Float64Array(idxs.length), n = new Int32Array(idxs.length);
  idxs.forEach((i, r) => {
    const k = scores(theta, i, scratch);
    let max = -Infinity;
    for (let j = 0; j < k; j++) if (scratch[j] > max) max = scratch[j];
    let sum = 0;
    for (let j = 0; j < k; j++) sum += Math.exp(scratch[j] - max);
    p[r] = Math.exp(scratch[CHOSEN[i]] - max) / sum;
    n[r] = k;
  });
  return { p, n };
}

/** Mean log-loss (nats) of the floored mixture over held-out results. */
function logLoss({ p, n }, floor = 0) {
  let s = 0;
  for (let r = 0; r < p.length; r++) s -= Math.log((1 - floor) * p[r] + floor / n[r]);
  return s / p.length;
}

/**
 * The floor that maximizes held-out likelihood. The log-likelihood is concave in
 * the mixture weight, so a bisection on its derivative finds the maximum.
 */
function fitFloor({ p, n }) {
  const d = f => { let s = 0; for (let r = 0; r < p.length; r++) s += (1 / n[r] - p[r]) / ((1 - f) * p[r] + f / n[r]); return s; };
  if (d(0) <= 0) return 0;
  let lo = 0, hi = 0.999;
  for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (d(mid) > 0) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

/**
 * Full-batch Adam on the mean log-likelihood minus (l2/2)·|θ|². Stops when the
 * objective has improved by less than 1e-7 nats over 50 iterations, or at ITERS.
 */
function fit(idxs, { l2, init = null, iters = ITERS, lr = 0.05 }) {
  const theta = init ? Float64Array.from(init) : new Float64Array(NUM_ENTRIES);
  const g = new Float64Array(NUM_ENTRIES), m1 = new Float64Array(NUM_ENTRIES), m2 = new Float64Array(NUM_ENTRIES);
  const s = new Float64Array(512);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  let best = -Infinity, sinceBest = 0, it = 0;
  for (it = 1; it <= iters; it++) {
    g.fill(0);
    let ll = 0;
    for (const i of idxs) {
      const k = scores(theta, i, s);
      let max = -Infinity;
      for (let j = 0; j < k; j++) if (s[j] > max) max = s[j];
      let sum = 0;
      for (let j = 0; j < k; j++) { s[j] = Math.exp(s[j] - max); sum += s[j]; }
      const c0 = CAND[i], ch = CHOSEN[i];
      ll += Math.log(s[ch] / sum);
      // ∇ = count(chosen) − E_π[count]
      for (let j = 0; j < k; j++) {
        const w = (j === ch ? 1 : 0) - s[j] / sum;
        if (w === 0) continue;
        const cj = c0 + j;
        for (let e = ENT[cj]; e < ENT[cj + 1]; e++) g[IDX[e]] += w * VAL[e];
      }
    }
    let pen = 0;
    for (let q = 0; q < NUM_ENTRIES; q++) {
      pen += theta[q] * theta[q];
      const grad = g[q] / idxs.length - l2 * theta[q];
      m1[q] = b1 * m1[q] + (1 - b1) * grad;
      m2[q] = b2 * m2[q] + (1 - b2) * grad * grad;
      theta[q] += lr * (m1[q] / (1 - b1 ** it)) / (Math.sqrt(m2[q] / (1 - b2 ** it)) + eps);
    }
    const obj = ll / idxs.length - 0.5 * l2 * pen;
    if (obj > best + 1e-7) { best = obj; sinceBest = 0; } else if (++sinceBest >= 50) break;
  }
  return { theta, iters: it };
}

const all = Array.from({ length: N }, (_, i) => i);
const inFold = (i, f) => GAME[i] % FOLDS === f;
const shippedTheta = Float64Array.from(vectorFromTables(MOVE_TABLES, FITTED_WEIGHTS.castleBonus), x => x / UNIT);

// --- cross-validation: choose the penalty, fit the floor ---------------------

console.log(`\n=== ${FOLDS}-fold CV by game, per-move log-loss (nats; lower better) ===`);
const uniformLL = (() => { let s = 0; for (let i = 0; i < N; i++) s += Math.log(CAND[i + 1] - CAND[i]); return s / N; })();
const cacheKey = JSON.stringify({ SESSIONS, MAX_GAMES, FOLDS, L2_GRID, ITERS, N, nCand });
const cached = FIT_CACHE && existsSync(FIT_CACHE) ? JSON.parse(readFileSync(FIT_CACHE, 'utf8')) : null;
const byL2 = new Map();   // l2 → { thetas[f], pooled held-out results }
if (cached?.key === cacheKey) {
  console.log(`  (fits loaded from ${FIT_CACHE})`);
  for (const [l2, thetasL2] of cached.byL2) {
    const thetasF = thetasL2.map(t => Float64Array.from(t));
    const pAll = [], nAll = [];
    for (let f = 0; f < FOLDS; f++) { const h = heldOut(thetasF[f], all.filter(i => inFold(i, f))); pAll.push(...h.p); nAll.push(...h.n); }
    byL2.set(l2, { thetas: thetasF, pooled: { p: Float64Array.from(pAll), n: Int32Array.from(nAll) } });
  }
} else {
  // Each fold warm-starts from ITS OWN fit at the previous l2 — never from another
  // fold's, which has trained on this fold's held-out games.
  const warm = new Array(FOLDS).fill(null);
  for (const l2 of L2_GRID) {
    const thetas = [], pAll = [], nAll = [];
    for (let f = 0; f < FOLDS; f++) {
      const tr = all.filter(i => !inFold(i, f)), te = all.filter(i => inFold(i, f));
      const t0 = Date.now();
      const { theta, iters } = fit(tr, { l2, init: warm[f] });
      warm[f] = theta;
      thetas.push(theta);
      const h = heldOut(theta, te);
      pAll.push(...h.p); nAll.push(...h.n);
      console.log(`  l2 ${l2}  fold ${f}: ${iters} iterations, ${((Date.now() - t0) / 1000).toFixed(0)} s, held-out ${logLoss(h).toFixed(4)}`);
    }
    const pooled = { p: Float64Array.from(pAll), n: Int32Array.from(nAll) };
    byL2.set(l2, { thetas, pooled });
    console.log(`  l2 ${l2}: pooled held-out ${logLoss(pooled).toFixed(4)}`);
  }
  if (FIT_CACHE) writeFileSync(FIT_CACHE, JSON.stringify({ key: cacheKey, byL2: [...byL2].map(([l2, v]) => [l2, v.thetas.map(t => Array.from(t))]) }));
}
const [L2] = [...byL2.entries()].sort((a, b) => logLoss(a[1].pooled) - logLoss(b[1].pooled))[0];
const { thetas, pooled } = byL2.get(L2);
const FLOOR = fitFloor(pooled);
// The shipped model, on the same held-out decisions in the same order.
const shippedPooled = (() => {
  const pAll = [], nAll = [];
  for (let f = 0; f < FOLDS; f++) { const h = heldOut(shippedTheta, all.filter(i => inFold(i, f))); pAll.push(...h.p); nAll.push(...h.n); }
  return { p: Float64Array.from(pAll), n: Int32Array.from(nAll) };
})();
console.log(`\n  chosen l2 ${L2}, fitted floor ${FLOOR.toFixed(4)}`);
console.log(`  POOLED: uniform ${uniformLL.toFixed(4)}  shipped ${logLoss(shippedPooled, FITTED_WEIGHTS.floor).toFixed(4)}` +
  `  learned ${logLoss(pooled).toFixed(4)}  learned+floor ${logLoss(pooled, FLOOR).toFixed(4)}`);

// --- the real gate: position log-loss on held-out games ----------------------

const servingWeights = (theta, floor) => {
  const { tables, castleBonus } = tablesFromVector(Float64Array.from(theta, x => x * UNIT));
  return {
    temperature: UNIT, floor, captureWeight: 1, promoWeight: 1,
    pstWeight: [0, 1, 1, 1, 1, 1, 1], castleBonus, tables,
  };
};

let gate = null;
if (has('e2e')) {
  const ARMS = ['uniform-π', 'shipped', 'learned+floor'];
  const perFold = Math.ceil(E2E_GAMES / FOLDS);
  const tasks = [];
  for (let f = 0; f < FOLDS; f++) {
    const held = games.map((g, gi) => gi).filter(gi => gi % FOLDS === f).slice(0, perFold);
    for (const gi of held) for (const seat of ['white', 'black']) for (const arm of ARMS) tasks.push({ id: tasks.length, arm, fold: f, gi, seat });
  }
  console.log(`\n=== ${FOLDS}-fold CV, BELIEF log-loss of the true position (the gate), ` +
    `${tasks.length / ARMS.length} held-out replays per arm, ${WORKERS} workers ===`);
  const acc = new Map(ARMS.map(k => [k, { ll: [], base: [], ranks: [], notIn: 0, giveups: 0 }]));
  const byReplay = new Map();   // "gi|seat|arm" → Map(ply → log-loss, or null if the truth was lost)
  const t0 = Date.now();
  let next = 0, done = 0, lastReport = 0;
  await new Promise((resolve, reject) => {
    const workerData = {
      sessions: SESSIONS, maxGames: MAX_GAMES, unit: UNIT, floor: FLOOR,
      thetas: thetas.map(t => Array.from(t)),
    };
    const pool = Array.from({ length: Math.min(WORKERS, tasks.length) }, () =>
      new Worker(new URL('./belief-gate-worker.mjs', import.meta.url), { workerData }));
    const feed = (w) => { if (next < tasks.length) w.postMessage(tasks[next++]); else w.terminate(); };
    for (const w of pool) {
      w.on('error', reject);
      w.on('message', ({ id, arm, giveup, turns }) => {
        const a = acc.get(arm);
        if (giveup) a.giveups++;
        const plies = new Map();
        byReplay.set(`${tasks[id].gi}|${tasks[id].seat}|${arm}`, plies);
        for (const [found, ll, base, rank, ply] of turns) {
          plies.set(ply, found ? ll : null);
          if (!found) { a.notIn++; continue; }
          a.ll.push(ll); a.base.push(base); a.ranks.push(rank);
        }
        done++;
        const now = Date.now();
        if (done === tasks.length || now - lastReport > 30000) {
          lastReport = now;
          const min = (now - t0) / 60000;
          console.log(`  gate: ${done}/${tasks.length} replays, ${min.toFixed(1)} min` +
            (done < tasks.length ? `, ~${(min / done * (tasks.length - done)).toFixed(0)} min to go` : ''));
        }
        if (done === tasks.length) resolve();
        feed(w);
      });
      feed(w);
    }
  });
  for (const [name, a] of acc) {
    a.ranks.sort((x, y) => x - y);
    console.log(`  ${name.padEnd(18)} ll=${mean(a.ll).toFixed(3)}  flat=${mean(a.base).toFixed(3)}  ` +
      `Δ=${(mean(a.base) - mean(a.ll)).toFixed(3)}  medRank=${a.ranks[a.ranks.length >> 1]}  ` +
      `notInP=${a.notIn}  giveups=${a.giveups}  turns=${a.ll.length}`);
  }
  // Past CAP or the time guard P is a SAMPLE (exactBelief.js), which can lose the
  // true position — and a sharper prior keeps it more often, so each arm's mean
  // above is over a different set of turns. The gate is therefore PAIRED: the
  // log-loss difference on turns where both arms kept the truth, and how often
  // each lost it where the other did not.
  const diffs = [];
  let lostL = 0, lostS = 0;
  for (const t of tasks) {
    if (t.arm !== 'shipped') continue;
    const S = byReplay.get(`${t.gi}|${t.seat}|shipped`), L = byReplay.get(`${t.gi}|${t.seat}|learned+floor`);
    if (!S || !L) continue;
    for (const [ply, ls] of S) {
      if (!L.has(ply)) continue;
      const ll = L.get(ply);
      if (ls != null && ll != null) diffs.push(ll - ls);
      else if (ll == null && ls != null) lostL++;
      else if (ls == null && ll != null) lostS++;
    }
  }
  const md = mean(diffs);
  const se = Math.sqrt(diffs.reduce((q, x) => q + (x - md) ** 2, 0) / (diffs.length - 1) / diffs.length);
  console.log(`  PAIRED, learned − shipped, on ${diffs.length} turns where both kept the true position: ` +
    `${md.toFixed(3)} ± ${se.toFixed(3)} nats (z = ${(md / se).toFixed(1)}; negative favours learned)`);
  console.log(`  true position lost by learned only: ${lostL} turns; by shipped only: ${lostS} turns`);
  gate = { shipped: mean(acc.get('shipped').ll), learned: mean(acc.get('learned+floor').ll), paired: md, se, ok: md + 2 * se < 0 && lostL <= lostS };
}

// --- opponent rating: slopes on the per-term multipliers ---------------------
//
// weight_k(r) = 1 + slope_k · z(r) on each FEATURE_NAMES term, over the tables
// fitted above. The terms of a candidate are its entries summed by `term`. Fitted
// on rated decisions only, and compared with the flat model on the SAME held-out
// rated decisions: slopes ship only if they beat it by MIN_RATING_GAIN.
let ratingSlope = null;
if (RATING) {
  const rated = all.filter(i => Number.isFinite(RATINGS[i]));
  console.log(`\n=== opponent rating, as slopes on the per-term multipliers — held out ===`);
  if (rated.length < 500) {
    console.log(`  only ${rated.length} rated decisions — too few; skipping.`);
  } else {
    const spread = ratingSpread(games);
    const z = i => Math.max(-1.5, Math.min(1.5, ratingZ(RATINGS[i], RATING_PIVOT, RATING_SCALE)));
    // Per-candidate term sums under θ, into F (NUM_FEATURES per candidate).
    const termsOf = (theta, i, F) => {
      const c0 = CAND[i], c1 = CAND[i + 1];
      F.fill(0, 0, (c1 - c0) * NUM_FEATURES);
      for (let j = c0; j < c1; j++) for (let e = ENT[j]; e < ENT[j + 1]; e++) F[(j - c0) * NUM_FEATURES + TERM[e]] += theta[IDX[e]] * VAL[e];
      return c1 - c0;
    };
    const F = new Float64Array(512 * NUM_FEATURES), s = new Float64Array(512);
    // a: multipliers (start at 1); b: slopes. Wide model has both; flat has a only.
    const fitTerms = (theta, idxs, wide) => {
      const W = new Float64Array(2 * NUM_FEATURES); for (let k = 0; k < NUM_FEATURES; k++) W[k] = 1;
      const m1 = new Float64Array(W.length), m2 = new Float64Array(W.length), g = new Float64Array(W.length);
      for (let it = 1; it <= 400; it++) {
        g.fill(0);
        for (const i of idxs) {
          const k = termsOf(theta, i, F), zi = wide ? z(i) : 0;
          let max = -Infinity;
          for (let j = 0; j < k; j++) { let x = 0; for (let q = 0; q < NUM_FEATURES; q++) x += (W[q] + W[NUM_FEATURES + q] * zi) * F[j * NUM_FEATURES + q]; s[j] = x; if (x > max) max = x; }
          let sum = 0; for (let j = 0; j < k; j++) { s[j] = Math.exp(s[j] - max); sum += s[j]; }
          for (let j = 0; j < k; j++) {
            const w = (j === CHOSEN[i] ? 1 : 0) - s[j] / sum;
            for (let q = 0; q < NUM_FEATURES; q++) { g[q] += w * F[j * NUM_FEATURES + q]; if (wide) g[NUM_FEATURES + q] += w * F[j * NUM_FEATURES + q] * zi; }
          }
        }
        for (let q = 0; q < W.length; q++) {
          const grad = g[q] / idxs.length;
          m1[q] = 0.9 * m1[q] + 0.1 * grad; m2[q] = 0.999 * m2[q] + 0.001 * grad * grad;
          W[q] += 0.01 * (m1[q] / (1 - 0.9 ** it)) / (Math.sqrt(m2[q] / (1 - 0.999 ** it)) + 1e-8);
        }
      }
      return W;
    };
    const llTerms = (theta, W, idxs, wide) => {
      let ll = 0;
      for (const i of idxs) {
        const k = termsOf(theta, i, F), zi = wide ? z(i) : 0;
        let max = -Infinity;
        for (let j = 0; j < k; j++) { let x = 0; for (let q = 0; q < NUM_FEATURES; q++) x += (W[q] + W[NUM_FEATURES + q] * zi) * F[j * NUM_FEATURES + q]; s[j] = x; if (x > max) max = x; }
        let sum = 0; for (let j = 0; j < k; j++) sum += Math.exp(s[j] - max);
        ll += s[CHOSEN[i]] - max - Math.log(sum);
      }
      return -ll / idxs.length;
    };
    let flat = 0, sloped = 0, cnt = 0;
    for (let f = 0; f < FOLDS; f++) {
      const tr = rated.filter(i => !inFold(i, f)), te = rated.filter(i => inFold(i, f));
      flat += llTerms(thetas[f], fitTerms(thetas[f], tr, false), te, false) * te.length;
      sloped += llTerms(thetas[f], fitTerms(thetas[f], tr, true), te, true) * te.length;
      cnt += te.length;
    }
    const delta = (flat - sloped) / cnt;
    console.log(`  ${rated.length} rated decisions, ratings ${spread.min}–${spread.max}; held-out flat ${(flat / cnt).toFixed(4)}  ` +
      `sloped ${(sloped / cnt).toFixed(4)}  Δ ${delta.toFixed(4)}  ${delta > MIN_RATING_GAIN ? 'ships' : 'does not ship'}`);
    if (delta > MIN_RATING_GAIN) {
      // castleBonus serves as a VALUE, not a multiplier: its slope is in value units.
      const base = fit(all, { l2: L2, init: thetas[0] }).theta;
      const W = fitTerms(base, rated, true);
      ratingSlope = Array.from(W.subarray(NUM_FEATURES), (b, q) => (q === 8 ? b * base[ENTRY.CASTLE] * UNIT : b));
    }
  }
}

// --- final fit on everything, and optionally write it back -------------------

console.log(`\n=== fitted on all ${N} decisions at l2 ${L2} (this is what --write ships) ===`);
const final = cached?.key === cacheKey && cached.final ? Float64Array.from(cached.final) : fit(all, { l2: L2, init: thetas[0] }).theta;
if (FIT_CACHE && !(cached?.key === cacheKey && cached.final)) {
  const c = JSON.parse(readFileSync(FIT_CACHE, 'utf8'));
  c.final = Array.from(final);
  writeFileSync(FIT_CACHE, JSON.stringify(c));
}
const served = servingWeights(final, FLOOR);
const r1 = x => x.toFixed(0).padStart(6);
console.log(`  capture: ${Object.entries(served.tables.capture).map(([t, v]) => `${t} ${r1(v)}`).join('  ')}`);
console.log(`  promo:   ${Object.entries(served.tables.promo).map(([t, v]) => `${t} ${r1(v)}`).join('  ')}`);
console.log(`  castle ${r1(served.castleBonus)}   floor ${FLOOR.toFixed(4)}   (units: logits × ${UNIT})`);

if (has('write')) {
  if (!gate) { console.error('\n--write needs --e2e: the belief gate decides whether the tables ship.'); process.exit(1); }
  if (!gate.ok) { console.error(`\nNot writing: the learned tables do not beat the shipped ones on the paired belief gate.`); process.exit(1); }
  const provenance = `Fitted ${new Date().toISOString().slice(0, 10)} on ${games.length} games / ${N} decisions ` +
    `(${SESSIONS.split('/').pop()}), l2 ${L2} chosen by ${FOLDS}-fold CV.\n` +
    `// Held-out move log-loss ${logLoss(pooled, FLOOR).toFixed(4)} (shipped ${logLoss(shippedPooled, FITTED_WEIGHTS.floor).toFixed(4)}, ` +
    `uniform ${uniformLL.toFixed(4)});\n// held-out belief log-loss, paired with the shipped tables: ${gate.paired.toFixed(3)} ± ${gate.se.toFixed(3)} nats per turn.`;
  writeFileSync(join(HERE, '..', 'src', 'moveTables.js'), renderTables(served.tables, provenance));
  const file = join(HERE, '..', 'src', 'movePrior.js');
  let src = readFileSync(file, 'utf8');
  const re = /export const FITTED_WEIGHTS = \{[\s\S]*?\n\};/;
  src = src.replace(re, `export const FITTED_WEIGHTS = {
  temperature: ${UNIT},
  floor: ${FLOOR.toFixed(4)},
  captureWeight: 1,
  promoWeight: 1,
  //          -  pawn  knight bishop  rook  queen   king
  pstWeight: [0, 1, 1, 1, 1, 1, 1],
  castleBonus: ${served.castleBonus.toFixed(1)},
};`);
  if (ratingSlope) {
    src = src.replace(/export const RATING_SLOPE = \[[\s\S]*?\];/,
      `export const RATING_SLOPE = [${ratingSlope.map(x => x.toFixed(3)).join(', ')}];`);
  }
  writeFileSync(file, src);
  console.log(`\nWrote src/moveTables.js and FITTED_WEIGHTS${ratingSlope ? ' and RATING_SLOPE' : ''}. Re-run the tests.`);
}
