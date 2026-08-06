// ---------------------------------------------------------------------------
// Exact fog-of-war belief: the paper's position set P (Zhang & Sandholm 2026).
//
// Obscuro "maintains the full set P of possible positions given the
// observations it has seen so far" and samples its search worlds from it. This
// module implements that exactly for FoW chess:
//
//   • P starts as {the standard initial position} (common knowledge).
//   • Before each of our moves, P advances one opponent ply: for every
//     position, every pseudo-legal fog move the opponent could have made
//     (minus moves that would have captured our king — the game would have
//     ended) yields a successor.
//   • Each successor is filtered against our CURRENT observation: our own
//     pieces must match exactly, every square we can see must show exactly
//     what we observe, and the position must reproduce our exact visibility
//     set (so hidden pieces sit only where they wouldn't change what we see —
//     including the pawn-block rule).
//   • Our own chosen move is applied to every position after we commit it.
//   • Positions are deduplicated by placement + castling rights + en passant
//     (P is a set of STATES, not histories — the paper does the same, fn. 21).
//
// P IS A DISTRIBUTION, NOT JUST A SET. Every member carries a weight in
// `this.weights` (parallel to `this.positions`, summing to 1), which is what
// makes "which board is the real one?" answerable at all. Two ingredients:
//
//   • The ply advance multiplies a parent's weight by π(move | parent) — a move
//     prior (movePrior.js), injectable, defaulting to whatever
//     setDefaultMovePrior() was last handed. Because P is a set of STATES,
//     colliding histories have their weights SUMMED rather than the second
//     dropped; that plus "a parent with fewer legal moves passes more mass to
//     each child" means even a UNIFORM π gives a genuinely non-uniform
//     posterior, with no opponent model at all.
//   • The observation filter below is the Bayesian update, for free:
//     successors that fail consistent() are pruned, which removes mass, and
//     renormalizing what survives IS conditioning on the observation.
//
// A re-acquired set (tryReacquire) has no history to weigh, so it falls back to
// uniform weights and stays flagged `approx`.
//
// REPRESENTATION (the capacity tier): positions are Int8Array(66) — 64 signed
// piece codes (+1..+6 white P N B R Q K, negative for black), castling bits at
// [64], en-passant index+1 at [65]. Move generation, application, and the
// visibility test all run on the typed array; visibility is a 2×32-bit mask
// compared with two integer equality checks; dedupe uses a 53-bit Zobrist
// hash. This is ~10× faster and ~15× smaller than the previous object-board
// version, which is what pays for the larger CAP. The typed encodings MUST
// mirror src/moves.js (fog pseudo-legal, castle quirks, promotions,
// en passant) and board.js getVisibleSquares (pawn-block rule, blocker-included
// rays) exactly — the invariant test replays real recorded games and asserts
// the true position stays in P.
//
// Exactness is abandoned (for this game) if P would exceed CAP positions, a
// single update runs past a time guard, an update empties P, or the tracker is
// first attached mid-game — the caller then falls back to the heuristic
// particle belief (belief.js), and tryReacquire() below can later restore a
// tight SUPERSET of P once few pieces remain hidden. While |P| = 1 the agent
// literally knows the true position.
// ---------------------------------------------------------------------------

import { makeMovePrior, UNIFORM_PRIOR, UNIFORM_ONLY, FITTED_WEIGHTS } from './movePrior.js';
import { param, settingsEpoch } from './config.js';

// Exported so src/settings.js can list them; kept defined here,
// next to the tracker that tunes them.
export const CAP = 200000;          // paper: |P| usually ≤ 10⁶ (C++); avg ~17k
export const TIME_GUARD_MS = 4000;  // per-turn update budget
export const REACQUIRE_BOUND = 60000;

// Effective values (defaults above; see docs/SETTINGS.md). Raising CAP and
// TIME_GUARD_MS together is the standard way to trade turn latency for staying
// exact longer, which is why they are settable rather than baked in.
const cap = () => param('chess.EXACT_BELIEF_CAP', CAP);
const timeGuardMs = () => param('chess.EXACT_BELIEF_TIME_GUARD_MS', TIME_GUARD_MS);
const reacquireBound = () => param('chess.REACQUIRE_BOUND', REACQUIRE_BOUND);

// The π used by trackers that don't ask for a specific one — i.e. all of
// production, via getExactBelief.
//
// THESE WEIGHTS ARE FITTED, NOT TUNED BY HAND (movePrior.js FITTED_WEIGHTS;
// regenerate with scripts/fit-move-prior.mjs --write). Mean log-loss of the
// true position, as nats better than a flat posterior over the same set — higher
// is better. 3-fold CV over 37 recorded fog games, both seats, 1320 turns: each
// fold's weights fitted on the other two thirds, belief measured on the held-out
// third (`fit-move-prior.mjs --e2e`):
//
//   uniform π  +0.006   τ=200  +0.135   τ=150  +0.155   τ=60  −0.236
//   FITTED     +0.691   FITTED+floor (shipped)  +0.683   FITTED×1.5  +0.626
//
// Median rank of the true board: 18 under τ=200, 9 under FITTED. The single-τ
// model could not beat ~+0.16 wherever τ was put, because its terms want
// temperatures a factor of 14 apart (rook PST τ≈11 … promotion τ≈154) and one
// knob had to cover them all. That is also what made the old cliff so steep:
// dropping τ to sharpen the PST terms drove capture into confidently-wrong first.
//
// SHARPENING IS STILL NOT FREE — ×1.5 on the fitted weights costs 0.065. What
// changed is that MLE lands on the model's own optimum, so no knob points at the
// cliff any more, and `floor` bounds what a single confident error can cost.
// Log-loss is still the gate and rank still is not — see belief.js's header
// (THREAT_BIAS, MAX_LURKERS) for the two earlier times an over-sharp belief made
// the AI worse.
// Compiled on first use, not at import: `chess.MOVE_PRIOR_FITTED_WEIGHTS` in a
// settings file is deep-merged onto FITTED_WEIGHTS, and a host configures the
// AI after this module has loaded. Compiling is not free (makeMovePrior builds
// a scorer from the weight vector), so it is cached — but keyed on the settings
// epoch, or the first read of the weights would freeze them for the process and
// a sweep over them would silently measure one arm many times.
let defaultPrior = null;
let defaultPriorEpoch = -1;

/** Swap the production π. Pass null to restore the uniform baseline. */
export function setDefaultMovePrior(prior) {
  defaultPrior = prior ?? UNIFORM_PRIOR;
  defaultPriorEpoch = Infinity;   // an explicit choice outlives any settings change
}
export function getDefaultMovePrior() {
  if (defaultPrior && defaultPriorEpoch >= settingsEpoch()) return defaultPrior;
  defaultPriorEpoch = settingsEpoch();
  if (uniformOnly()) return defaultPrior = UNIFORM_PRIOR;
  return defaultPrior = makeMovePrior(param('chess.MOVE_PRIOR_FITTED_WEIGHTS', FITTED_WEIGHTS));
}

// `chess.MOVE_PRIOR_UNIFORM` — serve the model-free baseline instead of the
// fitted model (see movePrior.js's UNIFORM_ONLY).
const uniformOnly = () => param('chess.MOVE_PRIOR_UNIFORM', UNIFORM_ONLY);

// Per-seat override, for A/B harnesses that need one seat's belief to run a
// different model from the other's IN THE SAME PROCESS — which is what a
// seat-swapped strength comparison requires. Production never sets this.
const priorBySeat = new Map();
export function setMovePriorForSeat(color, prior) {
  if (prior) priorBySeat.set(color, prior); else priorBySeat.delete(color);
}

// Exponent applied to the posterior when SAMPLING search worlds: draw ∝ w^α.
// α = 1 is the posterior itself; α = 0 is uniform over P, ignoring the weights.
//
// IT SHIPS AT 0 — the belief is weighted, the SEARCH'S DRAW FROM IT IS NOT — and
// that is the conservative reading of two measurements that disagree:
//
//   • Sample coverage FAVOURS α = 1, mildly. Over 558 turns, the chance a 16-world
//     draw contains the TRUE position is 39.3% at α = 1 vs 36.1% at α = 0
//     (`calibrate-belief.mjs --sample-n 16`). So weighting does not, as one might
//     fear, spend the sample inside a confident slice that excludes reality.
//   • Actual PLAY favours α = 0. Seat-swapped self-play, α=1 vs α=0 over the same
//     τ=200 belief: 4 wins to 11 (`strength-belief.mjs --arm alpha`). Weak — 15
//     decisive games — but it is the only measurement of the thing we actually care
//     about, and on the informative subset (games the black seat won, the ones not
//     swamped by white's large first-move advantage) it is 3-0 to α = 0.
//
// Coverage is a proxy; win/loss is the target. When a proxy and the target disagree,
// follow the target, and prefer the option that changes nothing: α = 0 reproduces
// the world sampling the AI had before any of this, so the belief's new weights
// cannot regress play. Everything the weights were built for — calibration, the
// analysis panel's real posterior, every mass-weighted aggregate — is independent of
// α and keeps the full posterior.
//
// The reason to suspect α = 1 is genuinely hard here, and it is worth knowing before
// re-litigating: the weight ordering puts the true position at median rank ~28 while
// the search looks at ~16 worlds, so α = 1 concentrates a sample smaller than the
// uncertainty it is concentrating over. Raising α needs a higher-powered strength
// measurement than the harness currently gives (see its header) — not this comment.
export const SAMPLE_ALPHA_DEFAULT = 0;
// null = nobody called the setter, so follow the settings layer (which itself
// falls back to the constant above). Keeping "unset" distinct from "set to the
// default value" is what lets a settings file supply the starting α while
// setBeliefSampleAlpha() still wins for a caller that asks explicitly.
let sampleAlpha = null;
const defaultSampleAlpha = () => param('chess.SAMPLE_ALPHA_DEFAULT', SAMPLE_ALPHA_DEFAULT);
export function setBeliefSampleAlpha(a) { sampleAlpha = Number.isFinite(a) ? a : null; }
export function getBeliefSampleAlpha() { return sampleAlpha ?? defaultSampleAlpha(); }

// Per-seat counterpart of the above, same purpose as setMovePriorForSeat.
const alphaBySeat = new Map();
export function setBeliefSampleAlphaForSeat(color, a) {
  if (a == null) alphaBySeat.delete(color); else alphaBySeat.set(color, a);
}

// REACH WEIGHTING — the second, and until 2026-08-02 the missing, channel by
// which the posterior can reach play.
//
// α decides which worlds get SAMPLED. This decides what each sampled world is
// WORTH once the search has it: the CFR weights every world's counterfactual
// value by its root reach (vendor/obscuro/src/infoset.js), and that reach was a flat
// 1/N. With α = 0 the draw is uniform too, so the two channels together meant the
// AI evaluated under a uniform belief over P — the entire fitted posterior was
// computed, displayed in the analysis panel, and then discarded before it could
// affect a single move. That is also why raising α measured as nothing: it moved
// worlds between two uniform treatments.
//
// SHIPPED OFF, on a measurement that leans against it. 1,420 paired positions
// (move-quality.mjs --arm reach): mean +24.4 ± 18.9 cp and a sign test of 46.8%
// (z = 1.30 and −1.29) — both pointing at uniform reach being BETTER, neither at
// 2σ. Off is also the option that changes nothing, which is how the two previous
// belief knobs (τ<60, α=1) were settled after the principled choice measured
// worse.
//
// The likely reason is variance, not incorrectness: the weights inside a 16-world
// draw are steep (effective sample size 3–11, dipping to 1.4 when one world holds
// 82% of the mass), so weighting is a correct estimator of the right measure
// computed from ~4 effective worlds instead of 16. The posterior's median true-
// board rank is 9 — good, not good enough to bet a 4× smaller sample on.
//
// TWO THINGS BEFORE ANYONE FLIPS THIS ON. (1) That run had 21% static-eval
// fallbacks against ~1% clean — a shared machine, so it is not quotable, and the
// degradation plausibly penalises the weighted arm harder (a bad leaf value in a
// heavily-weighted world costs more). Re-run it idle. (2) The interesting variant
// is not on/off but TEMPERED: reach ∝ w^β with β≈0.5 keeps part of the correction
// while raising the effective sample. That is the arm worth measuring next.
// The value is an EXPONENT β, not a flag: reach ∝ w^β. 0 = off (flat 1/N), 1 =
// the full posterior, and β in between tempers it — which is the variant worth
// measuring, since the objection to β=1 is variance (a correct estimator over ~4
// effective worlds) rather than incorrectness. `true`/`false` still work.
export const REACH_WEIGHTING_DEFAULT = 0;
const asBeta = (v) => (v == null ? null : v === true ? 1 : v === false ? 0 : Number(v));
let reachWeighting = null;   // null = unset; see setBeliefSampleAlpha above
export function setBeliefReachWeighting(on) {
  reachWeighting = asBeta(on);
}
const reachBySeat = new Map();
export function setBeliefReachWeightingForSeat(color, on) {
  const b = asBeta(on);
  if (b == null) reachBySeat.delete(color); else reachBySeat.set(color, b);
}
export function getBeliefReachWeighting(color) {
  return reachBySeat.get(color) ?? reachWeighting
    ?? param('chess.REACH_WEIGHTING_DEFAULT', REACH_WEIGHTING_DEFAULT);
}

// --- encoding ---------------------------------------------------------------

const PIECE_CODE = { pawn: 1, knight: 2, bishop: 3, rook: 4, queen: 5, king: 6 };
const CODE_TYPE = [null, 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
const FILES = 'abcdefgh';

const sqToIdx = sq => (sq.charCodeAt(1) - 49) * 8 + (sq.charCodeAt(0) - 97);
const idxToSq = i => FILES[i & 7] + (((i >> 3) | 0) + 1);

// cr bits at [64]
const WK = 1, WQ = 2, BK = 4, BQ = 8;

function signOf(color) { return color === 'white' ? 1 : -1; }

// Convert an engine board object into a compact position.
export function fromBoardObject(board, cr, ep) {
  const p = new Int8Array(66);
  for (const sq of Object.keys(board)) {
    const pc = board[sq];
    if (!pc) continue;
    p[sqToIdx(sq)] = PIECE_CODE[pc.type] * signOf(pc.ownerId);
  }
  p[64] = (cr?.white?.kingSide ? WK : 0) | (cr?.white?.queenSide ? WQ : 0)
        | (cr?.black?.kingSide ? BK : 0) | (cr?.black?.queenSide ? BQ : 0);
  p[65] = ep ? sqToIdx(ep) + 1 : 0;
  return p;
}

// Convert a compact position back to the engine's board-object shape (with
// synthesised piece ids — P is a set of states, identity is not tracked).
export function toBoardObject(pos) {
  const board = {};
  const counts = {};
  for (let i = 0; i < 64; i++) {
    const c = pos[i];
    if (!c) continue;
    const color = c > 0 ? 'white' : 'black';
    const type = CODE_TYPE[Math.abs(c)];
    const key = color[0] + type;
    const n = counts[key] = (counts[key] ?? 0) + 1;
    const sq = idxToSq(i);
    board[sq] = { id: color[0] + type.toUpperCase()[0] + '_' + n, ownerId: color, type, position: sq, alive: true };
  }
  return board;
}

export function crObjectOf(pos) {
  const b = pos[64];
  return {
    white: { kingSide: !!(b & WK), queenSide: !!(b & WQ) },
    black: { kingSide: !!(b & BK), queenSide: !!(b & BQ) },
  };
}

export function epOf(pos) { return pos[65] ? idxToSq(pos[65] - 1) : null; }

// --- Zobrist hashing (dedupe key; 53-bit, fixed-seeded, deterministic) ------

const ZOB = (() => {
  let s = 0x9e3779b9 | 0;
  const next = () => {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) | 0;
  };
  const sq1 = new Int32Array(64 * 13), sq2 = new Int32Array(64 * 13);
  for (let i = 0; i < sq1.length; i++) { sq1[i] = next(); sq2[i] = next(); }
  const cr1 = new Int32Array(16), cr2 = new Int32Array(16);
  for (let i = 0; i < 16; i++) { cr1[i] = next(); cr2[i] = next(); }
  const ep1 = new Int32Array(65), ep2 = new Int32Array(65);
  for (let i = 0; i < 65; i++) { ep1[i] = next(); ep2[i] = next(); }
  return { sq1, sq2, cr1, cr2, ep1, ep2 };
})();

function hashPos(p) {
  let h1 = 0, h2 = 0;
  for (let i = 0; i < 64; i++) {
    const c = p[i];
    if (!c) continue;
    const k = i * 13 + (c + 6);
    h1 ^= ZOB.sq1[k]; h2 ^= ZOB.sq2[k];
  }
  h1 ^= ZOB.cr1[p[64]]; h2 ^= ZOB.cr2[p[64]];
  h1 ^= ZOB.ep1[p[65]]; h2 ^= ZOB.ep2[p[65]];
  return (h1 >>> 11) * 4294967296 + (h2 >>> 0); // 21 + 32 = 53 bits
}

// --- visibility mask (mirrors board.js getVisibleSquares exactly) -----------

const KNIGHT_D = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
const KING_D = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const ROOK_D = [[0, 1], [0, -1], [1, 0], [-1, 0]];
const BISHOP_D = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const QUEEN_D = [...ROOK_D, ...BISHOP_D];

// Returns visibility for `sign`'s pieces as a 2×32-bit mask [lo, hi].
function visibilityMask(p, sign) {
  let lo = 0, hi = 0;
  const add = (i) => { if (i < 32) lo |= (1 << i); else hi |= (1 << (i - 32)); };
  for (let i = 0; i < 64; i++) {
    const c = p[i];
    if (!c || (c > 0) !== (sign > 0)) continue;
    add(i);
    const f = i & 7, r = i >> 3;
    const t = c > 0 ? c : -c;
    if (t === 1) { // pawn: pushes only when unblocked; diagonals always
      const dr = sign > 0 ? 1 : -1;
      const r1 = r + dr;
      if (r1 >= 0 && r1 < 8) {
        const push = r1 * 8 + f;
        if (!p[push]) {
          add(push);
          if (r === (sign > 0 ? 1 : 6)) {
            const push2 = (r + 2 * dr) * 8 + f;
            if (!p[push2]) add(push2);
          }
        }
        if (f > 0) add(r1 * 8 + f - 1);
        if (f < 7) add(r1 * 8 + f + 1);
      }
    } else if (t === 2) {
      for (const [df, dr] of KNIGHT_D) {
        const nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) add(nr * 8 + nf);
      }
    } else if (t === 6) {
      for (const [df, dr] of KING_D) {
        const nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) add(nr * 8 + nf);
      }
    } else { // sliders: ray includes the first blocker
      const dirs = t === 4 ? ROOK_D : t === 3 ? BISHOP_D : QUEEN_D;
      for (const [df, dr] of dirs) {
        let nf = f + df, nr = r + dr;
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
          const j = nr * 8 + nf;
          add(j);
          if (p[j]) break;
          nf += df; nr += dr;
        }
      }
    }
  }
  return [lo | 0, hi | 0];
}

// --- fog pseudo-legal move generation (mirrors moves.js getAllFogMoves) -----
// Moves are {f, t, promo (code 0|2..5), dbl, ep (captured idx | -1), castle
// (0 none, 1 kingside, 2 queenside)}.

// Exported for fit-move-prior.mjs: the fitter must normalize π over EXACTLY the
// choice set production normalizes over, or the weights it learns are for a
// different model than the one being served.
export function genFogMoves(p, sign) {
  const out = [];
  const push = (f, t, promo = 0, dbl = false, ep = -1, castle = 0) =>
    out.push({ f, t, promo, dbl, ep, castle });
  const epIdx = p[65] - 1; // -1 when none
  for (let i = 0; i < 64; i++) {
    const c = p[i];
    if (!c || (c > 0) !== (sign > 0)) continue;
    const f = i & 7, r = i >> 3;
    const t = c > 0 ? c : -c;
    if (t === 1) { // pawn
      const dr = sign > 0 ? 1 : -1;
      const startR = sign > 0 ? 1 : 6;
      const promoR = sign > 0 ? 7 : 0;
      const r1 = r + dr;
      if (r1 >= 0 && r1 < 8) {
        const one = r1 * 8 + f;
        if (!p[one]) {
          if (r1 === promoR) { for (const pc of [5, 4, 3, 2]) push(i, one, pc); }
          else {
            push(i, one);
            if (r === startR) {
              const two = (r + 2 * dr) * 8 + f;
              if (!p[two]) push(i, two, 0, true);
            }
          }
        }
        for (const df of [-1, 1]) {
          const nf = f + df;
          if (nf < 0 || nf > 7) continue;
          const cap = r1 * 8 + nf;
          if (cap === epIdx) {
            const victim = r * 8 + nf; // the double-pushed pawn
            if (p[victim] === -sign) push(i, cap, 0, false, victim);
          }
          const occ = p[cap];
          if (occ && (occ > 0) !== (sign > 0)) {
            if (r1 === promoR) { for (const pc of [5, 4, 3, 2]) push(i, cap, pc); }
            else push(i, cap);
          }
        }
      }
    } else if (t === 2 || t === 6) {
      for (const [df, dr] of (t === 2 ? KNIGHT_D : KING_D)) {
        const nf = f + df, nr = r + dr;
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        const j = nr * 8 + nf;
        const occ = p[j];
        if (occ && (occ > 0) === (sign > 0)) continue;
        push(i, j);
      }
      if (t === 6) {
        // Castling under fog: rights bit + king on its home square + empty path
        // + SOMETHING on the rook corner (mirrors moves.js, quirks included:
        // no attack checks, and the corner occupant's identity is trusted).
        const home = sign > 0 ? 4 : 60; // e1 / e8
        if (i === home) {
          const base = sign > 0 ? 0 : 56;
          const rights = p[64];
          const ks = sign > 0 ? (rights & WK) : (rights & BK);
          const qs = sign > 0 ? (rights & WQ) : (rights & BQ);
          if (ks && !p[base + 5] && !p[base + 6] && p[base + 7]) push(i, base + 6, 0, false, -1, 1);
          if (qs && !p[base + 1] && !p[base + 2] && !p[base + 3] && p[base]) push(i, base + 2, 0, false, -1, 2);
        }
      }
    } else {
      const dirs = t === 4 ? ROOK_D : t === 3 ? BISHOP_D : QUEEN_D;
      for (const [df, dr] of dirs) {
        let nf = f + df, nr = r + dr;
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
          const j = nr * 8 + nf;
          const occ = p[j];
          if (occ) { if ((occ > 0) !== (sign > 0)) push(i, j); break; }
          push(i, j);
          nf += df; nr += dr;
        }
      }
    }
  }
  return out;
}

// Castling-rights corners: clearing masks by square index.
const CR_CLEAR = new Int8Array(64).fill(0xF);
CR_CLEAR[0] &= ~WQ; CR_CLEAR[7] &= ~WK; CR_CLEAR[56] &= ~BQ; CR_CLEAR[63] &= ~BK;

function applyMove(p, m, sign) {
  const n = p.slice();
  let cr = n[64];
  if (m.castle) {
    const base = sign > 0 ? 0 : 56;
    n[m.t] = n[m.f]; n[m.f] = 0;
    if (m.castle === 1) { n[base + 5] = n[base + 7]; n[base + 7] = 0; }
    else { n[base + 3] = n[base]; n[base] = 0; }
    cr &= sign > 0 ? ~(WK | WQ) : ~(BK | BQ);
  } else {
    const mover = n[m.f];
    const t = mover > 0 ? mover : -mover;
    n[m.t] = m.promo ? m.promo * sign : mover;
    n[m.f] = 0;
    if (m.ep >= 0) n[m.ep] = 0;
    if (t === 6) cr &= sign > 0 ? ~(WK | WQ) : ~(BK | BQ);
    cr &= CR_CLEAR[m.f]; // a rook leaving its corner loses that right…
    cr &= CR_CLEAR[m.t]; // …as does a rook captured on its corner
  }
  n[64] = cr;
  n[65] = m.dbl ? (((m.f + m.t) >> 1) + 1) : 0;
  return n;
}

// --- the tracker -------------------------------------------------------------

const other = c => (c === 'white' ? 'black' : 'white');

function initialPosition() {
  const p = new Int8Array(66);
  const back = [4, 2, 3, 5, 6, 3, 2, 4];
  for (let f = 0; f < 8; f++) {
    p[f] = back[f]; p[8 + f] = 1;          // white
    p[56 + f] = -back[f]; p[48 + f] = -1;  // black
  }
  p[64] = WK | WQ | BK | BQ;
  return p;
}

// Precompute the per-turn observation context used by every consistency check.
function obsContext(observation, mySign) {
  const obsArr = new Int8Array(64);
  for (const sq of Object.keys(observation.board)) {
    const pc = observation.board[sq];
    if (pc) obsArr[sqToIdx(sq)] = PIECE_CODE[pc.type] * signOf(pc.ownerId);
  }
  let visLo = 0, visHi = 0;
  for (const sq of observation.visibleSquares ?? []) {
    const i = sqToIdx(sq);
    if (i < 32) visLo |= (1 << i); else visHi |= (1 << (i - 32));
  }
  visLo |= 0; visHi |= 0;
  return { obsArr, visLo, visHi, mySign };
}

// Candidate consistency: (a) every visible square shows exactly the observed
// content, (b) our pieces match exactly everywhere (the observation always
// includes ALL our pieces, visible-square or not), (c) the candidate
// reproduces our exact visibility mask (blocking + pawn rule).
function consistent(ctx, cand) {
  const { obsArr, visLo, visHi, mySign } = ctx;
  for (let i = 0; i < 64; i++) {
    const vis = i < 32 ? (visLo >>> i) & 1 : (visHi >>> (i - 32)) & 1;
    const c = cand[i], o = obsArr[i];
    if (vis) { if (c !== o) return false; }
    else if ((c > 0) === (mySign > 0) && c !== 0) { if (c !== o) return false; }
    else if ((o > 0) === (mySign > 0) && o !== 0) { return false; }
  }
  const [lo, hi] = visibilityMask(cand, mySign);
  return lo === visLo && hi === visHi;
}

export class ExactBelief {
  constructor(aiColor, movePrior = null) {
    this.aiColor = aiColor;
    this.oppColor = other(aiColor);
    this.mySign = signOf(aiColor);
    this.exact = null;      // null = not initialised; true/false afterwards
    this.approx = false;    // true when P was re-acquired (tight superset)
    this.positions = null;  // Int8Array(66)[]
    this.weights = null;    // Float64Array parallel to positions, Σ = 1
    this.firstTurnDone = false;
    this._lastTurnKey = null;
    // Resolved at construction, not per sweep, so a tracker's model can't change
    // under it mid-game — that would make its own weights incomparable.
    this._prior = movePrior ?? getDefaultMovePrior();
    this._alpha = alphaBySeat.get(aiColor) ?? getBeliefSampleAlpha();
    this._pi = new Float64Array(256); // per-parent π scratch; grown if needed
  }

  _giveUp() { this.exact = false; this.positions = null; this.weights = null; }

  /**
   * The sampling exponent this tracker resolved at construction. Public because
   * a consumer that draws worlds has to know how much of the posterior is
   * already in the draw before deciding how much to carry as an importance
   * weight — see FogChess.sampleWorlds.
   */
  get sampleAlpha() { return this._alpha; }

  // Normalize a freshly built weight array to Σ = 1. Pruning inconsistent
  // successors removed mass, so this is the conditioning step. A total of 0 can
  // only come from float underflow over a long game (every surviving world
  // vanishingly unlikely relative to the ones the observation killed); fall back
  // to uniform rather than propagate NaN, since a flat belief over the right set
  // is still correct-if-vague, while NaN weights would silently break sampling.
  _setWeights(list) {
    const n = list.length;
    const w = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += list[i];
    if (sum > 0) { const inv = 1 / sum; for (let i = 0; i < n; i++) w[i] = list[i] * inv; }
    else w.fill(n ? 1 / n : 0);
    this.weights = w;
  }

  /**
   * Advance + filter P for this turn. Idempotent per `turnKey` (the agent may
   * sample worlds more than once per decision). Must be called on EVERY one of
   * our turns from the very first, or the tracker gives up (it cannot
   * reconstruct a missed history).
   */
  beginTurn(observation, turnKey = null) {
    if (turnKey != null) {
      if (this._lastTurnKey === turnKey) return;
      this._lastTurnKey = turnKey;
    }
    if (this.exact === false) return;
    const t0 = Date.now();
    const ctx = obsContext(observation, this.mySign);
    if (!this.firstTurnDone) {
      this.firstTurnDone = true;
      // Exactness needs the full history: only attach at the game's first turn.
      if ((observation.turnNumber ?? 1) !== 1) { this._giveUp(); return; }
      this.positions = [initialPosition()];
      this.weights = Float64Array.of(1); // common knowledge: one world, certainly
      this.exact = true;
      if (this.aiColor === 'black') {
        this._advanceOpponent(ctx, t0); // white has already made one ply
      } else {
        // Filter positions and weights in LOCKSTEP — they are parallel arrays and
        // a `positions.filter()` that leaves `weights` alone silently misaligns
        // every world with someone else's probability.
        const kept = [], keptW = [];
        for (let i = 0; i < this.positions.length; i++) {
          if (!consistent(ctx, this.positions[i])) continue;
          kept.push(this.positions[i]);
          keptW.push(this.weights[i]);
        }
        this.positions = kept;
        this._setWeights(keptW);
      }
    } else {
      this._advanceOpponent(ctx, t0);
    }
    if (this.exact && (!this.positions || this.positions.length === 0)) this._giveUp();
  }

  /**
   * Apply our chosen move (an engine action) to every position in P. Our move is
   * KNOWN, so it carries probability 1 and each position's weight passes straight
   * through — but distinct parents can still collide into one successor (our move
   * can erase the very difference between them, e.g. capturing on a square two
   * worlds disagreed about), and those weights must be SUMMED, not dropped.
   */
  commitOurMove(action) {
    if (!this.exact || !this.positions || !action?.from) return;
    const m = {
      f: sqToIdx(action.from),
      t: sqToIdx(action.to),
      promo: action.payload?.promote ? PIECE_CODE[action.payload.promote] : 0,
      dbl: !!action.isDoublePush,
      ep: action.isEnPassant && action.capturedSquare ? sqToIdx(action.capturedSquare) : -1,
      castle: action.type === 'castle' ? (action.side === 'kingside' ? 1 : 2) : 0,
    };
    const next = [];
    const nextW = [];
    const seen = new Map(); // hash → index into next
    const W = this.weights;
    for (let i = 0; i < this.positions.length; i++) {
      const pos = this.positions[i];
      const mover = pos[m.f];
      if (!mover || (mover > 0) !== (this.mySign > 0)) continue; // inconsistent
      const np = applyMove(pos, m, this.mySign);
      const w = W ? W[i] : 1;
      const h = hashPos(np);
      const at = seen.get(h);
      if (at !== undefined) { nextW[at] += w; continue; }
      seen.set(h, next.length);
      next.push(np);
      nextW.push(w);
    }
    this.positions = next;
    if (next.length === 0) { this._giveUp(); return; }
    this._setWeights(nextW);
  }

  // One opponent ply: successors of every position under every fog-legal
  // opponent move, minus impossibilities (king captured / no move available),
  // filtered INLINE against the current observation.
  //
  // This is where the belief's whole probabilistic content is created, so the
  // bookkeeping matters more than it looks:
  //
  //   • π is computed over the parent's FULL fog-legal move list, BEFORE any
  //     pruning, because that list is the opponent's actual choice set. The mass
  //     on moves we then prune (they'd have captured our king, or the successor
  //     contradicts what we see) is evidence, and dropping it is the Bayesian
  //     update — _setWeights renormalizes what survives.
  //   • Colliding successors ACCUMULATE. P is a set of states, so two histories
  //     landing on one position are one member with the sum of their
  //     probabilities. Dropping the second (what a Set does) is exactly the line
  //     that used to make the posterior flat.
  _advanceOpponent(ctx, t0) {
    const oppSign = -this.mySign;
    const myKing = 6 * this.mySign;
    const { obsArr, visLo, visHi } = ctx;
    const next = [];
    const nextW = [];
    const seen = new Map(); // hash → index into next
    const W = this.weights;
    const prior = this._prior;
    for (let pi = 0; pi < this.positions.length; pi++) {
      const pos = this.positions[pi];
      if (Date.now() - t0 > timeGuardMs()) { this._giveUp(); return; }
      const moves = genFogMoves(pos, oppSign);
      if (moves.length === 0) continue; // the opponent DID move
      if (moves.length > this._pi.length) this._pi = new Float64Array(moves.length * 2);
      const pmf = this._pi;
      prior(pos, moves, oppSign, pmf);
      const pw = W ? W[pi] : 1;
      for (let j = 0; j < moves.length; j++) {
        const m = moves[j];
        // Capturing our king ends the game — it didn't, so prune.
        if (pos[m.t] === myKing) continue;
        // Cheap pre-filter: a visible destination must show the moved piece.
        const vis = m.t < 32 ? (visLo >>> m.t) & 1 : (visHi >>> (m.t - 32)) & 1;
        if (vis && !m.castle) {
          const after = m.promo ? m.promo * oppSign : pos[m.f];
          if (obsArr[m.t] !== after) continue;
        }
        const np = applyMove(pos, m, oppSign);
        if (!consistent(ctx, np)) continue;
        const w = pw * pmf[j];
        const h = hashPos(np);
        const at = seen.get(h);
        if (at !== undefined) { nextW[at] += w; continue; }
        seen.set(h, next.length);
        next.push(np);
        nextW.push(w);
        if (next.length > cap()) { this._giveUp(); return; }
      }
    }
    this.positions = next;
    this._setWeights(nextW);
  }

  /**
   * Try to RE-ACQUIRE a lost position set from the heuristic belief's
   * per-piece possible-square sets. Only possible when every hidden piece's
   * set is still a guaranteed superset of the truth (never truncated, no
   * possible promotion) and the cross-product of placements is small — in
   * practice late-game positions with few hidden pieces. The result is a
   * TIGHT SUPERSET of the true P (per-piece sets can't encode inter-piece
   * move-history correlations), marked `approx`; from here on it is advanced
   * exactly again, so it stays a superset — strictly better than particles.
   *
   * It is also the one path with NO POSTERIOR. Per-piece possible-squares carry
   * no history, so there is nothing to weigh the placements by: the weights come
   * out uniform, and the `approx` flag (which the panel already warns on) has to
   * be read as "these numbers are not a posterior" as much as "this set is a
   * superset". Subsequent plies re-introduce real weights on top.
   */
  tryReacquire(observation, belief, turnKey = null) {
    if (this.exact || !belief) return;
    if (turnKey != null) {
      if (this._lastReacqKey === turnKey) return;
      this._lastReacqKey = turnKey;
    }
    const ctx = obsContext(observation, this.mySign);
    const obsBoard = observation.board;
    const seenIds = new Set();
    for (const sq of Object.keys(obsBoard)) {
      const pc = obsBoard[sq];
      if (pc && pc.ownerId === this.oppColor) seenIds.add(pc.id);
    }
    const visSet = new Set(observation.visibleSquares ?? []);
    const hidden = [];
    for (const pc of belief.pieces.values()) {
      if (!pc.alive || seenIds.has(pc.id)) continue;
      if (pc.truncated) return; // set may exclude the truth — cannot re-acquire
      const cands = [...pc.possible].filter(sq => !visSet.has(sq) && !obsBoard[sq]).map(sqToIdx);
      if (cands.length === 0) return; // contradiction — leave it to the particles
      hidden.push({ code: PIECE_CODE[pc.type] * -this.mySign, cands });
    }
    let bound = 1;
    for (const h of hidden) { bound *= h.cands.length; if (bound > reacquireBound()) return; }

    // Base array: the observed board (all our pieces + visible enemies), with
    // OUR castling rights from the observation; the opponent's rights are
    // granted per placement when king+rook stand on their home squares
    // (necessary condition; over-granting is the safe direction).
    const base = new Int8Array(66);
    base.set(ctx.obsArr.subarray(0, 64));
    const myCr = observation.gameSpecific?.castlingRights?.[this.aiColor];
    const myBits = this.aiColor === 'white'
      ? ((myCr?.kingSide ? WK : 0) | (myCr?.queenSide ? WQ : 0))
      : ((myCr?.kingSide ? BK : 0) | (myCr?.queenSide ? BQ : 0));
    const forced = [...(belief.forcedEnemy ?? new Set())].map(sqToIdx);
    const oppSign = -this.mySign;
    const homeR = this.mySign > 0 ? 56 : 0; // opponent's back-rank base index
    const out = [];
    const seen = new Set();
    const t0 = Date.now();
    const place = (i, arr) => {
      if (out.length > cap() || Date.now() - t0 > timeGuardMs()) return false;
      if (i === hidden.length) {
        for (const fi of forced) { // a piece of ours was just captured there
          const f = arr[fi];
          if (!f || (f > 0) === (this.mySign > 0)) return true; // inconsistent, skip
        }
        const np = arr.slice();
        const kingHome = np[homeR + 4] === 6 * oppSign;
        let oppBits = 0;
        if (kingHome && np[homeR + 7] === 4 * oppSign) oppBits |= this.mySign > 0 ? BK : WK;
        if (kingHome && np[homeR] === 4 * oppSign) oppBits |= this.mySign > 0 ? BQ : WQ;
        np[64] = myBits | oppBits;
        np[65] = 0;
        if (consistent(ctx, np)) {
          const h = hashPos(np);
          if (!seen.has(h)) { seen.add(h); out.push(np); }
        }
        return true;
      }
      const { code, cands } = hidden[i];
      for (const idx of cands) {
        if (arr[idx]) continue;
        arr[idx] = code;
        const ok = place(i + 1, arr);
        arr[idx] = 0;
        if (!ok) return false;
      }
      return true;
    };
    if (!place(0, base)) return; // bailed on cap/time
    if (out.length === 0 || out.length > cap()) return;
    this.positions = out;
    this._setWeights(new Array(out.length).fill(1)); // no history → no posterior
    this.exact = true;
    this.approx = true;              // superset, not the literal history-exact P
    this._lastTurnKey = turnKey;     // this turn is done; advance resumes next turn
  }

  /**
   * Sample up to n positions from P WITHOUT REPLACEMENT, in proportion to their
   * posterior weight, in the engine's object shape: { board, cr, ep }.
   *
   * This is the draw that decides how the AI spends its search. It draws ∝ w^α where
   * α is `sampleAlpha`, which DEFAULTS TO 0 — i.e. uniformly over P, deliberately
   * ignoring the posterior. Read that comment before changing it: "sample the worlds
   * that matter more often" is the obvious thing to want and it measured worse in
   * actual play.
   *
   * Because the picks are already distributed by whatever measure α selects, a
   * consumer must treat them as EQUALLY weighted samples — re-weighting them would
   * count the posterior twice (see FogChess.sampleWorlds).
   *
   * Efraimidis–Spirakis exponential race: draw E_i ~ Exp(1)/w_i^α and keep the n
   * smallest. One O(|P|) pass with a kept-sorted array of size n (a handful), so
   * no 200k-entry index array and no full sort. Weight 0 sorts last, which is
   * what "possible but vanishingly unlikely" should do.
   */
  samplePositions(n, rng = Math.random) {
    const idx = this.sampleIndices(n, rng);
    if (!idx) return null;
    return idx.map(i => {
      const p = this.positions[i];
      return { board: toBoardObject(p), cr: crObjectOf(p), ep: epOf(p) };
    });
  }

  /**
   * The draw itself, as absolute indices into P — samplePositions without the
   * object conversion. Split out so the calibration harness can ask the question
   * that actually explains the strength result ("does an n-world draw at this α
   * even contain the true position?") without materialising boards, and so `α` can
   * be overridden per call for that comparison.
   */
  sampleIndices(n, rng = Math.random, alpha = this._alpha) {
    if (!this.exact || !this.positions?.length) return null;
    const P = this.positions;
    if (P.length <= n) return P.map((_, i) => i);
    const W = alpha === 0 ? null : this.weights;
    const best = []; // { key, i }, ascending by key
    for (let i = 0; i < P.length; i++) {
      const w = W ? (alpha === 1 ? W[i] : Math.pow(W[i], alpha)) : 1;
      // rng() can legitimately return 0; -log(0) would make every such world an
      // unbreakable last place rather than a very-unlikely one.
      const key = w > 0 ? -Math.log(rng() || Number.MIN_VALUE) / w : Infinity;
      if (best.length === n && key >= best[n - 1].key) continue;
      let k = best.length;
      while (k > 0 && best[k - 1].key > key) k--;
      best.splice(k, 0, { key, i });
      if (best.length > n) best.pop();
    }
    return best.map(b => b.i);
  }

  /**
   * Positions at the given absolute indices into P, in the engine's object
   * shape { board, cr, ep, w } — the enumeration counterpart of samplePositions
   * (which draws at random, in proportion to w). Lets a caller walk the WHOLE
   * set once, in batches, without replacement. Out-of-range indices are skipped;
   * null when exact tracking isn't active.
   *
   * `w` is the position's posterior weight, and an enumerating caller MUST use it
   * — an unweighted mean over enumerated worlds is an average over the wrong
   * measure. (Sampled worlds are the opposite case: the weight is already in the
   * draw.) See ObscuroAgent.cpSumsOverWorlds.
   */
  positionsAt(indices) {
    if (!this.exact || !this.positions?.length) return null;
    const P = this.positions;
    const W = this.weights;
    const out = [];
    for (const i of indices) {
      const p = P[i];
      if (p) out.push({ board: toBoardObject(p), cr: crObjectOf(p), ep: epOf(p), w: W ? W[i] : 1 / P.length });
    }
    return out;
  }

  /**
   * Rank P by POSTERIOR LIKELIHOOD, so a viewer can be shown "the most likely
   * board" rather than an arbitrary member of the set.
   *
   * This used to be a much longer function, and the reason is worth knowing: the
   * posterior over P was flat — every position exactly as consistent with the
   * observation history as every other — so "the single most likely board" was
   * not a question the belief could answer, and this ranked by a
   * product-of-per-square-marginals surrogate instead (the CONSENSUS board, not
   * the likely one). With `this.weights` there is a real posterior and the whole
   * surrogate collapses to "sort by weight".
   *
   * `approx` flags a population that is a re-acquired SUPERSET of the true P
   * rather than the history-exact set (see tryReacquire / this.approx). There it
   * means two things at once: the set may contain impossible boards, AND the
   * weights are uniform rather than a posterior. The caller must not present
   * either as certainty.
   *
   * Returns { total, top: [{ index, prob }] (best `limit` first), probs, approx },
   * where `probs` is the weight of EVERY index (so a caller holding indices from
   * some other enumeration — e.g. the analysis walk's evaluated worlds — can
   * label them too). Null when exact tracking isn't active.
   */
  rankByLikelihood(limit = 32) {
    if (!this.exact || !this.positions?.length) return null;
    const P = this.positions;
    // A tracker built by hand (tests) or an older path may have no weights; a
    // flat distribution is the honest reading of "no posterior available".
    const probs = this.weights ?? new Float64Array(P.length).fill(1 / P.length);

    // Bounded selection rather than a full sort of up to CAP entries: `limit` is
    // a UI page size (tens), so an insertion into a kept-sorted small array is
    // cheaper than ordering the whole set.
    const cap = Math.max(1, Math.min(limit, P.length));
    const top = [];
    for (let k = 0; k < P.length; k++) {
      const w = probs[k];
      if (top.length === cap && w <= top[top.length - 1].prob) continue;
      let i = top.length;
      while (i > 0 && top[i - 1].prob < w) i--;
      top.splice(i, 0, { index: k, prob: w });
      if (top.length > cap) top.pop();
    }

    return { total: P.length, probs, top, approx: !!this.approx };
  }
}

// Per-game store, same pattern as belief.js: keyed by the players array
// identity, then by colour.
const store = new WeakMap();

export function getExactBelief(state, aiColor) {
  let byColor = store.get(state.players);
  if (!byColor) { byColor = new Map(); store.set(state.players, byColor); }
  let b = byColor.get(aiColor);
  if (!b) { b = new ExactBelief(aiColor, priorBySeat.get(aiColor) ?? null); byColor.set(aiColor, b); }
  return b;
}
