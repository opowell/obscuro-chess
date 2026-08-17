// ---------------------------------------------------------------------------
// π(move | position) — the opponent-model that makes the belief a DISTRIBUTION
// instead of a set. See docs/STRENGTH-PLAN.md.
//
// exactBelief.js advances P one opponent ply by expanding every position by
// every fog-legal opponent move. Without a model of how the opponent chooses,
// every resulting state is equally consistent with what we observed and the
// posterior over P is flat. This module supplies the missing conditional: a
// softmax over a cheap per-move score, so "the opponent probably took the free
// queen" becomes a number the belief can carry.
//
// THE HARD CONSTRAINT IS COST. |P| averages ~17k and fog branching is ~30, so
// one sweep scores ~500k moves inside exactBelief's 4s guard. Calling the
// search's static `evaluate(board, color)` per successor — an object-board
// conversion plus a 64-square walk each — is two orders of magnitude too
// expensive. So the score here is computed INCREMENTALLY FROM THE MOVE ITSELF,
// in constant time, on exactBelief's Int8Array(66) representation: everything it
// needs is in the move record plus one array read for the captured piece.
//
//   capture   the victim's material value. Predicted to be the dominant term;
//             measured to be nearly worthless (+0.058 nats alone, and zeroing it
//             costs ~nothing) because a capture on a square we can see is priced
//             by the observation filter before π is consulted.
//   promotion the value gained over the pawn
//   PST delta PST[type][to] − PST[type][from], sharing the evaluator's own
//             tables (pieceTables.js) — this is where the signal actually is: it
//             discriminates among the genuinely hidden QUIET moves
//   castling  a flat bonus, and the single biggest term in the fitted model
//
// Deliberately NOT included: "gives check". It needs an attack test against our
// king square, which is not O(1) on this representation, and the plan calls for
// measuring before paying for it.
//
// TWO APPROXIMATIONS, both known and both load-bearing:
//
//  1. FOG ASYMMETRY. π conditions on the full position p, but the opponent chose
//     their move under their OWN fog and could not see p. A principled prior
//     would score from their information set — another belief computation per
//     node, hopeless at this budget. belief.js makes the same approximation.
//     The visible consequence is `kingCaptureValue` below.
//  2. LEVEL-1 ONLY. The opponent is a fixed static-eval softmax player. They do
//     not model us modelling them. Do not start down the recursive road here.
//
// SCAR TISSUE — read before sharpening anything. belief.js's header records two
// separate incidents where an over-sharp belief prior made the AI WORSE:
// THREAT_BIAS is deliberately modest and MAX_LURKERS exists because
// over-weighting phantom attackers "hallucinates coordinated mating attacks and
// the AI huddles instead of saving real material". A confident wrong belief is
// worse than an honest vague one, so the defaults here are near-uniform and are
// only justified by the log-loss numbers in docs/STRENGTH-PLAN.md — not by
// how reasonable they look.
//
// 2026-07-31 — THE WEIGHTS ARE NOW FITTED, NOT HAND-SET. The original model gave
// every term weight 1 and divided the lot by a single temperature. Fitting the
// same terms as a conditional logit on recorded games (`fit-move-prior.mjs`)
// showed the terms want sharpness spread over a factor of FOURTEEN — the rook
// PST wants τ≈11 and the pawn PST τ≈24, while capture wants τ≈126 and promotion
// τ≈154. One τ had to split that difference, which is both why the model was
// weak (Δ 0.135 nats where the same terms fitted get 0.691) and why lowering τ
// globally fell off a cliff: it drove capture into confidently-wrong long before
// the PST terms were sharp enough.
//
// This does NOT mean sharpening is safe now. Scaling every fitted weight by 1.5
// still costs 0.065 nats. What changed is that there is no longer a knob aimed
// at the cliff: MLE lands on the model's own optimum by construction. The FLOOR
// below bounds what is left. See FITTED_WEIGHTS.
// ---------------------------------------------------------------------------

import { PIECE_VALUE, PST } from './pieceTables.js';
import { param } from './config.js';

// Material value by exactBelief piece code (1..6 = P N B R Q K).
//
// The king is 1000, not PIECE_VALUE.king (20000), and that is the whole fog
// asymmetry in one number. A move that captures OUR king is always pruned by
// exactBelief (the game did not end, so no such move was played), which means
// its π mass is removed as evidence — "you could have taken my king and didn't,
// so this world is unlikely". That inference is only valid if the opponent could
// SEE our king, and under fog they very often could not. At 20000 the softmax
// would put essentially all of a parent's mass on the king capture and annihilate
// any world in which one was available, including true ones. 1000 keeps the
// evidence real but bounded: a parent that could have taken our king loses most
// but not all of its weight.
const VALUE = [0, PIECE_VALUE.pawn, PIECE_VALUE.knight, PIECE_VALUE.bishop,
  PIECE_VALUE.rook, PIECE_VALUE.queen, 1000];

// PST flattened onto exactBelief's square indexing (i = rank*8 + file, rank 0 =
// rank 1) for both colours, since the tables are written from the mover's
// perspective. ChessAgent's pstIndex(sq, color) is (8−r)*8+f for white and
// (r−1)*8+f for black; in index terms that is a rank flip for white and the
// identity for black. Int16Array: values fit in ±50 and the lookup is on the hot
// path.
const PST_BY_SIGN = (() => {
  const CODE_TYPE = [null, 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
  const white = new Int16Array(7 * 64);
  const black = new Int16Array(7 * 64);
  for (let t = 1; t <= 6; t++) {
    const table = PST[CODE_TYPE[t]];
    if (!table) continue;
    for (let i = 0; i < 64; i++) {
      const r = i >> 3, f = i & 7;
      white[t * 64 + i] = table[(7 - r) * 8 + f];
      black[t * 64 + i] = table[r * 8 + f];
    }
  }
  return { white, black };
})();

/**
 * Score one fog-legal move in centipawn-ish units, O(1).
 *
 * `pos` is exactBelief's Int8Array(66) BEFORE the move; `m` is a move record
 * from genFogMoves ({ f, t, promo, dbl, ep, castle }); `sign` is +1 for white,
 * −1 for black. Exported so the calibration harness (and tests) can inspect the
 * model separately from the softmax that turns it into a distribution.
 *
 * `w.pstWeight` may be a NUMBER (one weight for every piece type, the original
 * model) or an ARRAY indexed by piece code 1..6 (the fitted model — the terms
 * span more than a factor of six, and the king's is ~0, so a single number
 * cannot express them). Defaults are all-1 / castleBonus 0, i.e. the raw feature
 * sum, which is what the unit tests pin; the shipped numbers live in
 * FITTED_WEIGHTS.
 */
export function scoreMove(pos, m, sign, w = {}) {
  const captureWeight = w.captureWeight ?? 1;
  const pw = w.pstWeight;
  const promoWeight = w.promoWeight ?? 1;
  const castleBonus = w.castleBonus ?? 0;
  const pstTab = sign > 0 ? PST_BY_SIGN.white : PST_BY_SIGN.black;

  if (m.castle) {
    // Both king and rook move; the king's own PST delta is the dominant part and
    // castling is generically good, so add a flat bonus rather than pretending
    // to price the rook's repositioning. (The fit makes the bonus the single
    // biggest term in the model — see FITTED_WEIGHTS.)
    const kw = pw === undefined ? 1 : (typeof pw === 'number' ? pw : pw[6]);
    return castleBonus + kw * (pstTab[6 * 64 + m.t] - pstTab[6 * 64 + m.f]);
  }

  const mover = pos[m.f];
  const type = mover > 0 ? mover : -mover;
  let s = 0;

  // Capture. En passant takes a pawn that is NOT on the destination square, so
  // read the victim from m.ep when it is set.
  const victim = m.ep >= 0 ? pos[m.ep] : pos[m.t];
  if (victim) s += captureWeight * VALUE[victim > 0 ? victim : -victim];

  // Promotion: the material actually gained.
  if (m.promo) s += promoWeight * (VALUE[m.promo] - VALUE[1]);

  // Piece-square delta. The piece that ARRIVES is the promoted type, so the two
  // ends of the delta can come from different tables. The WEIGHT is the mover's,
  // not the arriving type's — a promotion is a pawn's decision.
  const arriving = m.promo ? m.promo : type;
  const pstWeight = pw === undefined ? 1 : (typeof pw === 'number' ? pw : pw[type]);
  s += pstWeight * (pstTab[arriving * 64 + m.t] - pstTab[type * 64 + m.f]);

  return s;
}

// The model's terms, in the order `moveFeatures` writes them. Exported so the
// fitter can label its output and so a weight vector can never silently
// transpose two terms between training and serving.
export const FEATURE_NAMES = ['capture', 'promo', 'pst.pawn', 'pst.knight',
  'pst.bishop', 'pst.rook', 'pst.queen', 'pst.king', 'castle'];
export const NUM_FEATURES = FEATURE_NAMES.length;

/**
 * The same model as `scoreMove`, DECOMPOSED — fills `out[0..8]` with each term's
 * feature value in centipawn units (the castle indicator is 0/1). By
 * construction
 *
 *     scoreMove(pos, m, sign, w) === Σ_k weightVector(w)[k] * out[k]
 *
 * which is what makes it safe to fit weights offline against these features and
 * serve them through `scoreMove`'s fast path. move-prior.test.js asserts that
 * identity over random positions; if you add a term to one function, the test
 * fails until you add it to the other.
 */
export function moveFeatures(pos, m, sign, out) {
  for (let k = 0; k < NUM_FEATURES; k++) out[k] = 0;
  const pstTab = sign > 0 ? PST_BY_SIGN.white : PST_BY_SIGN.black;
  if (m.castle) {
    out[8] = 1;
    out[7] = pstTab[6 * 64 + m.t] - pstTab[6 * 64 + m.f];
    return out;
  }
  const mover = pos[m.f];
  const type = mover > 0 ? mover : -mover;
  const victim = m.ep >= 0 ? pos[m.ep] : pos[m.t];
  if (victim) out[0] = VALUE[victim > 0 ? victim : -victim];
  if (m.promo) out[1] = VALUE[m.promo] - VALUE[1];
  const arriving = m.promo ? m.promo : type;
  out[2 + type - 1] = pstTab[arriving * 64 + m.t] - pstTab[type * 64 + m.f];
  return out;
}

/** A weights object → the flat vector `moveFeatures` is dotted with. */
export function weightVector(w = {}) {
  const pw = w.pstWeight;
  const pst = t => (pw === undefined ? 1 : (typeof pw === 'number' ? pw : pw[t]));
  return [w.captureWeight ?? 1, w.promoWeight ?? 1,
    pst(1), pst(2), pst(3), pst(4), pst(5), pst(6), w.castleBonus ?? 0];
}

/** The inverse of `weightVector`: a flat vector → a weights object. */
export function weightsFromVector(v, extra = {}) {
  return {
    captureWeight: v[0], promoWeight: v[1],
    pstWeight: [0, v[2], v[3], v[4], v[5], v[6], v[7]],
    castleBonus: v[8], ...extra,
  };
}

// ---------------------------------------------------------------------------
// FITTED_WEIGHTS — regenerate with `node scripts/fit-move-prior.mjs --write`.
//
// Conditional-logit MLE over every fog-legal move list in the corpus. REFITTED
// 2026-08-06 on 246 Chess.com Fog of War games / 14,836 decisions by 192
// different players (`--sessions <crawl>.json`), replacing a 2026-07-31 fit on
// 37 games / 1520 decisions that were one human plus this engine. τ = 100 is not
// a tuned knob: it fixes the unit (weight = logits per centipawn × 100) and
// nothing else, because the sharpness lives per-term. Read the weights as
// effective temperatures — bishop PST τ≈15, rook τ≈22, pawn τ≈35, knight τ≈36,
// queen τ≈60, capture τ≈106, promotion τ≈133.
//
// WHY THE REFIT SHIPPED. Scored on held-out games of the new corpus, which the
// old weights had never seen: move log-loss 2.922 → 2.896 (better in 5 folds of
// 5), and on the gate that actually matters — BELIEF log-loss of the true
// position — 5.229 → 5.161, with the true board's median rank in the belief's
// own ordering improving 33 → 25. `notInP` stayed 0 in every arm.
//
// THE KING WEIGHT IS NOW ~0, AND THAT IS THE INTERESTING PART. The previous fit
// had it at −0.853 and this comment used to explain, at length, that under fog
// players walk kings toward the centre rather than to the corner a normal
// midgame table rewards. THAT FINDING DID NOT REPLICATE. Fitting the term on 8
// disjoint folds of the new corpus gives −0.2, +0.6, 0.0, +0.2, +0.3, −0.2,
// +0.2, −0.4: it flips sign in 5 of 8 and averages 0.03 (τ_eff ≈ 3000, i.e. no
// signal). Every other term is stable in sign and close in magnitude across the
// same folds. So the negative king weight was one player's habit read as a fact
// about fog chess — which the old comment half-anticipated by noting the term
// bought nothing measurable on the board posterior anyway.
//
// Do not "fix" this back to a confident value in either direction without a
// corpus that shows one. move-prior.test.js pins it near zero for that reason.
//
// FLOOR is the mixture with the uniform prior. It COSTS 0.008 nats; what it buys
// is a bound. No legal move can ever be assigned less than floor/|M|, so no
// single confident mistake can annihilate the true world — the failure mode that
// made the τ<60 cliff so steep. Insurance, deliberately bought, not tuned.
// ---------------------------------------------------------------------------
export const FITTED_WEIGHTS = {
  temperature: 100,
  floor: 0.03,
  captureWeight: 0.943,
  promoWeight: 0.753,
  //          -  pawn  knight bishop  rook  queen   king
  pstWeight: [0, 2.887, 2.804, 6.523, 4.509, 1.662, 0.032],
  castleBonus: 245.2,
};

// ---------------------------------------------------------------------------
// RATING_SLOPE — π conditioned on how strong the opponent is, CONTINUOUSLY.
//
// FITTED_WEIGHTS is one model of "the opponent", pooled over everyone in the
// corpus. That is the right default and it is also obviously an approximation:
// a 1500-rated player and a 2400-rated player do not choose moves from the same
// distribution, and π's whole job is to say which move the opponent will pick.
//
// Rating enters as an INTERACTION, not as a bucket. Each feature's weight is a
// straight line in the opponent's rating:
//
//     weight_k(r) = FITTED_WEIGHTS_k + RATING_SLOPE_k · z(r)
//     z(r)        = (r − RATING_PIVOT) / RATING_SCALE
//
// so π stays a conditional logit — `softmax(Σ_k weight_k(r) · f_k)` — and stays
// fittable by the same concave MLE, just over twice as many parameters.
//
// WHY NOT BANDS. Bucketing throws away most of the data for every parameter it
// estimates: split a corpus three ways and each band's nine weights are fitted
// on a third of the decisions, which is how a real effect gets buried under
// estimation variance. The slope form uses EVERY rated decision to estimate
// every slope, has no edges to choose, and cannot produce the discontinuity
// where a 1899-rated opponent and a 1901-rated one are served different models.
// It also cannot have gaps, so there is no fallback rule to get wrong.
//
// RATING_SLOPE IS ALL ZEROS, which is exactly the null the corpus supports:
// serving reduces to FITTED_WEIGHTS at every rating. `fit-move-prior.mjs
// --rating` fits the slopes and only writes them if they beat the flat model on
// HELD-OUT games — a model with twice the parameters always fits training data
// better, so nothing here ships on in-sample improvement.
export const RATING_PIVOT = 2000;
export const RATING_SCALE = 400;
export const RATING_SLOPE = [0, 0, 0, 0, 0, 0, 0, 0, 0];
// How far z may run from the pivot before it is clamped, so an outlier rating (or
// a host passing a nonsense number) cannot extrapolate the fitted line past where
// the corpus went. ±1.5 is roughly 1400–2600 Elo at the shipped pivot/scale — a
// property of the CORPUS, which is why it moves when someone refits on their own.
export const RATING_Z_CLAMP = 1.5;

/** The centered, scaled rating the slopes multiply. 2400 → +1, 1600 → −1. */
export function ratingZ(rating, pivot = RATING_PIVOT, scale = RATING_SCALE) {
  return (rating - pivot) / scale;
}

// SERVE THE MODEL-FREE BASELINE instead of the fitted π — no opponent model at
// all, every fog-legal move equally likely (UNIFORM_PRIOR below).
//
// This is the paper's own setting: Zhang & Sandholm draw search worlds uniformly
// at random from P and model nothing about how the opponent chooses (see
// FogChess.sampleWorlds and presets.js `zhang-sandholm`). It is also the arm every
// measurement of this model is against, which is why it is a switch rather than
// something a caller has to reconstruct out of weights: `floor` cannot reach 1 and
// `temperature: Infinity` is not expressible in a JSON settings file.
//
// Note that uniform π is NOT a flat posterior over P — see UNIFORM_PRIOR.
export const UNIFORM_ONLY = false;

/**
 * The weights to serve against an opponent of the given rating.
 *
 * Returns `base` unchanged when the rating is unknown or the slopes are all
 * zero — the pooled model is the floor, and rating can only ever move it by an
 * amount that was measured. `clamp` bounds z so an outlier rating (or a host
 * passing a nonsense number) cannot extrapolate the line to somewhere the
 * corpus never went (see RATING_Z_CLAMP).
 *
 * The three line parameters are resolved through settings when the caller does
 * not name them, because they describe THE CORPUS THE SLOPES WERE FITTED ON: a
 * host serving its own `chess.MOVE_PRIOR_RATING_SLOPE` has to be able to say
 * which pivot, scale and clamp those slopes were fitted against, or the line is
 * evaluated in the wrong units.
 */
export function weightsForRating(rating, {
  base = FITTED_WEIGHTS, slope = RATING_SLOPE,
  pivot = param('chess.MOVE_PRIOR_RATING_PIVOT', RATING_PIVOT),
  scale = param('chess.MOVE_PRIOR_RATING_SCALE', RATING_SCALE),
  clamp = param('chess.MOVE_PRIOR_RATING_Z_CLAMP', RATING_Z_CLAMP),
} = {}) {
  if (rating == null || !Number.isFinite(rating)) return base;
  if (!slope?.some(x => x !== 0)) return base;
  const z = Math.max(-clamp, Math.min(clamp, ratingZ(rating, pivot, scale)));
  const b = weightVector(base);
  return weightsFromVector(b.map((x, k) => x + (slope[k] ?? 0) * z), {
    temperature: base.temperature, floor: base.floor,
  });
}

/**
 * Build a prior. The returned function fills `out[0..moves.length-1]` with
 * π(m | pos), NORMALIZED so Σ_m π = 1.
 *
 *   prior(pos, moves, sign, out) -> void
 *
 * The batch shape (rather than one call per move returning an unnormalized
 * number) is deliberate: normalizing per parent is not optional — without it a
 * high-branching position hands out more total mass than a cramped one and mass
 * is not conserved — and doing the softmax here means it can subtract the
 * per-parent max, which keeps `temperature` free to be small without overflowing.
 *
 * `temperature` is in the same centipawn-ish units as scoreMove, so τ = 300 makes
 * a pawn capture ~1.4× as likely as a quiet move and a queen capture ~20×.
 * Higher is vaguer; Infinity is exactly UNIFORM_PRIOR.
 *
 * `floor` ∈ [0, 1) mixes in the uniform prior: π = (1−floor)·softmax + floor/|M|.
 * It bounds the damage a confidently wrong model can do to one ply at −log(floor)
 * nats — see FITTED_WEIGHTS. floor = 0 is the pure softmax.
 */
export function makeMovePrior({ temperature = 300, floor = 0, ...weights } = {}) {
  if (!(temperature > 0)) throw new Error('movePrior: temperature must be > 0');
  if (!(floor >= 0 && floor < 1)) throw new Error('movePrior: floor must be in [0, 1)');
  if (temperature === Infinity) return UNIFORM_PRIOR;
  const invT = 1 / temperature;
  const keep = 1 - floor;
  return function prior(pos, moves, sign, out) {
    const n = moves.length;
    let max = -Infinity;
    for (let j = 0; j < n; j++) {
      const s = scoreMove(pos, moves[j], sign, weights) * invT;
      out[j] = s;
      if (s > max) max = s;
    }
    let sum = 0;
    for (let j = 0; j < n; j++) { const e = Math.exp(out[j] - max); out[j] = e; sum += e; }
    const inv = keep / sum, u = floor / n;
    for (let j = 0; j < n; j++) out[j] = out[j] * inv + u;
  };
}

/**
 * The baseline: every fog-legal move equally likely. Note that this is NOT the
 * same thing as a flat posterior over P — see exactBelief's weight bookkeeping.
 * A state reachable from several parents accumulates their mass, and a parent
 * with fewer legal moves passes more mass to each child, so uniform π already
 * yields a genuinely non-uniform distribution over states with no model at all.
 */
export const UNIFORM_PRIOR = function uniformPrior(pos, moves, sign, out) {
  const p = 1 / moves.length;
  for (let j = 0; j < moves.length; j++) out[j] = p;
};
