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
// THE HARD CONSTRAINT IS COST. |P| runs to 10⁶ and fog branching is ~30, so
// one sweep scores millions of moves inside exactBelief's time guard. So the
// score is computed INCREMENTALLY FROM THE MOVE ITSELF, in constant time, on
// exactBelief's Int8Array(66) representation: a few table reads.
//
//   capture   the captured piece type's value
//   promotion the promoted piece type's value
//   squares   table[type][to] − table[type][from] for every piece that moves
//             (the promoted type arrives; castling moves king and rook) — this
//             is where most of the signal is: it discriminates among the
//             genuinely hidden QUIET moves
//   castling  a value of its own
//
// EVERY NUMBER IN THOSE TABLES IS FITTED (src/moveTables.js, written by
// scripts/fit-move-prior.mjs). Until 2026-10-01 the square tables were the
// static evaluator's hand-written piece-square tables, the capture values were
// textbook material with a king hand-set to 1000, and only a weight per term was
// fitted on top.
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
//     node, hopeless at this budget. (A king the opponent can capture is always
//     visible to them — they see every square their pieces can move to — so the
//     king-capture value at least is learned from decisions made in sight of it.)
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

import { param } from './config.js';
import { MOVE_TABLES } from './moveTables.js';

// THE TABLES. Every number the score is built from — a value per (piece type,
// square) from the mover's side of the board, a value per captured piece type,
// per promotion type — lives in MOVE_TABLES (src/moveTables.js), written by
// `fit-move-prior.mjs --write`. Squares are indexed rank*8 + file with rank 0 the
// MOVER's back rank, so one table serves both colours (a b1–c3 and a b8–c6 are
// the same entry pair). Units are logits × FITTED_WEIGHTS.temperature (100).
//
// Compiled once per tables object into flat typed arrays for the hot path.
const compiledTables = new WeakMap();
const TYPES = [null, 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
function compile(tables) {
  let c = compiledTables.get(tables);
  if (c) return c;
  const pst = new Float64Array(7 * 64), capture = new Float64Array(7), promo = new Float64Array(7);
  for (let t = 1; t <= 6; t++) {
    const row = tables.pst?.[TYPES[t]];
    if (row) for (let i = 0; i < 64; i++) pst[t * 64 + i] = row[i] ?? 0;
    capture[t] = tables.capture?.[TYPES[t]] ?? 0;
    if (t >= 2 && t <= 5) promo[t] = tables.promo?.[TYPES[t]] ?? 0;
  }
  c = { pst, capture, promo };
  compiledTables.set(tables, c);
  return c;
}

// A square from the mover's side: white's own back rank is rank 0 already;
// black's is rank 7, so black's squares are mirrored rank-wise.
const rel = (sq, sign) => (sign > 0 ? sq : ((7 - (sq >> 3)) << 3) | (sq & 7));

// A castle's rook squares, absolute: [from, to].
function castleRook(m, sign) {
  const base = sign > 0 ? 0 : 56;
  return m.castle === 1 ? [base + 7, base + 5] : [base, base + 3];
}

/**
 * Score one fog-legal move, O(1): the change in table value of every piece that
 * moves, plus the value of what it captures, of what it promotes to, and of
 * castling itself. In units of logits × temperature.
 *
 * `pos` is exactBelief's Int8Array(66) BEFORE the move; `m` is a move record
 * from genFogMoves ({ f, t, promo, dbl, ep, castle }); `sign` is +1 for white,
 * −1 for black. Exported so the calibration harness (and tests) can inspect the
 * model separately from the softmax that turns it into a distribution.
 *
 * `w` holds per-term MULTIPLIERS on the tables (all 1 in the shipped model, and
 * the handle the rating slopes and ablations turn) and `w.tables` the tables
 * themselves, MOVE_TABLES by default:
 *
 *   captureWeight   × the captured piece's value
 *   promoWeight     × the promoted piece's value
 *   pstWeight[type] × the moving piece's square delta (the MOVER's type, so a
 *                     promotion is a pawn's decision); may be one number
 *   castleBonus       added for castling (a value, not a multiplier)
 */
export function scoreMove(pos, m, sign, w = {}) {
  const T = compile(w.tables ?? MOVE_TABLES);
  const pw = w.pstWeight;
  const pstW = t => (pw === undefined ? 1 : (typeof pw === 'number' ? pw : pw[t]));
  const pst = T.pst;
  if (m.castle) {
    const [rf, rt] = castleRook(m, sign);
    return (w.castleBonus ?? 0)
      + pstW(6) * (pst[6 * 64 + rel(m.t, sign)] - pst[6 * 64 + rel(m.f, sign)])
      + pstW(4) * (pst[4 * 64 + rel(rt, sign)] - pst[4 * 64 + rel(rf, sign)]);
  }
  const mover = pos[m.f];
  const type = mover > 0 ? mover : -mover;
  let s = 0;
  // En passant takes a pawn that is NOT on the destination square, so read the
  // victim from m.ep when it is set.
  const victim = m.ep >= 0 ? pos[m.ep] : pos[m.t];
  if (victim) s += (w.captureWeight ?? 1) * T.capture[victim > 0 ? victim : -victim];
  if (m.promo) s += (w.promoWeight ?? 1) * T.promo[m.promo];
  // The piece that ARRIVES is the promoted type, so the two ends of the delta
  // can come from different tables.
  const arriving = m.promo ? m.promo : type;
  s += pstW(type) * (pst[arriving * 64 + rel(m.t, sign)] - pst[type * 64 + rel(m.f, sign)]);
  return s;
}

// The model's terms, in the order `moveFeatures` writes them. Exported so the
// fitter can label its output and so a weight vector can never silently
// transpose two terms between training and serving.
export const FEATURE_NAMES = ['capture', 'promo', 'pst.pawn', 'pst.knight',
  'pst.bishop', 'pst.rook', 'pst.queen', 'pst.king', 'castle'];
export const NUM_FEATURES = FEATURE_NAMES.length;

/**
 * The same model as `scoreMove`, DECOMPOSED by term — fills `out[0..8]` with each
 * term's value from the tables (the castle term is the 0/1 indicator). By
 * construction
 *
 *     scoreMove(pos, m, sign, w) === Σ_k weightVector(w)[k] * out[k]
 *
 * for the same tables, which is what lets the rating slopes be fitted against
 * these terms and served through scoreMove. move-prior.test.js asserts it.
 */
export function moveFeatures(pos, m, sign, out, tables = MOVE_TABLES) {
  for (let k = 0; k < NUM_FEATURES; k++) out[k] = 0;
  const pst = compile(tables).pst;
  if (m.castle) {
    const [rf, rt] = castleRook(m, sign);
    out[8] = 1;
    out[7] = pst[6 * 64 + rel(m.t, sign)] - pst[6 * 64 + rel(m.f, sign)];
    out[5] = pst[4 * 64 + rel(rt, sign)] - pst[4 * 64 + rel(rf, sign)];
    return out;
  }
  const T = compile(tables);
  const mover = pos[m.f];
  const type = mover > 0 ? mover : -mover;
  const victim = m.ep >= 0 ? pos[m.ep] : pos[m.t];
  if (victim) out[0] = T.capture[victim > 0 ? victim : -victim];
  if (m.promo) out[1] = T.promo[m.promo];
  const arriving = m.promo ? m.promo : type;
  out[2 + type - 1] = pst[arriving * 64 + rel(m.t, sign)] - pst[type * 64 + rel(m.f, sign)];
  return out;
}

// THE TABLES AS ONE PARAMETER VECTOR, for the fitter. Entry layout:
//   pst      (type − 1) · 64 + square-from-the-mover's-side    0 … 383
//   capture  384 + (captured type − 1)                         384 … 389
//   promo    390 + (promoted type − 2)                         390 … 393
//   castle   394                                               (castleBonus)
export const NUM_ENTRIES = 395;
export const ENTRY = { PST: 0, CAPTURE: 384, PROMO: 390, CASTLE: 394 };

/**
 * A move as sparse ±1 counts over the table entries: `scoreMove` with every
 * multiplier at 1 is exactly Σ entry value × count. Writes into idx/val and
 * returns how many entries it wrote (at most 5). The fitter learns the tables
 * against this, and move-prior.test.js pins it to scoreMove.
 *
 * `term`, if given, receives each entry's index into FEATURE_NAMES — the
 * multiplier that scales it in scoreMove — so the per-term multipliers (and the
 * rating slopes on them) can be fitted on top of the tables.
 */
export function moveEntries(pos, m, sign, idx, val, term = null) {
  let n = 0;
  const put = (i, v, k) => { idx[n] = i; val[n] = v; if (term) term[n] = k; n++; };
  if (m.castle) {
    const [rf, rt] = castleRook(m, sign);
    put(5 * 64 + rel(m.t, sign), 1, 7); put(5 * 64 + rel(m.f, sign), -1, 7);
    put(3 * 64 + rel(rt, sign), 1, 5); put(3 * 64 + rel(rf, sign), -1, 5);
    put(ENTRY.CASTLE, 1, 8);
    return n;
  }
  const mover = pos[m.f];
  const type = mover > 0 ? mover : -mover;
  const victim = m.ep >= 0 ? pos[m.ep] : pos[m.t];
  if (victim) put(ENTRY.CAPTURE + (victim > 0 ? victim : -victim) - 1, 1, 0);
  if (m.promo) put(ENTRY.PROMO + m.promo - 2, 1, 1);
  const arriving = m.promo ? m.promo : type;
  put((arriving - 1) * 64 + rel(m.t, sign), 1, 1 + type);
  put((type - 1) * 64 + rel(m.f, sign), -1, 1 + type);
  return n;
}

/** A parameter vector (NUM_ENTRIES long) → tables, and the castle value. */
export function tablesFromVector(v) {
  const pst = {}, capture = {}, promo = {};
  for (let t = 1; t <= 6; t++) {
    pst[TYPES[t]] = Array.from({ length: 64 }, (_, i) => v[(t - 1) * 64 + i]);
    capture[TYPES[t]] = v[ENTRY.CAPTURE + t - 1];
    if (t >= 2 && t <= 5) promo[TYPES[t]] = v[ENTRY.PROMO + t - 2];
  }
  return { tables: { pst, capture, promo }, castleBonus: v[ENTRY.CASTLE] };
}

/** Tables and castle value → a parameter vector. */
export function vectorFromTables(tables, castleBonus = 0) {
  const T = compile(tables);
  const v = new Float64Array(NUM_ENTRIES);
  for (let t = 1; t <= 6; t++) {
    for (let i = 0; i < 64; i++) v[(t - 1) * 64 + i] = T.pst[t * 64 + i];
    v[ENTRY.CAPTURE + t - 1] = T.capture[t];
    if (t >= 2 && t <= 5) v[ENTRY.PROMO + t - 2] = T.promo[t];
  }
  v[ENTRY.CASTLE] = castleBonus;
  return v;
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
// The tables themselves are in src/moveTables.js. What is here:
//
//   temperature  100: the unit the tables are written in (logits × 100). Not a
//                knob — sharpness lives in the fitted table values.
//   floor        the mixture with the uniform prior, π = (1 − floor)·softmax +
//                floor/|M|, FITTED as the weight that maximizes held-out
//                likelihood. It also bounds the damage one confident mistake
//                can do: no legal move gets less than floor/|M|.
//   multipliers  captureWeight, promoWeight, pstWeight[type]: all 1. They scale
//                the tables term by term, which is what the rating slopes and
//                ablations act on.
//   castleBonus  the fitted value of castling, in table units.
//
// REFITTED 2026-10-01 on 2,872 Chess.com Fog of War games / 182,919 decisions
// (fow-crawl-2026-08-06, 1,604 players), now fitting every table entry instead
// of nine weights on hand-written tables. Held out (3-fold CV by game): move
// log-loss 3.002 → 2.661 nats; belief log-loss of the true position, paired turn
// by turn over ~4,600 turns, 0.97–1.09 nats better (± 0.03; two runs — the gate
// is not exactly repeatable, since past the time guard P is a sample), and the
// true position lost from a sampled P on 66–72 turns where the old tables kept
// it, against 277–324 the other way. See scripts/fit-move-prior.mjs.
// ---------------------------------------------------------------------------
export const FITTED_WEIGHTS = {
  temperature: 100,
  floor: 0.0391,
  captureWeight: 1,
  promoWeight: 1,
  //          -  pawn  knight bishop  rook  queen   king
  pstWeight: [0, 1, 1, 1, 1, 1, 1],
  castleBonus: 302.4,
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
