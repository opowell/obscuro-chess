// ---------------------------------------------------------------------------
// π(move | position) — the opponent-model that makes the belief a DISTRIBUTION
// instead of a set. See OBSCURO-MOVE-PRIOR-PLAN.md.
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
// only justified by the log-loss numbers in OBSCURO-MOVE-PRIOR-PLAN.md — not by
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
 * model) or an ARRAY indexed by piece code 1..6 (the fitted model — the king's
 * weight comes out negative, so a single number cannot express it). Defaults are
 * all-1 / castleBonus 0, i.e. the raw feature sum, which is what the unit tests
 * pin; the shipped numbers live in FITTED_WEIGHTS.
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
// Conditional-logit MLE over every fog-legal move list in sessions/, fitted
// 2026-07-31 on 37 games / 1520 decisions. τ = 100 is not a tuned knob: it fixes
// the unit (weight = logits per centipawn × 100) and nothing else, because the
// sharpness now lives per-term. Read the weights as effective temperatures —
// rook PST τ≈11, pawn τ≈24, bishop τ≈24, knight τ≈38, queen τ≈48, capture τ≈126,
// promotion τ≈154 — which is the 14× spread one τ could not cover.
//
// THE KING WEIGHT IS NEGATIVE ON PURPOSE. ChessAgent's king table is a normal
// midgame table that pushes the king to the corner; under fog, players (human
// and Obscuro alike) walk kings toward the centre instead, and the fit says so.
// It is the term the old model predicted worst — at τ=200 it bought 0.08 nats on
// king moves against 0.41–0.53 on every other piece. Note this barely moves the
// BOARD posterior (ablating it costs nothing measurable; king moves are 9% of
// plies and the observation filter pins kings down anyway). It is here because
// it is right, not because it is where the win came from — that was `castle`.
//
// FLOOR is the mixture with the uniform prior. It COSTS 0.008 nats; what it buys
// is a bound. No legal move can ever be assigned less than floor/|M|, so no
// single confident mistake can annihilate the true world — the failure mode that
// made the τ<60 cliff so steep. Insurance, deliberately bought, not tuned.
// ---------------------------------------------------------------------------
export const FITTED_WEIGHTS = {
  temperature: 100,
  floor: 0.03,
  captureWeight: 0.791,
  promoWeight: 0.651,
  //          -  pawn  knight bishop  rook  queen   king
  pstWeight: [0, 4.252, 2.652, 4.115, 9.523, 2.064, -0.853],
  castleBonus: 202.5,
};

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
