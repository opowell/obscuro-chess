// ---------------------------------------------------------------------------
// A distilled leaf evaluator: a small net that answers the question Stockfish is
// currently asked ~1.5M times per measurement run, at a fraction of the cost.
//
// WHY THIS SHAPE. The leaf-depth grid (README) settled that evaluation QUALITY is
// the binding constraint and tree size is not — quadrupling the tree bought
// nothing, while depth 1 → 7 bought 4-5 cp. But depth 7 costs 1031 ms/move
// against 159 at depth 1. So the prize is depth-7 judgement at depth-1 price,
// which is a distillation problem: the engine is the teacher, this is the
// student, and the training data is already flowing (scripts/collect-evals.mjs).
//
// ARCHITECTURE: 768 → H → 1, which is NNUE's first trick and none of its others.
//   • 768 inputs = 12 piece kinds × 64 squares, one-hot per occupied square.
//     A board activates ~32 of them, so the first layer is a SUM OF ~32 COLUMNS
//     rather than a 768×H matrix multiply — that is what makes this affordable
//     at ~90k evaluations per move. Everything else is H×1.
//   • Values are in CENTIPAWNS from the side to move, matching what multiPV
//     hands back, so nothing downstream has to learn a new scale.
//   • ReLU, because the alternative is explaining a tanh's saturation point to
//     every future reader of a clipped cp target.
//
// The net is deliberately NOT a search. It replaces the leaf call, and the CFR
// tree above it is unchanged.
//
// ---------------------------------------------------------------------------
// IT DOES NOT WORK YET, AND THE NUMBERS ARE HERE SO NOBODY REBUILDS IT BLIND.
//
// Trained on 2.02M labelled children (59,384 sampled belief worlds, depth 7) and
// measured against its own teacher on 40 holdout games / 2,145 paired positions:
//
//   config              mean cp   ms/move   best-move%
//   net, 6 rounds          79.7        58        21.4
//   depth 1, 6 rounds      60.7       147        28.2
//   depth 7, 6 rounds      55.2       849        31.8
//
//   paired vs net:  depth 1  −19.05 ± 2.03 (z = −9.40)
//                   depth 7  −24.49 ± 2.05 (z = −11.94)
//
// It is 2.5× cheaper than depth 1 and 19 cp worse than it. The speed cannot be
// spent back into quality either: the leaf-depth grid (ObscuroAgent._leafEval)
// found that quadrupling the tree buys nothing, so more rounds at 58 ms do not
// recover what the leaves gave up.
//
// WHY, and it is structural rather than a tuning failure. This asks a STATIC
// function to reproduce a SEVEN-PLY TACTICAL SEARCH — and depth 7's whole
// advantage in that grid was seeing tactics the tree cannot. Capacity is not the
// missing piece: hidden 128 was no better than 32 (top-1 18.2% vs 20.6%) and
// hidden 512 diverged outright under plain SGD. On holdout ranking the net
// managed 20.6% top-1 agreement with its teacher and Spearman 0.324 — real
// signal, well above material-only (−0.020) and a third of its regret (23.7 cp
// vs 56.0), but nowhere near the teacher.
//
// USEFUL BY-PRODUCT: those cheap ranking metrics PREDICTED the expensive result.
// Screen a candidate with scripts/eval-valuenet.mjs in seconds before spending an
// hour on --grid; 20% top-1 was never going to survive contact.
//
// WHAT WOULD ACTUALLY BE NEEDED: king-relative features (NNUE's real trick,
// HalfKP-style, ~40k inputs rather than 768), an optimiser that does not diverge
// at width (momentum/Adam), and data measured in the hundreds of millions rather
// than 2M. Or a different target entirely — training on GAME OUTCOMES instead of
// search scores, which is the one version that could also fix the blind spot
// this whole path inherits.
// ---------------------------------------------------------------------------
//
// WHAT IT CANNOT LEARN, said here so nobody rediscovers it as a bug: it is
// trained on Stockfish labels, so it inherits Stockfish's blindness to
// information value. It can be cheaper than the teacher and it can cover
// positions the teacher REFUSES (kings adjacent, side-not-to-move in check —
// 6% of sampled belief worlds, where multiPV returns nothing and the old path
// fell back to a static evaluator), but it cannot be wiser about fog than the
// thing that taught it. Training on game outcomes instead is the next step and a
// different objective.
// ---------------------------------------------------------------------------

const PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];
const PIECE_INDEX = new Map(PIECES.map((p, i) => [p, i]));
export const INPUTS = 768;

/** Square name ('e4') → 0..63, a1 = 0. Returns -1 for anything unparseable. */
function squareIndex(sq) {
  if (typeof sq !== 'string' || sq.length < 2) return -1;
  const f = sq.charCodeAt(0) - 97;        // 'a'
  const r = sq.charCodeAt(1) - 49;        // '1'
  return (f >= 0 && f < 8 && r >= 0 && r < 8) ? r * 8 + f : -1;
}

/**
 * The indices this board switches on, from `side`'s point of view.
 *
 * MIRRORED FOR BLACK, so the net only ever learns one side's game: a black-to-
 * move position is flipped rank-wise and its colours swapped, which halves the
 * function to be learned and is why NNUE-style nets do it. Feed a raw black
 * position to a net trained this way and it will answer confidently about the
 * wrong game.
 */
export function features(board, side, out = []) {
  out.length = 0;
  const flip = side === 'b' || side === 'black';
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.alive === false) continue;
    const kind = PIECE_INDEX.get(p.type);
    if (kind === undefined) continue;
    let idx = squareIndex(p.position ?? sq);
    if (idx < 0) continue;
    let mine = (p.ownerId ?? p.color) === (flip ? 'black' : 'white');
    if (flip) idx = (7 - (idx >> 3)) * 8 + (idx & 7);   // mirror ranks
    out.push((mine ? 0 : 6) * 64 + kind * 64 + idx);
  }
  return out;
}

export class ValueNet {
  constructor(hidden = 32, scale = 300) {
    // The net trains on targets normalised to roughly [-1, 1] and multiplies by
    // `scale` on the way out. Not cosmetic: with raw centipawn targets the error
    // term is ~300, so any learning rate that moves the weights at all overshoots
    // — the first version of this trained to a flat mean prediction (val rmse
    // 312 against a 317 constant baseline) purely because of the units.
    this.scale = scale;
    this.h = hidden;
    this.w1 = new Float32Array(INPUTS * hidden);   // column-major: w1[i*h + j]
    this.b1 = new Float32Array(hidden);
    this.w2 = new Float32Array(hidden);
    this.b2 = 0;
    this._acc = new Float32Array(hidden);
    this._idx = [];
  }

  /** Small random init; He-ish scaling for the ReLU layer. */
  randomize(rng = Math.random) {
    const s = Math.sqrt(2 / 32);
    for (let i = 0; i < this.w1.length; i++) this.w1[i] = (rng() * 2 - 1) * s;
    for (let j = 0; j < this.h; j++) this.w2[j] = (rng() * 2 - 1) * s;
  }

  /** Raw (normalised) forward pass — what the trainer optimises. */
  rawIndices(idx) {
    const { h, w1, b1, w2, _acc } = this;
    _acc.set(b1);
    for (let k = 0; k < idx.length; k++) {
      const base = idx[k] * h;
      for (let j = 0; j < h; j++) _acc[j] += w1[base + j];
    }
    let v = this.b2;
    for (let j = 0; j < h; j++) if (_acc[j] > 0) v += w2[j] * _acc[j];
    return v;
  }

  /** Forward pass from pre-computed feature indices. Returns CENTIPAWNS. */
  evalIndices(idx) { return this.rawIndices(idx) * this.scale; }

  evalBoard(board, side) { return this.evalIndices(features(board, side, this._idx)); }

  toJSON() {
    return { hidden: this.h, scale: this.scale, w1: [...this.w1], b1: [...this.b1], w2: [...this.w2], b2: this.b2 };
  }

  static fromJSON(o) {
    const n = new ValueNet(o.hidden, o.scale ?? 300);
    n.w1.set(o.w1); n.b1.set(o.b1); n.w2.set(o.w2); n.b2 = o.b2;
    return n;
  }
}
