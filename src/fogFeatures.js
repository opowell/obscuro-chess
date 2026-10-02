// ---------------------------------------------------------------------------
// The inputs of the fog value model (src/fogValueModel.js), computed one way for
// both the fit (scripts/fit-fog-value.mjs) and the search's leaves
// (src/ObscuroAgent.js), so the model is applied to exactly what it was fitted on.
//
// For a position with side X to move and opponent Y:
//
//   L      logit of the engine's expected score for X, clipped to (clip, 1 − clip)
//   e      that expected score itself (with L, a more flexible curve in it)
//   m      material on the board / 78 (the phase of the game)
//   bal    X's material minus Y's / 39: the engine's score saturates once a
//          position is clearly decided, and past that point it no longer tells
//          a bishop up from a bishop down — under fog, where a decided position
//          is still won ~15% of the time, that difference still matters
//   w      1 if X is white
//   visX   squares X can see / 64           visY   squares Y can see / 64
//   seenX  share of Y's pieces X can see    seenY  share of X's pieces Y can see
//
// The last four are the INFORMATION STATE: Stockfish values the true board as if
// both sides saw all of it, so without them two positions that differ only in
// who can see what get the same value, and a move is never credited for what it
// reveals or hides beyond the search's horizon. Visibility is the game's own
// (board.js getVisibleSquares); nothing here is a weight — the model's
// coefficients are fitted to how fog games ended. The shipped model does not
// use them: they fit slightly better and played worse (docs/PARAMETERS.md §2.1),
// and they are only computed when a model names one.
// ---------------------------------------------------------------------------

import { getVisibleSquares } from './board.js';

const MATERIAL = { pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9 };

export function materialOf(board) {
  let m = 0;
  for (const sq of Object.keys(board ?? {})) { const p = board[sq]; if (p) m += MATERIAL[p.type] ?? 0; }
  return m;
}

// `color`'s material minus the other side's, / 39.
function balanceOf(board, color) {
  let b = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq]; if (!p) continue;
    b += (p.ownerId === color ? 1 : -1) * (MATERIAL[p.type] ?? 0);
  }
  return b / 39;
}

// Visible-square count and how many of the other side's pieces fall in it.
function sight(board, color) {
  const vis = getVisibleSquares(board, color);
  let seen = 0, total = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.ownerId === color) continue;
    total++;
    if (vis.has(sq)) seen++;
  }
  return { vis: vis.size / 64, seen: total ? seen / total : 0 };
}

/** Every feature the fit may use, by name, for side `toMove` in `board` with
 *  engine expected score `e` for it. */
export function fogFeatures(e, board, toMove, clip, names) {
  const x = Math.min(1 - clip, Math.max(clip, e));
  const L = Math.log(x / (1 - x)), m = materialOf(board) / 78;
  const f = { 1: 1, L, Lm: L * m, e: x, em: x * m, m, w: toMove === 'white' ? 1 : 0, bal: balanceOf(board, toMove) };
  // Visibility is only computed when the model uses it.
  if (!names || names.some(n => INFO.has(n))) {
    const X = sight(board, toMove), Y = sight(board, toMove === 'white' ? 'black' : 'white');
    Object.assign(f, {
      visX: X.vis, visY: Y.vis, seenX: X.seen, seenY: Y.seen,
      visD: X.vis - Y.vis, seenD: X.seen - Y.seen,
    });
  }
  return f;
}

const INFO = new Set(['visX', 'visY', 'seenX', 'seenY', 'visD', 'seenD']);

/** P(the side to move wins under fog) under `model` ({ clip, features, beta }). */
export function fogWinProbabilityWith(model, e, board, toMove) {
  const f = fogFeatures(e, board, toMove, model.clip, model.features);
  let z = 0;
  for (let k = 0; k < model.features.length; k++) z += model.beta[k] * f[model.features[k]];
  return 1 / (1 + Math.exp(-z));
}
