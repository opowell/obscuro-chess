// ---------------------------------------------------------------------------
// The move prior as a MODEL, separate from the belief plumbing that consumes it
// (exact-belief.test.js). Two properties matter: it is a proper conditional
// distribution over each position's own move list, and its ordering matches what
// a chess player would actually do.
// ---------------------------------------------------------------------------

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  makeMovePrior, UNIFORM_PRIOR, scoreMove, moveFeatures, weightVector,
  weightsFromVector, NUM_FEATURES, FITTED_WEIGHTS,
} from '../src/movePrior.js';
import { fromBoardObject, genFogMoves, getDefaultMovePrior } from '../src/exactBelief.js';

const unit = (id, ownerId, type, position) => ({ id, ownerId, type, position, alive: true });
const idx = (sq) => (sq.charCodeAt(1) - 49) * 8 + (sq.charCodeAt(0) - 97);
const mv = (from, to, extra = {}) =>
  ({ f: idx(from), t: idx(to), promo: 0, dbl: false, ep: -1, castle: 0, ...extra });

// Black rook on d8, a white queen on d4 for it to take, and a white pawn on h2
// well out of the way. Black to move.
function fixture() {
  return fromBoardObject({
    d8: unit('bR', 'black', 'rook', 'd8'),
    e8: unit('bK', 'black', 'king', 'e8'),
    d4: unit('wQ', 'white', 'queen', 'd4'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, null, null);
}

test('movePrior: normalized over each position\'s own move list', () => {
  const pos = fixture();
  const moves = [mv('d8', 'd7'), mv('d8', 'd6'), mv('d8', 'd4'), mv('d8', 'c8')];
  const out = new Float64Array(8);
  for (const prior of [UNIFORM_PRIOR, makeMovePrior({ temperature: 300 }), makeMovePrior({ temperature: 40 })]) {
    prior(pos, moves, -1, out);
    let sum = 0;
    for (let i = 0; i < moves.length; i++) {
      assert.ok(out[i] >= 0 && out[i] <= 1, `π in [0,1], got ${out[i]}`);
      sum += out[i];
    }
    assert.ok(Math.abs(sum - 1) < 1e-12, `Σπ = 1, got ${sum}`);
    // Only the first `moves.length` slots are written — the scratch buffer the
    // belief passes in is longer than the move list and reused across parents.
    assert.equal(out[moves.length + 1], 0, 'nothing written past the move list');
  }
});

test('movePrior: taking the free queen is the most likely move', () => {
  const pos = fixture();
  const moves = [mv('d8', 'd7'), mv('d8', 'd6'), mv('d8', 'd4'), mv('d8', 'c8')];
  const out = new Float64Array(8);
  makeMovePrior({ temperature: 300 })(pos, moves, -1, out);
  const best = [...out.subarray(0, moves.length)];
  assert.equal(best.indexOf(Math.max(...best)), 2, 'Rxd4 is the mode of the distribution');
  // And it is not a rounding-level preference.
  assert.ok(best[2] > best[0] * 5, `capture should dominate a quiet move: ${best[2]} vs ${best[0]}`);
});

test('movePrior: temperature controls sharpness, and uniform is the τ→∞ limit', () => {
  const pos = fixture();
  const moves = [mv('d8', 'd7'), mv('d8', 'd4')];
  const out = new Float64Array(4);
  const capAt = (t) => { makeMovePrior({ temperature: t })(pos, moves, -1, out); return out[1]; };
  const sharp = capAt(50), mid = capAt(300), vague = capAt(5000);
  assert.ok(sharp > mid && mid > vague, `sharper τ concentrates: ${sharp} > ${mid} > ${vague}`);
  assert.ok(vague > 0.5 && vague < 0.6, `τ→∞ approaches uniform (0.5), got ${vague}`);
  assert.equal(makeMovePrior({ temperature: Infinity }), UNIFORM_PRIOR, 'τ=∞ IS the uniform prior');
  assert.throws(() => makeMovePrior({ temperature: 0 }), /temperature/);
});

test('movePrior: promotion, en passant and castling all price correctly', () => {
  // Black pawn on b2 promoting, and a white pawn on a2 it can take en route.
  const pos = fromBoardObject({
    b2: unit('bP', 'black', 'pawn', 'b2'),
    a2: unit('wP', 'white', 'pawn', 'a2'),
    e8: unit('bK', 'black', 'king', 'e8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, null, null);
  const quiet = scoreMove(pos, mv('b2', 'b1'), -1);
  const toQueen = scoreMove(pos, mv('b2', 'b1', { promo: 5 }), -1);
  const toKnight = scoreMove(pos, mv('b2', 'b1', { promo: 2 }), -1);
  assert.ok(toQueen > toKnight, 'a queen is worth more than a knight');
  assert.ok(toQueen > quiet + 700, 'promotion is worth roughly a queen minus a pawn');

  // En passant reads the victim from m.ep, not from the (empty) destination.
  const epPos = fromBoardObject({
    b4: unit('bP', 'black', 'pawn', 'b4'),
    a4: unit('wP', 'white', 'pawn', 'a4'),
    e8: unit('bK', 'black', 'king', 'e8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, null, 'a3');
  const ep = scoreMove(epPos, mv('b4', 'a3', { ep: idx('a4') }), -1);
  const push = scoreMove(epPos, mv('b4', 'b3'), -1);
  assert.ok(ep > push + 50, `en passant is scored as a pawn capture: ${ep} vs ${push}`);

  // Castling has no victim and no `from` piece delta worth pricing beyond the
  // king's own; it must at least come out finite and respect the bonus knob.
  const cPos = fromBoardObject({
    e8: unit('bK', 'black', 'king', 'e8'),
    h8: unit('bR', 'black', 'rook', 'h8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, { white: {}, black: { kingSide: true, queenSide: false } }, null);
  const castle = mv('e8', 'g8', { castle: 1 });
  assert.ok(Number.isFinite(scoreMove(cPos, castle, -1)));
  assert.equal(
    scoreMove(cPos, castle, -1, { castleBonus: 60 }) - scoreMove(cPos, castle, -1),
    60, 'castleBonus is additive');
});

test('movePrior: the PST is oriented per colour', () => {
  // A knight to the centre is good for both sides; the SAME square must score as
  // an advance for whoever is moving toward it. b1-c3 for white and b8-c6 for
  // black are mirror images and must score identically.
  const w = fromBoardObject({
    b1: unit('wN', 'white', 'knight', 'b1'),
    e1: unit('wK', 'white', 'king', 'e1'),
    e8: unit('bK', 'black', 'king', 'e8'),
  }, null, null);
  const b = fromBoardObject({
    b8: unit('bN', 'black', 'knight', 'b8'),
    e1: unit('wK', 'white', 'king', 'e1'),
    e8: unit('bK', 'black', 'king', 'e8'),
  }, null, null);
  const wDev = scoreMove(w, mv('b1', 'c3'), 1);
  const bDev = scoreMove(b, mv('b8', 'c6'), -1);
  assert.equal(wDev, bDev, 'mirrored development scores the same for both colours');
  assert.ok(wDev > 0, 'and developing a knight off the back rank is an improvement');
});

// A busy position with something of every kind in it: captures for both sides,
// a pawn one square from promoting, an en-passant target, and castling rights.
function busy() {
  return fromBoardObject({
    a1: unit('wR', 'white', 'rook', 'a1'),
    e1: unit('wK', 'white', 'king', 'e1'),
    h1: unit('wR', 'white', 'rook', 'h1'),
    c3: unit('wN', 'white', 'knight', 'c3'),
    d4: unit('wQ', 'white', 'queen', 'd4'),
    b5: unit('wP', 'white', 'pawn', 'b5'),
    g2: unit('wP', 'white', 'pawn', 'g2'),
    a7: unit('bP', 'black', 'pawn', 'a7'),
    c5: unit('bP', 'black', 'pawn', 'c5'),
    d7: unit('bB', 'black', 'bishop', 'd7'),
    e8: unit('bK', 'black', 'king', 'e8'),
    h8: unit('bR', 'black', 'rook', 'h8'),
    f2: unit('bP', 'black', 'pawn', 'f2'),
  }, { white: { kingSide: true, queenSide: true }, black: { kingSide: true, queenSide: false } }, 'c6');
}

test('movePrior: moveFeatures · weightVector IS scoreMove', () => {
  // The fitter (fit-move-prior.mjs) learns weights against `moveFeatures` and
  // production serves them through `scoreMove`. If the two ever describe
  // different models, the weights are silently for a model nobody runs — so the
  // identity is pinned here over every fog-legal move of a busy position.
  const pos = busy();
  const out = new Float64Array(NUM_FEATURES);
  const cases = [
    {}, { captureWeight: 0.7, promoWeight: 0.6, pstWeight: 2, castleBonus: 202.4 },
    FITTED_WEIGHTS, { pstWeight: [0, 4.2, 2.6, 4.1, 9.5, 2.0, -0.85], castleBonus: 33 },
  ];
  let checked = 0;
  for (const sign of [1, -1]) {
    for (const m of genFogMoves(pos, sign)) {
      moveFeatures(pos, m, sign, out);
      for (const w of cases) {
        const v = weightVector(w);
        let dot = 0;
        for (let k = 0; k < NUM_FEATURES; k++) dot += v[k] * out[k];
        assert.ok(Math.abs(dot - scoreMove(pos, m, sign, w)) < 1e-9,
          `features·weights must equal scoreMove for ${JSON.stringify(m)}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 40, `the fixture should exercise plenty of moves, got ${checked}`);
  // And the round trip through the fitter's flat representation is lossless.
  const w = weightsFromVector(weightVector(FITTED_WEIGHTS));
  for (const m of genFogMoves(pos, 1)) {
    assert.ok(Math.abs(scoreMove(pos, m, 1, w) - scoreMove(pos, m, 1, FITTED_WEIGHTS)) < 1e-9);
  }
});

test('movePrior: pstWeight may be per piece type, and picks the MOVER\'s', () => {
  const pos = busy();
  const nMove = { f: idx('c3'), t: idx('e4'), promo: 0, dbl: false, ep: -1, castle: 0 };
  const flat = scoreMove(pos, nMove, 1, { pstWeight: 3 });
  const perType = scoreMove(pos, nMove, 1, { pstWeight: [0, 1, 3, 1, 1, 1, 1] });
  assert.equal(flat, perType, 'a knight move reads the knight slot');
  // A promotion is a pawn's decision even though a queen arrives: the weight
  // comes from the mover, the TABLE from the arriving piece.
  const promo = { f: idx('f2'), t: idx('f1'), promo: 5, dbl: false, ep: -1, castle: 0 };
  assert.equal(
    scoreMove(pos, promo, -1, { pstWeight: [0, 2, 1, 1, 1, 1, 1] }),
    scoreMove(pos, promo, -1, { pstWeight: 2 }),
    'promotion uses the pawn weight, not the queen weight');
});

test('movePrior: the floor bounds how wrong one ply can be', () => {
  const pos = fixture();
  const moves = [mv('d8', 'd7'), mv('d8', 'd6'), mv('d8', 'd4'), mv('d8', 'c8')];
  const out = new Float64Array(8);
  // τ=1 is absurdly sharp on purpose: without a floor the non-capture moves
  // underflow to 0, which is exactly how a confident prior annihilates the true
  // world. The floor is what stops that being possible at all.
  makeMovePrior({ temperature: 1 })(pos, moves, -1, out);
  assert.equal(out[0], 0, 'unfloored, a sharp prior really does hand out zero');

  const floor = 0.03;
  makeMovePrior({ temperature: 1, floor })(pos, moves, -1, out);
  let sum = 0;
  for (let i = 0; i < moves.length; i++) {
    assert.ok(out[i] >= floor / moves.length - 1e-12,
      `no move below floor/|M| = ${floor / moves.length}, got ${out[i]}`);
    sum += out[i];
  }
  assert.ok(Math.abs(sum - 1) < 1e-12, `still normalized, got ${sum}`);
  // -log(floor/|M|) is the worst a single ply can cost, ~4.9 nats here.
  assert.ok(-Math.log(out[0]) < 5, 'worst-case per-ply log-loss is bounded');
  assert.throws(() => makeMovePrior({ temperature: 100, floor: 1 }), /floor/);
});

test('movePrior: the shipped model has NO opinion about where kings go', () => {
  // This test used to assert the opposite sign, and the story behind the change
  // is worth keeping. The 2026-07-31 fit (37 games, one human plus this engine)
  // put the king PST weight at −0.853, and that was read as a fact about fog
  // chess: players walk kings toward the centre, not to the corner ChessAgent's
  // normal midgame table rewards.
  //
  // Refitting on 246 Chess.com games by 192 players did not reproduce it. Across
  // 8 disjoint folds the term came out −0.2, +0.6, 0.0, +0.2, +0.3, −0.2, +0.2,
  // −0.4 — sign-flipping in 5 of 8, mean 0.03 — while every other term held its
  // sign and rough magnitude. One player's habit, not a property of the game.
  //
  // So what is pinned now is the ABSENCE of a claim: the term must stay near
  // zero. A confident value in EITHER direction needs a corpus that shows one.
  assert.ok(Math.abs(FITTED_WEIGHTS.pstWeight[6]) < 0.5,
    `king PST weight should be ~0, got ${FITTED_WEIGHTS.pstWeight[6]}`);

  // Concretely: king moves are priced almost entirely by the other terms, so
  // centralising and retreating score within a rounding error of each other.
  const pos = fromBoardObject({
    e4: unit('wK', 'white', 'king', 'e4'),
    a8: unit('bK', 'black', 'king', 'a8'),
  }, null, null);
  const toCentre = scoreMove(pos, mv('e4', 'd5'), 1, FITTED_WEIGHTS);
  const toEdge = scoreMove(pos, mv('e4', 'e3'), 1, FITTED_WEIGHTS);
  assert.ok(Math.abs(toCentre - toEdge) < 10,
    `neither king move is strongly preferred: ${toCentre} vs ${toEdge}`);
  // The hand model, by construction, had a strong opinion — that is the contrast.
  const handCentre = scoreMove(pos, mv('e4', 'd5'), 1);
  const handEdge = scoreMove(pos, mv('e4', 'e3'), 1);
  assert.ok(handCentre < handEdge, 'the unfitted model still prefers the corner');
  assert.ok(Math.abs(handCentre - handEdge) > Math.abs(toCentre - toEdge),
    'and it holds that opinion more strongly than the fitted model holds any');
});

test('movePrior: production actually serves the fitted model', () => {
  // The weights are worth nothing if getExactBelief still hands out the old one.
  // Compare distributions rather than function identity, so this survives
  // exactBelief building its prior however it likes.
  const pos = busy();
  const moves = genFogMoves(pos, -1);
  const a = new Float64Array(moves.length), b = new Float64Array(moves.length);
  getDefaultMovePrior()(pos, moves, -1, a);
  makeMovePrior(FITTED_WEIGHTS)(pos, moves, -1, b);
  for (let j = 0; j < moves.length; j++) {
    assert.ok(Math.abs(a[j] - b[j]) < 1e-12, 'the default π is FITTED_WEIGHTS');
  }
  // …and the floor is part of what ships, not just of the constant.
  for (let j = 0; j < moves.length; j++) {
    assert.ok(a[j] >= FITTED_WEIGHTS.floor / moves.length - 1e-12, 'floored in production');
  }
});

test('movePrior: the king is priced below a queen, on purpose', () => {
  // Capturing our king is always PRUNED by exactBelief, so its π mass is removed
  // as evidence ("you could have taken my king and didn't"). Under fog the
  // opponent often could not see the king, so that evidence must stay bounded —
  // at PIECE_VALUE.king (20000) it would annihilate any world offering one.
  const pos = fromBoardObject({
    d8: unit('bR', 'black', 'rook', 'd8'),
    d4: unit('wQ', 'white', 'queen', 'd4'),
    d1: unit('wK', 'white', 'king', 'd1'),
    e8: unit('bK', 'black', 'king', 'e8'),
  }, null, null);
  const takeQueen = scoreMove(pos, mv('d8', 'd4'), -1);
  const takeKing = scoreMove(pos, mv('d8', 'd1'), -1);
  assert.ok(takeKing > takeQueen, 'the king is still the most valuable capture');
  assert.ok(takeKing < takeQueen * 3, `but not by orders of magnitude: ${takeKing} vs ${takeQueen}`);
});
