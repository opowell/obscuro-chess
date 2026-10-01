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
  moveEntries, vectorFromTables, tablesFromVector, NUM_ENTRIES,
} from '../src/movePrior.js';
import { MOVE_TABLES } from '../src/moveTables.js';
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
  const T = MOVE_TABLES;
  const rank = (sq, sign) => (sign > 0 ? idx(sq) : (7 - (idx(sq) >> 3)) * 8 + (idx(sq) & 7));
  // Black pawn on b2 promoting: the pawn leaves its square, the PROMOTED piece
  // arrives, and the promotion's own value is added.
  const pos = fromBoardObject({
    b2: unit('bP', 'black', 'pawn', 'b2'),
    a2: unit('wP', 'white', 'pawn', 'a2'),
    e8: unit('bK', 'black', 'king', 'e8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, null, null);
  const toQueen = scoreMove(pos, mv('b2', 'b1', { promo: 5 }), -1);
  const expected = T.promo.queen + T.pst.queen[rank('b1', -1)] - T.pst.pawn[rank('b2', -1)];
  assert.ok(Math.abs(toQueen - expected) < 1e-9, `promotion = promo value + queen arriving − pawn leaving: ${toQueen} vs ${expected}`);

  // En passant reads the victim from m.ep, not from the (empty) destination.
  const epPos = fromBoardObject({
    b4: unit('bP', 'black', 'pawn', 'b4'),
    a4: unit('wP', 'white', 'pawn', 'a4'),
    e8: unit('bK', 'black', 'king', 'e8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, null, 'a3');
  const ep = scoreMove(epPos, mv('b4', 'a3', { ep: idx('a4') }), -1);
  const epQuiet = T.pst.pawn[rank('a3', -1)] - T.pst.pawn[rank('b4', -1)];
  assert.ok(Math.abs(ep - (epQuiet + T.capture.pawn)) < 1e-9, 'en passant is scored as a pawn capture');

  // Castling moves the king AND the rook, and the castle value is additive.
  const cPos = fromBoardObject({
    e8: unit('bK', 'black', 'king', 'e8'),
    h8: unit('bR', 'black', 'rook', 'h8'),
    e1: unit('wK', 'white', 'king', 'e1'),
  }, { white: {}, black: { kingSide: true, queenSide: false } }, null);
  const castle = mv('e8', 'g8', { castle: 1 });
  const squares = T.pst.king[rank('g8', -1)] - T.pst.king[rank('e8', -1)]
    + T.pst.rook[rank('f8', -1)] - T.pst.rook[rank('h8', -1)];
  assert.ok(Math.abs(scoreMove(cPos, castle, -1) - squares) < 1e-9, 'king and rook both move');
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

test('movePrior: the fitter\'s table entries ARE scoreMove', () => {
  // fit-move-prior.mjs learns the tables against `moveEntries` and production
  // serves them through `scoreMove`; if the two ever disagree, the tables are
  // for a model nobody runs. Pinned over every fog-legal move of a busy
  // position, for the shipped tables and for an arbitrary parameter vector.
  const pos = busy();
  const arbitrary = Float64Array.from({ length: NUM_ENTRIES }, (_, i) => Math.sin(i * 1.7) * 50);
  const { tables, castleBonus } = tablesFromVector(arbitrary);
  const cases = [
    { theta: vectorFromTables(MOVE_TABLES, FITTED_WEIGHTS.castleBonus), w: FITTED_WEIGHTS },
    { theta: arbitrary, w: { tables, castleBonus } },
  ];
  const ix = new Int16Array(8), vx = new Int8Array(8);
  let checked = 0;
  for (const sign of [1, -1]) {
    for (const m of genFogMoves(pos, sign)) {
      const k = moveEntries(pos, m, sign, ix, vx);
      for (const { theta, w } of cases) {
        let dot = 0;
        for (let e = 0; e < k; e++) dot += theta[ix[e]] * vx[e];
        assert.ok(Math.abs(dot - scoreMove(pos, m, sign, w)) < 1e-9,
          `entries·θ must equal scoreMove for ${JSON.stringify(m)}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 40, `the fixture should exercise plenty of moves, got ${checked}`);
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
