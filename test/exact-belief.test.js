// ---------------------------------------------------------------------------
// Exact belief (position set P) tests. Two things are guarded here:
//
//   1. THE INVARIANT — at every one of the AI's turns in a real game, the TRUE
//      position is a member of P. This is the do-not-regress guard for the whole
//      subsystem; the replays below are real recorded fog games.
//   2. THE POSTERIOR — P is a distribution, not just a set, so the weights must
//      stay a distribution (Σ = 1) through every operation, colliding histories
//      must SUM rather than drop, and adding weights must not have changed which
//      positions are in the set.
// ---------------------------------------------------------------------------

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FogChess } from '../src/FogChess.js';
import { ExactBelief, fromBoardObject, toBoardObject } from '../src/exactBelief.js';
import { replayBelief, placementSig } from '../src/beliefCalibration.js';
import { makeMovePrior, UNIFORM_PRIOR } from '../src/movePrior.js';

// Three real recorded fog games, shipped with the tests so the invariant below
// is checkable from a bare clone (they are move logs and nothing else).
const SESSIONS = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const session = (file) => JSON.parse(readFileSync(join(SESSIONS, file), 'utf8'));

// The replay walk itself lives in beliefCalibration.js, shared with
// calibrate-belief.mjs — the invariant and the calibration numbers are the same
// measurement, asserted here and reported there.
function replayWithTracker(file, aiColor, opts = {}) {
  const r = replayBelief(session(file), aiColor, opts);
  assert.equal(r.gaveUpAtPly, null, `tracker gave up at ply ${r.gaveUpAtPly}`);
  for (const t of r.turns) {
    assert.ok(t.found, `true position not in P at ply ${t.ply} (|P|=${t.size})`);
  }
  return { tracker: r.tracker, turns: r.turns, sizes: r.turns.map(t => t.size) };
}

test('exact belief: P always contains the true position (febb71bf replay)', () => {
  const { sizes } = replayWithTracker('2026-07-13T12-59-56-febb71bf.json', 'black');
  assert.ok(sizes.length >= 8, 'should have tracked several turns');
  assert.ok(sizes[0] >= 1, 'first turn has at least the true position');
});

test('exact belief: P always contains the true position (befd4820 replay)', () => {
  const { sizes } = replayWithTracker('2026-07-12T23-27-55-befd4820.json', 'black');
  assert.ok(sizes.length >= 8, 'should have tracked several turns');
});

test('exact belief: a real move prior does not change WHICH positions are in P', () => {
  // The prior reweighs the set; it must not add or remove members. Anything else
  // would mean the observation filter had become prior-dependent, which is how a
  // belief starts excluding the truth.
  const file = '2026-07-13T12-59-56-febb71bf.json';
  const flat = replayWithTracker(file, 'black', { movePrior: UNIFORM_PRIOR });
  const sharp = replayWithTracker(file, 'black', { movePrior: makeMovePrior({ temperature: 100 }) });
  assert.deepEqual(sharp.sizes, flat.sizes, '|P| is identical at every turn');
  const sig = (t) => t.positions.map(p => [...p].join(',')).sort().join('|');
  assert.equal(sig(sharp.tracker), sig(flat.tracker), 'and so are the members themselves');
});

test('exact belief: the posterior is a distribution, and not a flat one', () => {
  // UNIFORM π on purpose: the claim being tested is that the MECHANISM alone
  // (colliding histories summing, branching, the observation filter) makes the
  // posterior non-flat, with no opponent model involved.
  const { tracker, turns } = replayWithTracker('2026-07-14T07-37-02-6f908d7b.json', 'black',
    { movePrior: UNIFORM_PRIOR });
  assert.ok(turns.length >= 20, 'should have tracked a long game');
  const W = tracker.weights;
  assert.equal(W.length, tracker.positions.length, 'one weight per position');
  const sum = [...W].reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `weights sum to 1, got ${sum}`);
  assert.ok([...W].every(w => w >= 0), 'no negative weights');
  // Uniform π still yields a NON-uniform posterior over states: mass accumulates
  // on states reachable by several histories, and low-branching parents pass more
  // to each child. If this ever ties, the collision accumulation has regressed.
  const flat = 1 / W.length;
  assert.ok([...W].some(w => Math.abs(w - flat) > flat * 0.01),
    'the posterior over P must not be flat');
});

test('exact belief: colliding histories SUM their weight instead of dropping', () => {
  // The line this pins down is the one that used to make the posterior flat:
  // dedupe by DROPPING collisions. Two parents, one opponent ply, arranged so
  // some successors are reachable from both parents and some from only one.
  //
  // White has a lone king on e1 and can see nothing past its own neighbours, so
  // every placement below is hidden and consistent. Black is a lone king on d8 in
  // one parent and f8 in the other; both can step to e8 (and to e7), while c8 is
  // reachable only from d8 and g8 only from f8. Each parent has exactly 5 moves,
  // so under uniform π a two-history successor must weigh exactly twice a
  // one-history one.
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const wK = { id: 'wK', ownerId: 'white', type: 'king', position: 'e1', alive: true };
  const bKAt = (sq) => ({ id: 'bK', ownerId: 'black', type: 'king', position: sq, alive: true });
  const board = { e1: wK, d8: bKAt('d8') };
  const base = FogChess.createInitialState(players, { fogOfWar: true, fog: true });
  const view = FogChess.getVisibleState(
    { ...base, board, units: Object.values(board), turnNumber: 99 }, 'white');

  // UNIFORM π, so the arithmetic below is exactly 1/5 per move and the test is
  // about the accumulation rather than the model. (Production defaults to τ=200.)
  const b = new ExactBelief('white', UNIFORM_PRIOR);
  b.exact = true;
  b.firstTurnDone = true;
  b.positions = [
    fromBoardObject({ e1: wK, d8: bKAt('d8') }, null, null),
    fromBoardObject({ e1: wK, f8: bKAt('f8') }, null, null),
  ];
  b.weights = Float64Array.of(0.5, 0.5);
  b.beginTurn(view, 99);
  assert.equal(b.exact, true, 'the sweep should survive');
  assert.equal(b.positions.length, 8, 'c8 c7 d7 e7 e8 f7 g7 g8 — 10 moves, 8 distinct states');

  const sum = [...b.weights].reduce((a, x) => a + x, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `weights renormalize to 1, got ${sum}`);

  const at = (sq) => {
    let w = 0;
    for (let i = 0; i < b.positions.length; i++) {
      if (toBoardObject(b.positions[i])[sq]?.ownerId === 'black') w += b.weights[i];
    }
    return w;
  };
  const e8 = at('e8'), e7 = at('e7'), c8 = at('c8'), g8 = at('g8');
  assert.ok(Math.abs(c8 - 0.1) < 1e-9, `one history → 0.5 × 1/5 = 0.1, got ${c8}`);
  assert.ok(Math.abs(g8 - 0.1) < 1e-9, `one history → 0.1, got ${g8}`);
  assert.ok(Math.abs(e8 - 0.2) < 1e-9, `two histories SUM to 0.2, got ${e8}`);
  assert.ok(Math.abs(e7 - 0.2) < 1e-9, `two histories SUM to 0.2, got ${e7}`);
});

test('exact belief: commitOurMove keeps the weights a distribution', () => {
  const { tracker } = replayWithTracker('2026-07-12T23-27-55-befd4820.json', 'black');
  // The replay ends with a commitOurMove, so the tracker's current state is the
  // post-commit one.
  const sum = [...tracker.weights].reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `post-commit weights sum to 1, got ${sum}`);
  assert.equal(tracker.weights.length, tracker.positions.length);
});

test('exact belief: the sampling exponent α switches weighted vs uniform draws', () => {
  // Both modes of `sampleAlpha`, set explicitly rather than inherited: production
  // ships α=0 (uniform — weighted sampling measured worse in play, see exactBelief.js)
  // but the weighted path must keep working, since the whole reason α is a knob is
  // that a higher-powered strength measurement could turn it back on.
  const b = new ExactBelief('white');
  b._alpha = 1;
  b.exact = true;
  b.positions = [0, 1, 2, 3].map(i => {
    const p = fromBoardObject({
      e1: { id: 'wK', ownerId: 'white', type: 'king', position: 'e1', alive: true },
      ['abcd'[i] + '8']: { id: 'bK', ownerId: 'black', type: 'king', position: 'abcd'[i] + '8', alive: true },
    }, null, null);
    return p;
  });
  b.weights = Float64Array.of(0.97, 0.01, 0.01, 0.01);
  let heavy = 0;
  const trials = 400;
  for (let t = 0; t < trials; t++) {
    const [pick] = b.samplePositions(1, mulberry(t + 1));
    if (pick.board.a8) heavy++;
  }
  assert.ok(heavy > trials * 0.8, `the 97% world should dominate a size-1 draw, got ${heavy}/${trials}`);
  // Drawing the whole set still returns the whole set, weights notwithstanding.
  assert.equal(b.samplePositions(4).length, 4);

  // α = 0 — the SHIPPED default — ignores the weights entirely, so each of the four
  // worlds should come up about a quarter of the time even though one holds 97% of
  // the posterior.
  b._alpha = 0;
  let flatHeavy = 0;
  for (let t = 0; t < trials; t++) {
    if (b.samplePositions(1, mulberry(t + 1))[0].board.a8) flatHeavy++;
  }
  assert.ok(flatHeavy > trials * 0.1 && flatHeavy < trials * 0.45,
    `α=0 should draw uniformly (~25%), got ${flatHeavy}/${trials}`);
});

// Deterministic rng so the sampling test can't flake.
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('exact belief: white first turn knows the exact position', () => {
  const state = FogChess.createInitialState(
    [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }],
    { fogOfWar: true, fog: true });
  const view = FogChess.getVisibleState(state, 'white');
  const tracker = new ExactBelief('white');
  tracker.beginTurn(view, 1);
  assert.equal(tracker.exact, true);
  assert.equal(tracker.positions.length, 1, 'nothing has moved: P = {initial}');
});

test('exact belief: re-acquisition from the heuristic belief (few hidden pieces)', () => {
  // Fog endgame: we are white with Kd1; black has a king we can't see and one
  // pawn we CAN see. The heuristic belief knows the black king's possible
  // squares; with one hidden piece the cross-product is tiny, so a lost exact
  // tracker must re-acquire a superset P that contains the true position.
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, fog: true });
  // Carve the board down to a K+P vs K endgame.
  const keep = new Set(['wK', 'bK', 'bPa']);
  const board = {};
  for (const sq of Object.keys(state.board)) {
    const p = state.board[sq];
    if (p && keep.has(p.id)) board[sq] = p;
  }
  const endState = { ...state, board, units: Object.values(board).filter(Boolean), turnNumber: 30 };
  const view = FogChess.getVisibleState(endState, 'white');

  const tracker = new ExactBelief('white');
  tracker.beginTurn(view, 30); // mid-game attach → gives up
  assert.equal(tracker.exact, false);

  // Heuristic belief stub: one hidden piece (the black king on e8) with a
  // small, honest possible-set; the black a-pawn is visible? (it is not — no
  // white piece sees a7 — so include it as a second hidden piece).
  const belief = {
    forcedEnemy: new Set(),
    pieces: new Map([
      ['bK', { id: 'bK', type: 'king', alive: true, truncated: false, possible: new Set(['e8', 'd8', 'f8', 'e7']) }],
      ['bPa', { id: 'bPa', type: 'pawn', alive: true, truncated: false, possible: new Set(['a7', 'a6', 'a5']) }],
    ]),
  };
  tracker.tryReacquire(view, belief, 30);
  assert.equal(tracker.exact, true, 're-acquisition should succeed with 2 hidden pieces');
  assert.equal(tracker.approx, true, 're-acquired P is marked approximate');
  const trueSig = placementSig(endState.board);
  assert.ok(tracker.positions.some(p => placementSig(toBoardObject(p)) === trueSig),
    'true position must be in the re-acquired P');
  // Per-piece possible-squares carry no history, so there is nothing to weigh the
  // placements by: a re-acquired set gets a flat distribution, and `approx` above
  // is what tells the panel not to read the percentages as a posterior.
  const W = tracker.weights;
  assert.equal(W.length, tracker.positions.length);
  const flat = 1 / W.length;
  assert.ok([...W].every(w => Math.abs(w - flat) < 1e-12), 're-acquired weights are uniform');
});

test('exact belief: re-acquisition refuses truncated possible-sets', () => {
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, fog: true });
  const view = FogChess.getVisibleState({ ...state, turnNumber: 9 }, 'white');
  const tracker = new ExactBelief('white');
  tracker.beginTurn(view, 9);
  assert.equal(tracker.exact, false);
  const belief = {
    forcedEnemy: new Set(),
    pieces: new Map([
      ['bK', { id: 'bK', type: 'king', alive: true, truncated: true, possible: new Set(['e8']) }],
    ]),
  };
  tracker.tryReacquire(view, belief, 9);
  assert.equal(tracker.exact, false, 'must not re-acquire from a truncated set');
});

test('exact belief: attaching mid-game gives up gracefully', () => {
  const state = FogChess.createInitialState(
    [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }],
    { fogOfWar: true, fog: true });
  const mid = { ...state, turnNumber: 7 };
  const view = FogChess.getVisibleState(mid, 'white');
  const tracker = new ExactBelief('white');
  tracker.beginTurn(view, 7);
  assert.equal(tracker.exact, false);
  assert.equal(tracker.samplePositions(4), null);
});

test('exact belief: sampled worlds carry an importance weight for the search reach', async () => {
  // The posterior reaches play through the CFR's ROOT REACH (infoset.js:
  // `cfrDescend(w.node, me, 1, w.prob)`), and until 2026-08-02 that reach was a
  // flat 1/N — so the belief was computed and then discarded before it could
  // affect a move. This pins the channel open.
  const { setBeliefReachWeightingForSeat } = await import('../src/exactBelief.js');
  let state = FogChess.createInitialState(
    [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }],
    { fogOfWar: true, fog: true });
  // Play a few plies so the belief has something to be non-uniform about.
  for (const [pid, from, to] of [['white', 'e2', 'e4'], ['black', 'e7', 'e5'],
    ['white', 'g1', 'f3'], ['black', 'b8', 'c6']]) {
    const obs = FogChess.getVisibleState(state, pid);
    const action = FogChess.getLegalActions({ ...state, activePlayers: [pid] }, pid)
      .find(a => a.from === from && a.to === to);
    FogChess.sampleWorlds(obs, pid, 2, () => 0.5);
    FogChess.onActionCommitted(obs, pid, action);
    state = FogChess.applyActions(state, [{ playerId: pid, action }]);
  }

  // Enabled explicitly: the DEFAULT is off (measured — see REACH_WEIGHTING_DEFAULT),
  // so this pins that the channel still works when switched on, which is what any
  // future re-measurement depends on.
  setBeliefReachWeightingForSeat('white', true);
  const view = FogChess.getVisibleState(state, 'white');
  const worlds = FogChess.sampleWorlds(view, 'white', 8, Math.random);
  setBeliefReachWeightingForSeat('white', null);
  assert.ok(worlds.length > 1, 'the fog should leave more than one world');
  for (const w of worlds) {
    assert.equal(typeof w.beliefWeight, 'number', 'every sampled world carries a weight');
    assert.ok(w.beliefWeight >= 0 && Number.isFinite(w.beliefWeight), `finite weight, got ${w.beliefWeight}`);
  }
  assert.ok(worlds.some(w => w.beliefWeight > 0), 'and they are not all zero');

  // Off — which is the shipped default — the field is absent entirely, and that
  // absence is what makes the search fall back to uniform reach.
  setBeliefReachWeightingForSeat('white', false);
  const flat = FogChess.sampleWorlds(
    FogChess.getVisibleState(state, 'white'), 'white', 8, Math.random);
  setBeliefReachWeightingForSeat('white', null);
  for (const w of flat) {
    assert.equal(w.beliefWeight, undefined, 'reach weighting off ⇒ no weight on the world');
  }
});

test('belief: a contradicted piece falls back to TYPE-LEGAL squares only', async () => {
  // The contradiction fallback used to be "anywhere hidden", which put enemy
  // pawns on their own first rank. tryReacquire trusts these sets, so the exact
  // belief then built worlds that could never occur, and Stockfish answers an
  // illegal position with zero MultiPV lines — the leaf evaluator silently
  // substituted its static fallback for every child of that node.
  const { possibleSquaresFor, impossiblePlacement } = await import('../src/belief.js');

  const wp = possibleSquaresFor('pawn', 'white');
  assert.ok(!wp.some(sq => sq[1] === '1'), 'no white pawn on rank 1 (it starts on 2 and only advances)');
  assert.ok(!wp.some(sq => sq[1] === '8'), 'no white pawn on rank 8 (it would have promoted)');
  const bp = possibleSquaresFor('pawn', 'black');
  assert.ok(!bp.some(sq => sq[1] === '8') && !bp.some(sq => sq[1] === '1'), 'mirrored for black');
  assert.equal(possibleSquaresFor('rook', 'white').length, 64, 'other types are unconstrained');

  // And the validator catches the board that started this hunt.
  const unit = (id, ownerId, type, position) => ({ id, ownerId, type, position, alive: true });
  const bad = {
    d1: unit('wP', 'white', 'pawn', 'd1'),
    e1: unit('wK', 'white', 'king', 'e1'),
    e8: unit('bK', 'black', 'king', 'e8'),
  };
  assert.match(impossiblePlacement(bad) ?? '', /white pawn on its own first rank/);
  delete bad.d1;
  assert.equal(impossiblePlacement(bad), null, 'a legal board reports nothing');
  delete bad.e8;
  assert.match(impossiblePlacement(bad) ?? '', /black has 0 kings/);
});
