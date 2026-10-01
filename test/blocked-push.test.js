// A pawn push onto a dark square. The square ahead of a pawn is lit exactly
// when it is empty (board.js getVisibleSquares), so a dark one always holds a
// piece the pawn cannot see, and the push could only ever fail. It is not
// offered, and where an embedder's action set still holds one, the leaf
// evaluator prices it with the engine rather than the static evaluator.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { FogChess } from '../src/FogChess.js';
import { getAllFogMoves } from '../src/moves.js';
import { makeChessLeafEval } from '../src/ObscuroAgent.js';
import { toFEN } from '../src/fen.js';
import { available, multiPV, quit } from '../src/stockfish.js';

after(() => quit());

const key = a => FogChess.actionKey(a);

// 1. Nf3 d5 2. d4 Nf6: black's d5 pawn stands, unseen, in front of white's d4
// pawn. Both sides' exact beliefs are fed as the game goes, the way an agent
// feeds its own, so the position's belief population can be enumerated.
function blockedPosition() {
  let state = FogChess.createInitialState([{ id: 'white' }, { id: 'black' }], { fogOfWar: true });
  for (const [player, k] of [['white', 'g1f3'], ['black', 'd7d5'], ['white', 'd2d4'], ['black', 'g8f6']]) {
    const obs = FogChess.getVisibleState(state, player);
    FogChess.beliefPopulation(obs, player);
    const action = FogChess.getLegalActions(obs, player).find(a => key(a) === k);
    FogChess.onActionCommitted(obs, player, action);
    state = FogChess.applyActions(state, [{ playerId: player, action }]);
  }
  return state;
}

test('a push onto a dark square is not offered; the other pawn moves are', () => {
  const state = blockedPosition();
  const obs = FogChess.getVisibleState(state, 'white');
  assert.ok(!obs.visibleSquares.includes('d5'));
  assert.equal(obs.board.d5, undefined); // the blocker is hidden…
  assert.ok(getAllFogMoves(obs.board, 'white', obs.gameSpecific).some(a => key(a) === 'd4d5')); // …so it looks free
  const legal = FogChess.getLegalActions(obs, 'white').map(key);
  assert.ok(!legal.includes('d4d5'));
  for (const k of ['c2c4', 'e2e4', 'e2e3', 'h2h4']) assert.ok(legal.includes(k), k);
});

test('the observation offers exactly the moves every consistent world does', () => {
  const state = blockedPosition();
  const obs = FogChess.getVisibleState(state, 'white');
  const fromObservation = FogChess.getLegalActions(obs, 'white').map(key).sort();
  const pop = FogChess.beliefPopulation(obs, 'white');
  assert.ok(pop.exact && pop.total > 1);
  const worlds = FogChess.enumerateWorlds(obs, 'white', Array.from({ length: pop.total }, (_, i) => i));
  for (const world of worlds) {
    assert.deepEqual(getAllFogMoves(world.board, 'white', world.gameSpecific).map(key).sort(), fromObservation);
  }
});

test('a new position does not inherit the view it was reached from', () => {
  const state = blockedPosition();
  const obs = FogChess.getVisibleState(state, 'white');
  const next = FogChess.applyActions(obs, [{ playerId: 'white', action: FogChess.getLegalActions(obs, 'white')[0] }]);
  assert.equal(next.visibleSquares, undefined);
  assert.equal(next.viewerId, undefined);
});

test('a push the world blocks is priced by the engine, as the pass it is', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  const state = blockedPosition();
  const obs = FogChess.getVisibleState(state, 'white');
  // An embedder's action set may still hold the blocked push.
  const actions = getAllFogMoves(obs.board, 'white', obs.gameSpecific);
  const i = actions.findIndex(a => key(a) === 'd4d5');
  const children = actions.map(a => FogChess.applyActions(state, [{ playerId: 'white', action: a }]));
  assert.deepEqual(children[i].board, state.board); // the pawn stays put

  const depth = 8;
  const scores = await makeChessLeafEval(depth, actions.length)(state, 'white', actions, children);
  const child = children[i];
  const [line] = await multiPV(toFEN(child.board, child.gameSpecific, 'b', child.turnNumber), { multipv: 1, depth });
  assert.equal(scores[i], -(line.wdl[0] - line.wdl[2]) / 1000, "the child's own win/draw/loss, from white's side");
  assert.ok(scores[i] < Math.max(...scores), 'wasting the turn scored as the best move');
});
