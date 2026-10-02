// The leaf evaluator's values beyond the children: the position itself, which
// the generic search uses as a fresh world's ṽ(h) (FRESH_ALT_VALUE = 'self').

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { FogChess } from '../src/FogChess.js';
import { getAllFogMoves } from '../src/moves.js';
import { makeChessLeafEval, fogValue } from '../src/ObscuroAgent.js';
import { toFEN } from '../src/fen.js';
import { available, multiPV, quit } from '../src/stockfish.js';

after(() => quit());

test('the evaluator reports the position\'s own fog value, from its best line', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const state = FogChess.createInitialState(players, {});
  const actions = getAllFogMoves(state.board, 'white', state.gameSpecific);
  const children = actions.map(a => FogChess.applyActions(state, [{ playerId: 'white', action: a }]));
  const depth = 6;
  const scores = await makeChessLeafEval(depth, actions.length)(state, 'white', actions, children);
  // The same MultiPV call the evaluator made; its first line is the engine's
  // value of the parent for the mover.
  const [line] = await multiPV(toFEN(state.board, state.gameSpecific, 'w', state.turnNumber ?? 1), { multipv: actions.length, depth });
  const e = (line.wdl[0] + line.wdl[1] / 2) / 1000;
  assert.ok(Number.isFinite(scores.self), 'a value for the position itself');
  assert.ok(Math.abs(scores.self - fogValue(e, state.board, 'white')) < 1e-12);
});
