// ---------------------------------------------------------------------------
// Node-level tree carryover (paper §3.1 / Fig. 9): across consecutive moves of
// one agent instance, subtrees of the previous search that are consistent with
// the new observation must be grafted in as root worlds (with their infosets,
// regrets, and carried alternate values), not recomputed from scratch.
// ---------------------------------------------------------------------------

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { FogChess } from '../src/FogChess.js';
import { ChessObscuroAgent } from '../src/ObscuroAgent.js';
import { quit as stockfishQuit } from '../src/stockfish.js';

after(() => stockfishQuit());

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('fog search grafts carried subtrees across moves', async () => {
  const rng = mulberry32(777);
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const config = { fogOfWar: true, fog: true, difficulty: 60, difficultyMode: 'power', maxTurns: 300 };
  let state = FogChess.createInitialState(players, config);
  const agents = { white: new ChessObscuroAgent({ rng }), black: new ChessObscuroAgent({ rng }) };

  let totalCarried = 0;
  const perPly = [];
  for (let ply = 0; ply < 8; ply++) {
    if (FogChess.getResult(state)) break;
    const me = state.activePlayers[0];
    const view = FogChess.getVisibleState(state, me);
    const legal = FogChess.getLegalActions(state, me);
    const action = await agents[me].chooseAction(view, legal);
    assert.ok(action, `ply ${ply}: agent must move`);
    const carried = agents[me].lastAnalysis?.carried ?? 0;
    perPly.push(`${me}:${carried}`);
    totalCarried += carried;
    state = FogChess.applyActions(state, [{ playerId: me, action }]);
  }
  // The first move of each side cannot carry; later moves should graft at
  // least sometimes (the opponent's actual reply was usually expanded in the
  // previous tree and early-game observations are near-deterministic).
  assert.ok(totalCarried > 0, `expected carried subtrees over 8 plies, got: ${perPly.join(' ')}`);
});
