// ---------------------------------------------------------------------------
// A minimal self-play loop: enough engine to play one game between two agents
// and report who won, and no more. Embedders have their own engines (replay
// logs, renderers, timing, an HTTP layer); this exists so the tuning scripts and
// the demo in this repo can play real games without one.
//
// The only subtlety is WHOSE STATE EACH PARTY SEES. An agent is handed its own
// observation (game.getVisibleState) and derives its legal moves from that, never
// from the truth — under fog those differ, and passing the true state would
// quietly turn every measurement into a perfect-information one. The true state
// is advanced separately, by applyActions, and stays with the loop.
//
// Belief upkeep needs no help here: ObscuroAgent calls game.onActionCommitted
// itself as it commits a move, so a seat played by an agent keeps its exact
// belief. A seat played by anything else (a human, a scripted line) must make
// that call itself or its belief will decay to "the enemy is still at home".
// ---------------------------------------------------------------------------

import { FogChess } from './FogChess.js';

/**
 * Play one game to completion.
 *
 * @param {object}   agents        { white: agent, black: agent } — anything with
 *                                 `async chooseAction(observation, legalActions)`.
 * @param {object}   [opts]
 * @param {object}   [opts.game]     GameDefinition (default FogChess)
 * @param {object}   [opts.config]   passed to createInitialState
 * @param {object[]} [opts.players]  passed to createInitialState
 * @param {number}   [opts.maxTurns] turn cap; the game ends `unfinished` at it
 * @param {(info) => void} [opts.onPly] called after every ply
 * @returns {Promise<{result: object|null, state: object, plies: number}>}
 *          `result` is the game's own getResult value, or null at the turn cap.
 */
export async function playMatch(agents, opts = {}) {
  const game = opts.game ?? FogChess;
  const players = opts.players ?? [{ id: 'white', name: 'White' }, { id: 'black', name: 'Black' }];
  const maxTurns = opts.maxTurns ?? 300;

  let state = game.createInitialState(players, { fogOfWar: true, ...opts.config });
  let plies = 0;

  while (game.getResult(state) == null && state.turnNumber <= maxTurns) {
    const mover = state.activePlayers[0];
    const observation = game.getVisibleState(state, mover);
    const legal = game.getLegalActions(observation, mover);
    if (!legal.length) break;

    const action = await agents[mover].chooseAction(observation, legal);
    if (!action) break;
    state = game.applyActions(state, [{ playerId: mover, action }]);
    plies++;
    opts.onPly?.({ ply: plies, mover, action, state });
  }

  return { result: game.getResult(state), state, plies };
}
