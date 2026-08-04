// ---------------------------------------------------------------------------
// A runnable demo: two Obscuro agents play fog-of-war chess, each seeing only
// the squares its own pieces can reach. No install step of any kind:
//
//   node examples/demo.js
//   node examples/demo.js --difficulty 60 --max-turns 40
//
// Every ply prints the mover's OWN view of the board (the fog is real — the
// pieces it cannot see are not in the state it was handed), the move it chose,
// and how many belief worlds it was reasoning over. The true board is printed
// once at the end.
// ---------------------------------------------------------------------------

import { FogChess } from '../src/FogChess.js';
import { ChessObscuroAgent } from '../src/ObscuroAgent.js';
import { playMatch } from '../src/playMatch.js';
import { renderBoard } from '../src/board.js';
import { quit as stockfishQuit } from '../src/stockfish.js';

const arg = (flag, fallback) => {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] != null ? Number(process.argv[i + 1]) : fallback;
};
const difficulty = arg('--difficulty', 40);
const maxTurns = arg('--max-turns', 30);

const agents = {
  white: new ChessObscuroAgent(),
  black: new ChessObscuroAgent(),
};

console.log(`Fog-of-war chess — neither side sees the other's pieces (difficulty ${difficulty})\n`);

const { result, state, plies } = await playMatch(agents, {
  maxTurns,
  config: { fogOfWar: true, difficulty },
  onPly({ ply, mover, action, state }) {
    const view = FogChess.getVisibleState(state, mover);
    const a = agents[mover].lastAnalysis ?? {};
    console.log(`ply ${ply} — ${mover}: ${action.from}→${action.to}` +
      `  [${a.mode ?? 'n/a'}, ${a.worlds ?? '?'} world(s)]`);
    console.log(renderBoard(view.board));
    console.log('');
  },
});

console.log('true board:');
console.log(renderBoard(state.board));
console.log(result
  ? `\nresult after ${plies} plies: ${result.outcome}` +
    `${result.winnerId ? ` — ${result.winnerId} wins` : ''} (${result.reason})`
  : `\nno result after ${plies} plies (turn cap ${maxTurns})`);

await stockfishQuit();
