import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { FogChess } from '../src/FogChess.js';
import { ObscuroAgent } from '../src/ObscuroAgent.js';
import { ChessAgent } from '../src/ChessAgent.js';
import { toFEN, uciToAction } from '../src/fen.js';
import { available, bestMove, multiPV, quit } from '../src/stockfish.js';

after(() => quit());

// ---------------------------------------------------------------------------
// FEN / UCI helpers (no engine needed)
// ---------------------------------------------------------------------------

test('toFEN renders the starting position', () => {
  const s = FogChess.createInitialState([{ id: 'white' }, { id: 'black' }], { fogOfWar: false });
  const fen = toFEN(s.board, s.gameSpecific, 'w', 1);
  assert.equal(fen, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
});

test('uciToAction maps moves, promotions and castling', () => {
  const acts = [
    { from: 'e2', to: 'e4', isDoublePush: true },
    { from: 'e7', to: 'e8', payload: { promote: 'queen' } },
    { from: 'e7', to: 'e8', payload: { promote: 'knight' } },
    { type: 'castle', side: 'kingside', from: 'e1', to: 'g1' },
  ];
  assert.equal(uciToAction('e2e4', acts).to, 'e4');
  assert.equal(uciToAction('e7e8q', acts).payload.promote, 'queen');
  assert.equal(uciToAction('e7e8n', acts).payload.promote, 'knight');
  assert.equal(uciToAction('e1g1', acts).type, 'castle');
  assert.equal(uciToAction('a1a2', acts), null);
});

// ---------------------------------------------------------------------------
// Engine-backed tests (skip cleanly if the vendored engine can't load)
// ---------------------------------------------------------------------------

// Reach the position after 1. d4 Nf6 2. d5 (Black to move) — the exact spot
// where the weak search blundered ...Nxd5?? (the pawn is defended by Qd1).
//
// difficulty 100 is load-bearing, not incidental. With perfect information the
// agent picks by sampling MultiPV lines ∝ win-probability^beta, and beta is what
// the dial slides (CHESS_DIAL.proportionalPick: ~1 at mid-dial, 12 at the top).
// A mid-dial agent is therefore SUPPOSED to play a weaker move sometimes, so
// asserting a specific move there is asserting on a coin flip — it failed in CI
// on exactly one of three Node versions before this comment existed. At the top
// of the dial the pick is effectively the engine's best move, which is what this
// test is actually about: that the engine is wired up and being consulted.
function blunderPosition() {
  let s = FogChess.createInitialState(
    [{ id: 'white', name: 'W', agent: { id: 'x' } }, { id: 'black', name: 'B', agent: { id: 'x' } }],
    { fogOfWar: false, difficulty: 100 });
  for (const [pid, from, to] of [['white', 'd2', 'd4'], ['black', 'g8', 'f6'], ['white', 'd4', 'd5']]) {
    const a = FogChess.getLegalActions(s, pid).find(m => m.from === from && m.to === to);
    s = FogChess.applyActions(s, [{ playerId: pid, action: a }]);
  }
  return s; // Black to move
}

test('stockfish: engine is available and avoids ...Nxd5', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  const s = blunderPosition();
  const fen = toFEN(s.board, s.gameSpecific, 'b', s.turnNumber);
  const uci = await bestMove(fen, { movetime: 300 });
  assert.ok(uci, 'engine returned a move');
  assert.notEqual(uci, 'f6d5', 'Stockfish must not capture the defended d5 pawn');
});

test('multiPV returns several scored candidate moves (batched leaf heuristic)', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  const lines = await multiPV('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', { multipv: 5, depth: 4 });
  assert.ok(Array.isArray(lines) && lines.length >= 2, 'expected several scored moves');
  for (const l of lines) {
    assert.equal(typeof l.move, 'string');
    assert.equal(typeof l.cp, 'number');
  }
});

test('multiPV: the cache returns the same lines however often it is read', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  // The LRU touch is sampled (one read in TOUCH_EVERY writes), so reads cross
  // that boundary several times here. Sampling may change which entry is
  // eventually EVICTED; it must never change, drop or corrupt what a hit
  // returns. Everything downstream — the leaf evaluator, every measurement
  // harness — assumes a cached position answers identically to a fresh one.
  const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const first = await multiPV(fen, { multipv: 5, depth: 4 });
  assert.ok(Array.isArray(first) && first.length >= 2);
  for (let i = 0; i < 100; i++) {
    assert.deepEqual(await multiPV(fen, { multipv: 5, depth: 4 }), first, `read ${i} differed`);
  }
});

test('perfect-info agents use Stockfish and do not blunder the knight', async (t) => {
  if (!(await available())) { t.skip('vendored stockfish failed to load'); return; }
  const isBlunder = a => a.from === 'f6' && a.to === 'd5';

  for (const agent of [ObscuroAgent, ChessAgent]) {
    const s = blunderPosition();
    const legal = FogChess.getLegalActions(s, 'black');
    const action = await agent.chooseAction(s, legal);
    assert.ok(action, `${agent.id} returned an action`);
    assert.ok(!isBlunder(action), `${agent.id} must not play ...Nxd5`);
  }
});
