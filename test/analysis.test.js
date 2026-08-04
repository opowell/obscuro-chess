import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FogChess } from '../src/FogChess.js';
import { ChessAgent } from '../src/ChessAgent.js';
import { analyzeObscuro, obscuroStrategy } from '../src/ObscuroAgent.js';
import { getBelief } from '../src/belief.js';
import { ExactBelief, fromBoardObject } from '../src/exactBelief.js';
import { getAllLegalMoves } from '../src/moves.js';
import { quit as stockfishQuit } from '../src/stockfish.js';

const unit = (id, ownerId, type, position) => ({ id, ownerId, type, position, alive: true });
const noCastle = { white: { kingSide: false, queenSide: false }, black: { kingSide: false, queenSide: false } };

// A free queen sitting on a8 for white's rook to take — the same fixture
// obscuro.test.js uses, so both engines are checked against an unambiguous
// "obviously correct" answer.
function freeQueenState() {
  const board = {
    e1: unit('wK', 'white', 'king', 'e1'),
    a1: unit('wR', 'white', 'rook', 'a1'),
    a8: unit('bQ', 'black', 'queen', 'a8'),
    h8: unit('bK', 'black', 'king', 'h8'),
  };
  const gameSpecific = { enPassantTarget: null, castlingRights: noCastle, halfMoveClock: 0, inCheck: false, fogOfWar: false, difficulty: 'medium' };
  const state = { players: [{ id: 'white' }, { id: 'black' }], activePlayers: ['white'], board, units: Object.values(board), turnNumber: 1, gameSpecific };
  return { state, legal: getAllLegalMoves(board, 'white', gameSpecific) };
}

test('ChessAgent.analyze: perfect info ranks the free-queen capture first', async () => {
  const { state, legal } = freeQueenState();
  const r = await ChessAgent.analyze(state, legal);
  assert.equal(r.engine, 'chess-ai');
  assert.ok(r.candidates.length > 0);
  assert.equal(r.candidates[0].move.to, 'a8', 'best-ranked move should capture the queen');
  assert.ok(r.candidates[0].move.isCapture);
});

test('analyzeObscuro: perfect info ranks the free-queen capture first', async () => {
  // Perfect info is now just a belief population of size 1 (see
  // FogChess.beliefPopulation) walked by the one progressive analysis path, so
  // the solve mode is the CFR tree's perfect-information collapse, 'minimax'.
  // Its 100%/0% support carries no ranking information at a population of 1, so
  // ranking falls to the real, calibrated Stockfish cp — the same numbers the
  // old dedicated perfect-information branch showed.
  const { state, legal } = freeQueenState();
  const r = await analyzeObscuro(state, legal, { rng: () => 0, maxSfDepth: 8 });
  assert.equal(r.engine, 'obscuro');
  assert.equal(r.mode, 'minimax');
  assert.ok(r.candidates.length > 0);
  assert.equal(r.candidates[0].move.to, 'a8', 'best-ranked move should capture the queen');
  // Threshold 300, not 500: modern Stockfish normalises cp toward WIN
  // PROBABILITY rather than material, so the numbers are not centipawns in the
  // "queen = 900" sense. Measured on this exact position under the vendored
  // SF18: Rxa8 scores ~470, and up-a-whole-queen scores ~890 (SF11 rated the
  // same capture above 500, which is what this assertion used to encode). What
  // the test is actually for is that a free queen reads as a large swing and not
  // a rounding error, so assert that and stay off the engine's calibration.
  assert.ok(r.candidates[0].cp > 300, 'capturing a free queen should score as a large material swing');
});

test('analyzeObscuro: perfect info climbs the depth ladder, reporting top moves at each rung', async () => {
  // The point of the ladder: a complete, self-consistent answer at every depth,
  // not one opaque wait for a deep one. Every rung must report its own depth and
  // a full ranked candidate list, and the depths must march upward.
  const { state, legal } = freeQueenState();
  const frames = [];
  const r = await analyzeObscuro(state, legal, {
    rng: () => 0, maxSfDepth: 6, onProgress: (info) => frames.push(info),
  });
  const depths = frames.map(f => f.depth);
  assert.deepEqual(depths, [1, 2, 3, 4, 5, 6], 'one frame per rung, depth 1 upward');
  for (const f of frames) {
    assert.equal(f.maxDepth, 6);
    assert.equal(f.total, 1, 'nothing hidden → a belief population of exactly one world');
    assert.ok(f.candidates.length > 0, `rung ${f.depth} reports its ranked moves`);
    assert.equal(f.candidates[0].move.to, 'a8', `rung ${f.depth} already sees the free queen`);
  }
  assert.equal(frames.at(-1).exhaustive, true, 'the top rung over the whole population is settled');
  assert.equal(r.depth, 6, 'the final result reports the deepest rung actually searched');
});

// ---------------------------------------------------------------------------
// `opts.color` override: lets the analysis API always answer "what's good for
// MY side" even when it isn't literally that side's turn in the true state —
// the fog case api-server.js's handleAnalyze relies on (see its comment) so a
// viewer can preview their own position instead of being flatly blocked
// whenever the opponent is mid-turn.
// ---------------------------------------------------------------------------

test('ChessAgent.analyze: opts.color analyzes white even when activePlayers says black', async () => {
  const { state, legal } = freeQueenState();
  const notWhitesTurn = { ...state, activePlayers: ['black'] };
  const r = await ChessAgent.analyze(notWhitesTurn, legal, { color: 'white' });
  assert.ok(r.candidates.length > 0);
  assert.equal(r.candidates[0].move.to, 'a8', 'should still find the free-queen capture for white');
});

test('analyzeObscuro: opts.color analyzes white even when activePlayers says black', async () => {
  const { state, legal } = freeQueenState();
  const notWhitesTurn = { ...state, activePlayers: ['black'] };
  const r = await analyzeObscuro(notWhitesTurn, legal, { rng: () => 0, color: 'white', maxSfDepth: 6 });
  assert.ok(r.candidates.length > 0);
  assert.equal(r.candidates[0].move.to, 'a8', 'should still find the free-queen capture for white');
});

test('analyzeObscuro: fog produces a probability distribution summing to ~1', async () => {
  const players = [{ id: 'white', name: 'W' }, { id: 'black', name: 'B' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(state, 'white');

  const r = await analyzeObscuro(view, legal, { particles: 4, maxSfDepth: 4 });
  assert.equal(r.mode, 'cfr');
  const sum = r.candidates.reduce((a, c) => a + c.prob, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6, `probabilities should sum to 1, got ${sum}`);
  for (const c of r.candidates) assert.ok(c.prob >= -1e-9 && c.prob <= 1 + 1e-9, `probability out of range: ${c.prob}`);
});

// ---------------------------------------------------------------------------
// Regression guard: analyze() is read-only. It must never advance the shared
// per-color Belief beyond the one `beginTurn` a real decision for that same
// turn would also do (turnKey idempotency, belief.js:215), and must never
// call `commitOurMove` (belief.js:234, the "I actually played this" step —
// its only effect is setting `ownSnapshot`, which stays null otherwise).
// ---------------------------------------------------------------------------

test('ChessAgent.analyze() does not advance or commit the belief', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(state, 'white');

  await ChessAgent.analyze(view, legal);
  const belief = getBelief(view, 'white');
  const pliesAfterOne = belief.oppPlies;
  assert.equal(belief.ownSnapshot, null, 'analyze() must never commit a move to the belief');

  // Re-entering analyze() for the SAME turn must be a no-op on the belief
  // (idempotent via turnKey), exactly like a real agent re-sampling mid-decision.
  await ChessAgent.analyze(view, legal);
  await ChessAgent.analyze(view, legal);
  assert.equal(belief.oppPlies, pliesAfterOne, 'repeated analyze() calls for the same turn must not re-advance the belief');
  assert.equal(belief.ownSnapshot, null, 'analyze() must still never commit a move to the belief');
});

test('analyzeObscuro() does not commit a move to the belief', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(state, 'white');

  await analyzeObscuro(view, legal, { particles: 4, maxSfDepth: 2 });
  const belief = getBelief(view, 'white');
  assert.equal(belief.ownSnapshot, null, 'analyzeObscuro() must never commit a move to the belief');
});

// ---------------------------------------------------------------------------
// Whole-population enumeration (the batched, eventually-exhaustive belief walk).
// beliefPopulation reports the finite exact set's size; enumerateWorlds walks it
// once without replacement. At the game's first turn the belief is exact and the
// initial position is common knowledge, so |P| = 1 — the smallest possible walk.
// ---------------------------------------------------------------------------

test('beliefPopulation + enumerateWorlds: cover the whole exact set exactly once', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(state, 'white');

  const pop = FogChess.beliefPopulation(view, 'white');
  assert.equal(pop.exact, true, 'exact belief is active at the first turn');
  assert.ok(pop.total >= 1, 'a finite population size is reported');

  const all = FogChess.enumerateWorlds(view, 'white', [...Array(pop.total).keys()]);
  assert.equal(all.length, pop.total, 'enumerating every index yields the whole population');
  const keyOf = (w) => Object.entries(w.board)
    .filter(([, p]) => p).map(([sq, p]) => `${sq}:${p.ownerId[0]}${p.type[0]}`).sort().join(',');
  assert.equal(new Set(all.map(keyOf)).size, pop.total, 'enumerated worlds are distinct (no replacement)');
  for (const w of all) assert.equal(w.activePlayers[0], 'white', 'worlds carry the analyzed side to move');

  assert.equal(FogChess.enumerateWorlds(view, 'white', [pop.total + 5]).length, 0, 'out-of-range indices are skipped');
});

test('analyzeObscuroProgressive: a finite exact population exhausts and stops on its own', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(state, 'white');

  const frames = [];
  // isCancelled never fires, so ONLY exhaustion can end the walk — a regression
  // here (e.g. resample-with-replacement) would loop until maxTotalMs instead.
  const r = await analyzeObscuro(view, legal, {
    color: 'white', rng: () => 0.5, isCancelled: () => false,
    maxRounds: 4, expandPerRound: 2, cfrPerRound: 1, batchSize: 8, maxSfDepth: 3,
    onProgress: (info) => frames.push(info),
  });
  assert.ok(r, 'returns a final result');
  assert.equal(r.exhaustive, true, 'the whole population was covered');
  assert.equal(r.total, r.evaluated, 'evaluated exactly the population total');
  assert.ok(frames.length >= 1 && frames.some(f => f.exhaustive), 'emits an exhaustive progress frame');
  const sum = r.candidates.reduce((a, c) => a + c.prob, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6, `probabilities still sum to 1, got ${sum}`);
});

// Prompt cancellation (the memory-leak guard): when the analysis position
// changes, an in-flight solve must bail mid-flight, not run out its rounds — or
// stale walks pile up. isCancelled cuts the CFR round loop short, so a solve
// asked for 50 rounds stops within one round of the flag flipping.
test('obscuroStrategy: isCancelled cuts the CFR round loop short', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const st = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view = FogChess.getVisibleState(st, 'white');
  const legal = FogChess.getLegalActions(st, 'white');

  let rounds = 0, cancel = false;
  const r = await obscuroStrategy(view, legal, {
    color: 'white', rng: () => 0.5, maxRounds: 50, expandPerRound: 2, cfrPerRound: 1,
    isCancelled: () => cancel,
    onProgress: (info) => { if (info.kind === 'round') { rounds++; if (rounds >= 3) cancel = true; } },
  });
  assert.ok(r.action, 'still returns a usable move from the rounds it did run');
  assert.ok(rounds < 50, `round loop stopped early once cancelled: ran ${rounds}/50`);
  stockfishQuit();
});

// ---------------------------------------------------------------------------
// The per-world view: showing a human WHICH boards the fog could be hiding, not
// just the averaged move ranking derived from them. See ExactBelief
// .rankByLikelihood, FogChess.hiddenPiecesOf, and analyzeObscuro's
// `beliefWorlds` payload.
// ---------------------------------------------------------------------------

test('rankByLikelihood: the heaviest world tops the ranking, every index labelled', () => {
  // Three positions differing only in where two hidden black pieces sit, with an
  // explicit posterior over them. (Where those weights COME FROM — colliding
  // histories summing, π, the observation filter — is exact-belief.test.js's
  // business; this checks only that the ranking is the posterior and that the
  // returned shape still labels the whole population.)
  const mk = (extra) => fromBoardObject(
    { e1: unit('wK', 'white', 'king', 'e1'), e8: unit('bK', 'black', 'king', 'e8'), ...extra },
    noCastle, null);
  const b = new ExactBelief('white');
  b.exact = true;
  b.positions = [
    mk({ d8: unit('bQ', 'black', 'queen', 'd8'), b8: unit('bN', 'black', 'knight', 'b8') }),
    mk({ d8: unit('bQ', 'black', 'queen', 'd8'), g8: unit('bN', 'black', 'knight', 'g8') }),
    mk({ h4: unit('bQ', 'black', 'queen', 'h4'), b8: unit('bN', 'black', 'knight', 'b8') }),
  ];
  b.weights = Float64Array.of(0.6, 0.3, 0.1);

  const r = b.rankByLikelihood(3);
  assert.equal(r.total, 3);
  assert.equal(r.top[0].index, 0, 'the heaviest world ranks first');
  assert.equal(r.top[0].prob, 0.6, 'and its prob IS its posterior weight');
  assert.deepEqual(r.top.map(t => t.index), [0, 1, 2], 'the ranking is descending by weight');
  const sum = [...r.probs].reduce((a, p) => a + p, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `probabilities are a distribution, got ${sum}`);
  // Every index is labelled, not just the ones on the returned page.
  assert.equal(r.probs.length, 3);

  // A hand-built tracker with no weights at all must still answer, flatly, rather
  // than throw — several call sites construct one directly.
  b.weights = null;
  const flat = b.rankByLikelihood(3);
  assert.ok(flat.probs.every(p => Math.abs(p - 1 / 3) < 1e-12), 'no posterior → uniform');
});

test('hiddenPiecesOf: only what the viewer cannot already see, in grid coords', () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view  = FogChess.getVisibleState(state, 'white');
  // The one world of the first turn's population IS the true initial position;
  // white nonetheless cannot see black's back two ranks. (beliefPopulation first:
  // it is what establishes P for this turn — see enumerateWorlds.)
  FogChess.beliefPopulation(view, 'white');
  const [world] = FogChess.enumerateWorlds(view, 'white', [0]);
  const hidden = FogChess.hiddenPiecesOf(world, view, 'white');

  assert.equal(hidden.length, 16, "black's whole army is hidden at the start");
  assert.ok(hidden.every(h => h.type && h.type === h.type.toLowerCase()),
    'marker types are the one-letter codes the fog-marker renderer speaks');
  const a8 = hidden.find(h => h.sq === 'a8');
  assert.deepEqual({ x: a8.x, y: a8.y }, { x: 0, y: 0 }, 'a8 maps to the top-left grid cell');
  const e7 = hidden.find(h => h.sq === 'e7');
  assert.deepEqual({ x: e7.x, y: e7.y, type: e7.type }, { x: 4, y: 1, type: 'p' });
  // Nothing white, and nothing already on screen.
  assert.ok(!hidden.some(h => h.sq.endsWith('1') || h.sq.endsWith('2')), 'own pieces are never markers');
});

test('analyzeObscuro under fog: emits belief worlds with per-move cp per world', async () => {
  const players = [{ id: 'white' }, { id: 'black' }];
  const state = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const view  = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(state, 'white');

  // A stand-in engine: scores every move in every world, so the per-world
  // channel is exercised without waiting on a real search. It also asserts the
  // weighting contract — enumerated worlds must arrive carrying their posterior
  // probability, heaviest first — and reports a mass-weighted sum, the shape
  // cpSumsOverWorlds now returns.
  const frames = [];
  const seenWeights = [];
  const r = await analyzeObscuro(view, legal, {
    color: 'white', rng: () => 0.5, isCancelled: () => false,
    maxRounds: 4, expandPerRound: 2, cfrPerRound: 1, batchSize: 8, maxSfDepth: 1,
    cpEval: (worlds, actions, depth, onWorld) => {
      worlds.forEach((w, i) => onWorld?.(i, w, actions.map((_, j) => 10 * j)));
      let wsum = 0;
      for (const w of worlds) { seenWeights.push(w.beliefWeight); wsum += w.beliefWeight; }
      return { sums: actions.map((_, j) => 10 * j * wsum), wsum, n: worlds.length };
    },
    onProgress: (info) => frames.push(info),
  });
  assert.ok(seenWeights.length > 0 && seenWeights.every(w => typeof w === 'number' && w > 0),
    'every enumerated world carries a positive posterior weight');
  assert.deepEqual(seenWeights, [...seenWeights].sort((a, b) => b - a),
    'and the walk visits them heaviest first');

  const bw = r.beliefWorlds;
  assert.ok(bw, 'the final result carries the belief population');
  assert.equal(bw.exact, true);
  assert.deepEqual(bw.moves, legal.map(FogChess.actionKey), 'cp columns are keyed to the legal moves');
  assert.ok(bw.worlds.length >= 1, 'at least one world to show');
  for (const w of bw.worlds) {
    assert.ok(Array.isArray(w.hidden) && w.hidden.length > 0, 'each world says what the fog is hiding');
    assert.equal(w.cp.length, legal.length, 'one cp per legal move, aligned with `moves`');
  }
  // Candidates carry the same key, so a panel can line a row up with its column.
  assert.ok(r.candidates.every(c => bw.moves.includes(c.key)), 'every candidate row is addressable');

  // The likely boards are handed over BEFORE the engine has priced anything,
  // so the overlay can be on screen while the search is still running.
  const opener = frames[0];
  assert.equal(opener.kind, 'belief', 'the first frame is the engine-free board list');
  assert.ok(opener.beliefWorlds.worlds.length >= 1);
  assert.equal(opener.beliefWorlds.worlds[0].cp, null, 'nothing is scored yet at that point');
  stockfishQuit();
});

// ---------------------------------------------------------------------------
// The belief-upkeep contract the analysis path depends on. An imperfect-information
// belief is built ACROSS turns: each turn it advances the opponent one ply and
// filters against what we now see. That only works if our OWN last move was applied
// to it first — otherwise the advance starts from a position where our pieces are
// still on their old squares, nothing survives the filter, and exactness is
// abandoned (after which the fallback confidently reports the enemy army still at
// home). `onActionCommitted` is what applies our move; api-server.js's
// Session.syncSeatBelief is what calls it for a human seat, which nothing did
// before — see the second test for what that omission costs.
// ---------------------------------------------------------------------------

function fogGame() {
  const players = [{ id: 'white' }, { id: 'black' }];
  let st = FogChess.createInitialState(players, { fogOfWar: true, difficulty: 30 });
  const boardKey = b => Object.entries(b).filter(([, p]) => p)
    .map(([sq, p]) => `${sq}${p.ownerId[0]}${p.type[0]}`).sort().join(',');
  return {
    get state() { return st; },
    // Play `from`-`to` for `color`, optionally telling that seat's belief about it
    // first (what a maintained seat does, via FogChess.onActionCommitted).
    play(color, from, to, { commit = false } = {}) {
      const action = { type: 'move', from, to, unitId: st.board[from].id };
      if (commit) FogChess.onActionCommitted(FogChess.getVisibleState(st, color), color, action);
      st = FogChess.applyActions(st, [{ playerId: color, action }]);
    },
    // What the analysis sees for `color`: is the belief exact, how big, and does it
    // still contain the position that is actually on the board?
    belief(color) {
      const view = FogChess.getVisibleState(st, color);
      const pop = FogChess.beliefPopulation(view, color);
      if (!pop.exact) return { exact: false, total: null, holdsTruth: null };
      const all = FogChess.enumerateWorlds(view, color, [...Array(Math.min(pop.total, 20000)).keys()]);
      return { exact: true, total: pop.total, holdsTruth: all.some(w => boardKey(w.board) === boardKey(st.board)) };
    },
  };
}

test('belief upkeep: a seat told its own moves stays exact and keeps the true position', () => {
  const g = fogGame();
  assert.deepEqual(g.belief('white'), { exact: true, total: 1, holdsTruth: true },
    'the opening position is common knowledge, so P is the single true board');

  g.play('white', 'e2', 'e4', { commit: true });
  g.play('black', 'f7', 'f6');
  let b = g.belief('white');
  assert.equal(b.exact, true, 'still exact after one full round');
  assert.ok(b.total > 1, 'and now uncertain, since black\'s reply was hidden');
  assert.equal(b.holdsTruth, true, 'the real board is among the possibilities');

  g.play('white', 'g1', 'f3', { commit: true });
  g.play('black', 'd7', 'd6');
  b = g.belief('white');
  assert.equal(b.exact, true, 'still exact after two');
  assert.ok(b.total > 10, `uncertainty compounds with each hidden reply, got ${b.total}`);
  assert.equal(b.holdsTruth, true, 'the real board is STILL among the possibilities');
});

test('belief upkeep: a seat NOT told its own moves loses exactness from the second turn', () => {
  // The regression this pins: without the commit, chess falls back to a heuristic
  // guess, and the analysis silently averages over boards that cannot occur.
  const g = fogGame();
  assert.equal(g.belief('white').exact, true, 'turn 1 is fine either way — nothing of ours has moved yet');

  g.play('white', 'e2', 'e4'); // no commit — the omission
  g.play('black', 'f7', 'f6');
  assert.equal(g.belief('white').exact, false,
    'our own move never reached the belief, so nothing is consistent and exactness is dropped');
});

test('getVisibleState: the opponent\'s hidden move does not ride along in lastActions', () => {
  const g = fogGame();
  g.play('white', 'e2', 'e4', { commit: true });
  g.play('black', 'b8', 'c6'); // to a square white cannot see

  const view = FogChess.getVisibleState(g.state, 'white');
  assert.equal(view.board.c6, undefined, 'the knight itself is correctly hidden');
  assert.ok(!(view.lastActions ?? []).some(pa => pa.playerId === 'black'),
    'and so is the move that put it there — otherwise the from/to squares hand it straight back');
  // Our own move stays: it is not a secret from us, and the UI animates it.
  const own = FogChess.getVisibleState(g.state, 'black').lastActions ?? [];
  assert.ok(own.some(pa => pa.playerId === 'black' && pa.action.to === 'c6'),
    'a seat still sees its own last move');
});
