// ---------------------------------------------------------------------------
// ChessObscuroAgent — the chess specialisation of the generic ObscuroAgent.
//
// The search itself is entirely generic and lives in its own repository, vendored
// here at vendor/obscuro (github.com/opowell/obscuro-ai — the paper's real
// extensive-form machinery: a growing game tree, PCFR+ on the last iterate,
// one-sided GT-CFR expansion, purification). This file adds only the
// two things that are genuinely chess-specific, exactly matching the paper's
// division of labour ("game-independent search + a perfect-information eval"):
//
//   1. The LEAF EVALUATOR — a batched Stockfish node heuristic (`_leafEval`).
//      For each expanded node it asks Stockfish, in one MultiPV call, to score
//      every child of that node, and handles the two fog-of-war terminals a
//      standard engine cannot see: capturing the enemy king (a win, surfaced by
//      the game's getResult) and leaving one's own king capturable (a loss).
//
//   2. The PERFECT-INFORMATION SHORTCUT — with nothing hidden there is no belief
//      to reason over, so the strongest move is simply Stockfish's best move.
//      Routing it through the fog subgame would only flatten winning lines to the
//      same clamped leaf value (shuffling instead of converting), so we play the
//      engine directly, its strength scaled by the difficulty dial. This is an
//      information-model distinction, not a per-difficulty branch.
//
// Everything else — belief sampling, difficulty scaling, move selection — is
// inherited unchanged from the generic agent.
// ---------------------------------------------------------------------------

import { ObscuroAgent as GenericObscuroAgent, compactAction } from '../vendor/obscuro/src/ObscuroAgent.js';
import { makeHooks, runObscuroSearch } from '../vendor/obscuro/src/search.js';
import { isAttackedBy } from './board.js';
import { evaluate } from './ChessAgent.js';
import { toFEN, uciToAction } from './fen.js';
import {
  multiPV, stockfishBestAction, difficultyToNumber,
  available as stockfishAvailable,
} from './stockfish.js';
import { param, ramp, settingsProvenance } from './config.js';

// Difficulty-dial (0-100, t = difficulty/100) endpoints for chess's own two
// leaf evaluators (_leafEval's power-mode ladder rung, _proportionalPick's
// perfect-information sampling), on top of the generic search dial in
// vendor/obscuro/src/settings.js. Re-exported via src/settings.js.
export const CHESS_DIAL = {
  // Fog-search leaf eval (power mode): fixed Stockfish depth/breadth per node.
  leafEval: {
    sfDepth: { min: 2, max: 4, curve: 'linear' },   // measured — see the comment on this dial's use below
    cols:    { min: 5, max: 14, curve: 'linear' },  // MultiPV lines requested per node
  },
  // Perfect-information proportional pick: broad-but-shallow MultiPV scoring,
  // sampled by win-probability sharpness (beta).
  proportionalPick: {
    multipvCap: 20,                                 // min(legalActions.length, 20)
    depth: { min: 10, max: 16, curve: 'linear' },
    // beta: 0 → uniform, 1 at t=0.5 → probability ∝ win-prob, up to 12 at t=1 → near-best.
    betaAtHalf: 1,
    betaMax: 12,
  },
  // Time-mode leaf eval: how many evaluations of the move budget to reserve
  // per belief world for root expansion before the round loop starts (see
  // _leafEval's perCallMs formula: budget / (worlds * evalsPerWorldReserve)).
  timeModeEvalsPerWorldReserve: 8,
  timeModeMinPerCallMs: 30,
  // Perfect-information time mode plays Stockfish at full strength: the user's
  // clock is the only handicap, so there is no Skill Level one on top.
  timeModeSkill: 20,
};

// Effective values. The constants in this file are the DEFAULTS; a settings
// file, --set, or a per-session gameSpecific.obscuro bag may replace any of
// them, and every read goes through param() at use time so that reaches the
// module-level `ObscuroAgent` singleton below too. See docs/SETTINGS.md.
const chessDial = () => param('chess.CHESS_DIAL', CHESS_DIAL);
const analysisDefaults = () => param('chess.ANALYSIS_DEFAULTS', ANALYSIS_DEFAULTS);
const maxSfDepth = () => param('chess.MAX_SF_DEPTH', MAX_SF_DEPTH);
const searchWin = () => param('chess.SEARCH_WIN', SEARCH_WIN);
const leafClamp = () => param('chess.LEAF_CLAMP', LEAF_CLAMP);

// obscuroStrategy / analyzeObscuroProgressive defaults — the analysis-panel and
// test-harness search sizes, distinct from ChessObscuroAgent's own move-time
// dial above (these run outside a real move's difficulty setting).
export const ANALYSIS_DEFAULTS = {
  // obscuroStrategy: the inspection helper's own search-size fallbacks (used by
  // tests and by analyzeObscuroProgressive's per-batch mixing call below).
  strategy: { particles: 8, maxRounds: 30, expandPerRound: 8, cfrPerRound: 4, purifyMax: 3 },
  // analyzeObscuroProgressive: the belief-population walk's mixing search (run
  // once per batch, so it can afford to be bigger than obscuroStrategy's default).
  batchMixing: { maxRounds: 100, expandPerRound: 16, cfrPerRound: 8 },
  maxTotalMs: 5 * 60 * 1000,   // safety net for a missed disconnect, not a quality cap
  batchSize: 16,                // worlds enumerated/sampled per batch
  sweepBatches: 4,               // generative fallback: batches per ladder rung
  likelyWorldsCap: 32,           // "most likely boards" overlay size
  scoredWorldsCap: 96,           // per-world cp view cap
  // _captureStockfishAnalysis: the MultiPV ranking shown beside a time-mode
  // perfect-information move. Display only — it never changes what is played.
  captureMultipv: 8,
  captureDepth: 12,
};

// Material (Stockfish cp) scores are clamped so an imagined king capture from
// phantom hidden pieces can't swamp a concrete material decision.
export const LEAF_CLAMP = 1500;
const clip = v => { const c = leafClamp(); return v > c ? c : v < -c ? -c : v; };

// The search's terminal win/loss magnitude, on the same cp scale as the leaves.
// The paper bounds ALL utilities (u: Z → [−1,+1], evals clamped inside it):
// under fog, values are averaged across belief worlds, so a real game-ending
// outcome must outweigh material decisively but boundedly — with the generic
// default (±10⁶) a single phantom world in which the enemy king looked
// capturable would swamp every real consideration and send the AI lunging.
// 8000 ≈ 5.3× the material clamp: game-deciding, not belief-noise-proof.
export const SEARCH_WIN = 8000;

// Leaving your OWN king capturable is not merely "down some material" — it IS
// the terminal loss (−SEARCH_WIN), and it is a consequence of a move the mover
// CHOSE. Otherwise, under fog, a move that hangs the king in half the belief
// worlds gets averaged against ordinary material evals in the other half and
// comes out looking playable — which is exactly how the AI walked its king onto
// a square a hidden pawn was covering. This is deliberately asymmetric with the
// +LEAF_CLAMP cap on *capturing the enemy* king at a LEAF: an imagined capture
// is phantom-prone and must not be banked on, while exposing our own king is a
// real, self-inflicted loss we must avoid.
const kingHang = () => searchWin();

const otherColor = c => (c === 'white' ? 'black' : 'white');

function findKingSquare(board, color) {
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (p && p.ownerId === color && p.type === 'king') return sq;
  }
  return null;
}

// The top of the iterative-deepening ladder. A CEILING the ladder climbs toward,
// not a depth it will usually reach: timed at multipv 16 on an ordinary
// middlegame, depth 14 took ~1.4 s, depth 18 ~7 s, and everything past ~19 blew
// through multiPV's per-call timeout. (Those timings were taken on the Stockfish
// 11 build vendored until 2026-08-01; Stockfish 18 lite is ~2× faster at the
// shallow depths this search actually uses, so the deep rungs are reachable
// somewhat sooner — the ceiling is unchanged because it was never the binding
// constraint.) Whatever rung the caller's budget affords is the
// rung that gets reported, so the number the UI shows is always a real,
// completed search rather than an aspiration.
export const MAX_SF_DEPTH = 30;

// ONE rung of the ladder — the batched Stockfish node heuristic of the paper.
// Given a node `state` where `mover` is to play and the `actions` leading to its
// non-terminal children (with the already-applied `childStates`), returns the
// value TO THE MOVER of each child, plus whether the engine actually answered
// (`engineOk`) so a ladder above can tell "searched to depth d" from "fell back
// to the static evaluator because the engine timed out at depth d".
//
// One MultiPV call on the parent position scores all the mover's moves at once
// (cp is already from the mover's perspective). Children that hang the mover's
// king are a fog-of-war loss the engine cannot see, so they are scored directly.
// LEAF-EVAL DEGRADATION, MADE VISIBLE.
//
// When the engine returns nothing — a timeout, a busy cache, a worker that died
// mid-search — scoreChildren silently substitutes the static JS evaluator for
// that child, and the fixed-depth wrapper throws `engineOk` away. So the search
// keeps running and quietly plays a *different, weaker* game with no signal
// whatsoever. That is how five identical runs of move-quality.mjs, launched
// concurrently on one machine, produced five different cp-loss figures (52.3,
// 56.0, 67.6, 105.1, 84.4) despite identical seeds and identical positions: under
// load the engine calls started timing out into this fallback.
//
// The counters exist so that "was this measurement degraded?" is answerable
// after the fact instead of guessed at. Any harness reporting numbers from this
// search should print `fallbackLeaves` alongside them; a nonzero share means the
// run is not comparable with a clean one.
let leafStats = { calls: 0, engineLeaves: 0, fallbackLeaves: 0, truncated: 0, refusedNodes: 0,
  pvNullNodes: 0, pvShortNodes: 0, unmappedNodes: 0, engineUnavailable: 0 };
export function getLeafEvalStats() { return { ...leafStats }; }
export function resetLeafEvalStats() {
  leafStats = { calls: 0, engineLeaves: 0, fallbackLeaves: 0, truncated: 0, refusedNodes: 0,
    pvNullNodes: 0, pvShortNodes: 0, unmappedNodes: 0, engineUnavailable: 0 };
}

// Squares one king-step apart. Two kings adjacent is a position standard chess
// cannot reach and Stockfish will not evaluate — under fog it is ordinary.
function kingsAdjacent(a, b) {
  if (!a || !b) return false;
  return Math.abs(a.charCodeAt(0) - b.charCodeAt(0)) <= 1
      && Math.abs(a.charCodeAt(1) - b.charCodeAt(1)) <= 1;
}

/**
 * Would a standard chess engine REFUSE this position? Under fog these states are
 * routine — there is no check rule, so attacking the enemy king is just a strong
 * move rather than an illegal position — but Stockfish returns zero lines for
 * them, which used to silently drop every child of such a node onto the static
 * evaluator. Measured at 10.19% of all leaf evaluations.
 */
function engineWouldRefuse(board, mover) {
  const them = otherColor(mover);
  const theirK = findKingSquare(board, them);
  if (!theirK) return true;                            // king already captured
  if (isAttackedBy(board, theirK, mover)) return true; // side-not-to-move "in check"
  return kingsAdjacent(findKingSquare(board, mover), theirK);
}

// How many children of a refused node get a real engine evaluation of their own.
// The parent cannot be scored in one MultiPV call, so each child needs its own —
// which is why this is capped rather than unbounded: refused nodes are ~10% of
// all nodes, so pricing every child of every one of them would multiply engine
// work ~4×. The cap spends the budget on the children most likely to matter
// (best static score first) and leaves the tail on the static evaluator, which
// is what the whole node used to get.
const REFUSED_CHILD_CAP = Number(
  (typeof process !== 'undefined' && process.env?.OBSCURO_REFUSED_CHILD_CAP) ?? 8);

async function scoreChildren(state, mover, actions, childStates, { sfDepth, cols, isCancelled }) {
  const them = otherColor(mover);
  const out = new Array(actions.length);
  const need = [];
  for (let i = 0; i < actions.length; i++) {
    const board = (childStates?.[i] ?? state).board;
    const k = findKingSquare(board, mover);
    if (!k || isAttackedBy(board, k, them)) { out[i] = -kingHang(); continue; } // hung own king → losing move
    need.push(i);
  }
  // Nothing left for the engine to price (every child hangs the king) — the
  // answer is exact and depth-independent, so it counts as a completed rung.
  let engineOk = need.length === 0;
  let truncated = false;

  // A position the engine will refuse: skip the doomed MultiPV call on the
  // parent (it costs a full timeout to learn nothing) and price the children
  // individually instead. Each CHILD is legal — the opponent is to move and
  // merely in check — so the engine answers there. cp comes back from the
  // child's mover (them), so it is negated onto our scale.
  if (need.length && engineWouldRefuse(state.board, mover) && await stockfishAvailable()) {
    const order = [...need].sort((a, b) =>
      evaluate(childStates[b].board, mover) - evaluate(childStates[a].board, mover));
    const priced = new Set();
    const side = them === 'white' ? 'w' : 'b';
    for (const i of order.slice(0, REFUSED_CHILD_CAP)) {
      const cs = childStates[i];
      if (engineWouldRefuse(cs.board, them)) continue; // child refused too — leave it
      let pv = null;
      try {
        pv = await multiPV(toFEN(cs.board, cs.gameSpecific, side, cs.turnNumber ?? 1),
          { multipv: 1, depth: sfDepth, isCancelled, onStopped: () => { truncated = true; } });
      } catch { pv = null; }
      if (pv?.length && typeof pv[0].cp === 'number') {
        out[i] = clip(-pv[0].cp);
        priced.add(i);
        leafStats.engineLeaves++;
      }
      leafStats.calls++;
    }
    for (const i of need) {
      if (priced.has(i)) continue;
      leafStats.fallbackLeaves++;
      out[i] = clip(evaluate(childStates[i].board, mover));
    }
    leafStats.refusedNodes++;
    if (truncated) leafStats.truncated++;
    // The rung is "complete" when the engine answered for the children we chose
    // to price; the capped tail is a deliberate approximation, not a truncation.
    return { scores: out, engineOk: priced.size > 0 && !truncated };
  }

  if (need.length) {
    let pv = null;
    if (await stockfishAvailable()) {
      const side = mover === 'white' ? 'w' : 'b';
      try {
        pv = await multiPV(toFEN(state.board, state.gameSpecific, side, state.turnNumber ?? 1),
          {
            // ALL actions, not just `need`: MultiPV returns the engine's own top-N
            // legal moves, and asking for `need.length` (which excludes the
            // king-hanging children we score ourselves) means the engine's N and
            // ours are different sets — every move in ours but not in its top-N
            // came back unpriced and silently fell through to the static
            // evaluator. Width is nearly free (measured: multipv 1 and multipv 40
            // cost the same), so ask for enough to cover everything.
            multipv: Math.max(actions.length, cols), depth: sfDepth,
            isCancelled, onStopped: () => { truncated = true; },
          });
      } catch { pv = null; }
    }
    const cpByIdx = new Map();
    // Categorise WHY a node loses values, so the residual fallback rate can be
    // attributed instead of guessed at: engine said nothing / said less than we
    // asked / said plenty but not about our moves.
    if (!pv || !pv.length) leafStats.pvNullNodes++;
    else if (pv.length < need.length) leafStats.pvShortNodes++;
    if (pv && pv.length) {
      engineOk = !truncated;
      for (const { move, cp } of pv) {
        const a = uciToAction(move, actions);
        if (a) { const i = actions.indexOf(a); if (i >= 0) cpByIdx.set(i, cp); }
      }
    }
    if (pv && pv.length && cpByIdx.size < need.length) leafStats.unmappedNodes++;
    if (process.env?.OBSCURO_DEBUG_FALLBACK && cpByIdx.size < need.length
        && leafStats.calls % Number(process.env.OBSCURO_DEBUG_FALLBACK || 1) === 0) {
      const side = mover === 'white' ? 'w' : 'b';
      console.error(`[fallback] actions=${actions.length} need=${need.length} pv=${pv?.length ?? 'null'} ` +
        `mapped=${cpByIdx.size} depth=${sfDepth} cols=${cols}\n  fen: ` +
        toFEN(state.board, state.gameSpecific, side, state.turnNumber ?? 1) +
        `\n  unpriced: ` + need.filter(i => !cpByIdx.has(i)).slice(0, 8)
          .map(i => `${actions[i].from}${actions[i].to}${actions[i].type === 'castle' ? '(O-O)' : ''}`).join(' '));
    }
    for (const i of need) {
      const fromEngine = cpByIdx.has(i);
      if (fromEngine) leafStats.engineLeaves++; else leafStats.fallbackLeaves++;
      out[i] = fromEngine ? clip(cpByIdx.get(i)) : clip(evaluate(childStates[i].board, mover));
    }
    leafStats.calls++;
    if (truncated) leafStats.truncated++;
  }
  return { scores: out, engineOk };
}

// Fixed-depth leaf evaluator: the search asks for depth `sfDepth` and gets it.
// Stockfish deepens internally on the way (`go depth N` sweeps 1..N), so a
// single call already IS the ladder when the target depth is known up front —
// which is the power-mode case, where the dial fixes depth and breadth.
export function makeChessLeafEval(sfDepth, cols, { isCancelled } = {}) {
  return async (state, mover, actions, childStates) =>
    (await scoreChildren(state, mover, actions, childStates, { sfDepth, cols, isCancelled })).scores;
}

// Time-bounded leaf evaluator: climb the ladder one rung at a time — depth 1,
// then 2, then 3 … — re-scoring every child from scratch at each rung, and keep
// the deepest rung that COMPLETED before the budget ran out. Each rung is a
// self-contained `go depth d`, so a rung that gets cut short is simply discarded
// in favour of the last complete one; the caller always holds a coherent set of
// scores all measured at the same depth.
//
// Why re-search from depth 1 instead of one `go depth 30`: the point is to have
// a usable answer at every instant, not only at the end. That matters for a wall
// clock that can expire at any moment, and it is what lets the analysis panel
// report "these are the top moves as of depth d" for a whole population of
// belief worlds at once — every world must sit at the SAME depth for its scores
// to be averaged together. Stockfish's own internal deepening cannot do that,
// because it is private to a single call on a single position.
//
//   perCallMs — the slice of the move budget this one evaluation may spend.
//   deadline  — absolute wall-clock stop for the whole move; never overrun it.
export function makeIterativeChessLeafEval({
  maxDepth = maxSfDepth(), cols = 0, perCallMs = 250, deadline = Infinity, isCancelled,
} = {}) {
  return async (state, mover, actions, childStates) => {
    const stopAt = Math.min(deadline, Date.now() + perCallMs);
    // Rungs past the first are also stoppable mid-search, so overshoot is
    // bounded by the poll interval rather than by a whole rung's cost.
    const rungCancelled = () => (isCancelled?.() ?? false) || Date.now() > stopAt;
    let best = null;
    for (let d = 1; d <= maxDepth; d++) {
      // Depth 1 always runs: an evaluator that returns nothing would leave the
      // search with no values at all, which is worse than a shallow answer.
      if (d > 1 && rungCancelled()) break;
      const { scores, engineOk } = await scoreChildren(state, mover, actions, childStates, {
        sfDepth: d, cols, isCancelled: d > 1 ? rungCancelled : undefined,
      });
      if (!engineOk && best) break; // rung cut short or engine gave up — keep the last complete one
      best = scores;
      if (!engineOk) break;         // no engine at all: deeper rungs would be identical
    }
    return best;
  };
}

// THE GAME the agent reasons with. It defaults to this package's FogChess and is
// resolved lazily, because the definition imports this module (for its
// evaluateState/agents) and a static import back would be circular.
//
// An embedder whose engine has its own GameDefinition — battle-simulator's
// ChessGame, with fog markers, a renderer and difficulty options on top of these
// same rules — calls setGame() once at startup. The search then applies exactly
// the rules the engine will apply, which is the whole point: a tree built on a
// different applyActions than the one that will run is a tree of positions that
// will not happen.
let GAME = null;
export function setGame(game) { GAME = game; }
export async function getGame() {
  if (!GAME) GAME = (await import('./FogChess.js')).FogChess;
  return GAME;
}

export class ChessObscuroAgent extends GenericObscuroAgent {
  constructor(opts = {}) {
    // The generic base needs a truthy game; the real one is attached lazily on
    // first use (see getGame above).
    super({}, { id: 'obscuro', name: 'Obscuro (CFR)', ...opts });
  }

  async _game() {
    this.game = await getGame();
    return this.game;
  }

  // Per-move search sizes.
  //
  // The generic base owns the search knobs and resolves them itself — its dial
  // endpoints and curve are `search.*` settings, forwarded to it by config.js —
  // so all this adds is the per-SESSION override bag, between the settings file
  // and this instance's own opts:
  //
  //   constructor opts  >  gameSpecific.obscuro  >  settings file / --set  >  dial
  //
  // Resolution happens here per move rather than at construction because the
  // production entry point is the module-level `ObscuroAgent` singleton below,
  // built at import — before any host can configure anything.
  _config(observation) {
    const session = observation.gameSpecific?.obscuro;
    if (!session) return super._config(observation);
    const saved = this.opts;
    this.opts = { ...session, ...saved };   // the instance's own opts still win
    try { return super._config(observation); } finally { this.opts = saved; }
  }

  // Bounded terminal value for the fog search (see SEARCH_WIN above).
  _winValue() { return searchWin(); }

  // Chess's batched Stockfish node heuristic, its depth/width scaled by the dial.
  // The paper runs its leaf evaluation at DEPTH 1 (App. C.5) and gets its
  // strength from the search aggregating many worlds and growing the tree. Deep
  // leaves here (the old 2..10 ramp) dated from when the multi-world
  // aggregation was broken and each leaf had to carry the position alone; they
  // also made a cold-cache expansion so slow that only a couple of belief
  // worlds fit in the budget — reintroducing single-world behaviour through the
  // back door. Shallow-ish leaves keep every world expandable within budget;
  // see the measured depth-vs-tree-size tradeoff below (CHESS_DIAL.leafEval)
  // for where the current top of the range comes from.
  //
  // The two dials pick the two forms of the SAME evaluator (see
  // makeIterativeChessLeafEval): POWER fixes the ladder's top rung and its
  // breadth outright, so every leaf is priced at exactly the depth the dial
  // bought. A TIME limit buys no fixed depth at all — it buys full breadth
  // (every legal child, always) and as many rungs as the clock allows, which is
  // the same iterative deepening the analysis panel runs.
  _leafEval(observation) {
    const gs = observation.gameSpecific ?? {};
    const timeMs = gs.aiTimeMs;
    if (typeof timeMs === 'number' && timeMs > 0) {
      // Spread the move's wall clock over the evaluations a search actually
      // makes rather than letting the first one swallow it: root expansion alone
      // costs one evaluation per belief world BEFORE the round loop starts, so a
      // per-call slice of budget/(worlds × 8) keeps root expansion at an eighth
      // of the budget and leaves the rest for tree growth. cols 0 = no minimum
      // breadth, because breadth is already full: scoreChildren always asks for
      // at least one MultiPV line per child that needs one.
      const cfg = this._config(observation);
      const { timeModeEvalsPerWorldReserve: reserve, timeModeMinPerCallMs: floorMs } = chessDial();
      const perCallMs = Math.max(floorMs, Math.round(timeMs / (Math.max(1, cfg.worlds ?? 1) * reserve)));
      return makeIterativeChessLeafEval({
        maxDepth: maxSfDepth(), cols: 0, perCallMs, deadline: Date.now() + timeMs,
      });
    }
    const t = difficultyToNumber(gs.difficulty) / 100;
    // Leaf depth tops out at 4, NOT 7 — measured, not guessed. ~80% of a move's
    // wall clock is inside these calls (~7 ms at depth 2, ~11 ms at 4, ~23 ms at
    // 7), so depth trades directly against tree size, and move-quality.mjs
    // measured that trade on 128 identical positions at matched cost (~1.0 s/move
    // both ways): mean cp loss against a deep reference was
    //
    //   depth 2, 18 rounds  108.9      depth 7, 4 rounds  121.5
    //   depth 4,  8 rounds  109.2      depth 1, 36 rounds 142.3
    //
    // Depth 7 is DOMINATED — it buys tactical leaves with tree size, and the tree
    // was worth more. Depth 1 is the paper's own design point and is the worst
    // option here, because our trees are ~100× smaller than the ~10⁶-node trees
    // that make shallow leaves work: at this scale the leaves have to see the
    // tactics the tree cannot. Revisit the top of this range only after the tree
    // grows by an order of magnitude. `sfDepth` overrides it so that re-measuring
    // stays a flag rather than an edit.
    const { sfDepth: sfDepthR, cols: colsR } = chessDial().leafEval;
    const sfDepth = this.opts.sfDepth ?? Math.max(1, ramp(sfDepthR, t));
    const cols = ramp(colsR, t);
    return makeChessLeafEval(sfDepth, cols);
  }

  async chooseAction(state, legalActions) {
    if (!legalActions?.length) return null;
    if (legalActions.length === 1) return legalActions[0];
    await this._game();

    const gs = state.gameSpecific;
    // Perfect information (fog off): play Stockfish at FULL strength — no Skill Level
    // handicap. A 0 power level / 0 ms limit is random and falls through to the generic
    // random branch. Otherwise:
    //   • time mode  → the single strongest move within the movetime budget.
    //   • power mode → the engine scores every move at full strength and we SAMPLE one
    //     in proportion to its score (see _proportionalPick). Weaker play at lower power
    //     comes from the softer sampling, not from a hobbled engine, so the AI plays
    //     worse gradually instead of dropping pieces outright.
    const timeMs = gs.aiTimeMs;
    const isRandom = timeMs === 0 || (timeMs == null && gs.difficulty === 0);
    let action = null;
    if (!gs.fogOfWar && !isRandom) {
      if (typeof timeMs === 'number') {
        const sfOpts = { movetime: Math.min(Math.max(timeMs, 1), 600000), skill: chessDial().timeModeSkill };
        const sf = await stockfishBestAction(state, legalActions, sfOpts);
        if (sf) {
          await this._captureStockfishAnalysis(state, legalActions, sf, sfOpts, gs);
          action = this._matchLegal(sf, legalActions) ?? sf;
        }
      } else {
        action = await this._proportionalPick(state, legalActions, gs);
      }
    }
    action ??= await super.chooseAction(state, legalActions);
    this._reportSettings(state);
    return action;
  }

  // Record on lastAnalysis which parameters were not left at their defaults, so
  // "why did this move come out like that?" is answerable from the analysis
  // panel or a test rather than by guessing at the caller's environment. Empty
  // (and omitted) in the default configuration.
  _reportSettings(observation) {
    if (!this.lastAnalysis) return;
    const overridden = {};
    for (const [path, source] of settingsProvenance()) overridden[path] = source;
    for (const key of Object.keys(this.opts)) {
      if (!['id', 'name', 'rng'].includes(key)) overridden[`opts.${key}`] = 'agent';
    }
    for (const key of Object.keys(observation.gameSpecific?.obscuro ?? {})) {
      overridden[`opts.${key}`] = 'session';
    }
    if (Object.keys(overridden).length) this.lastAnalysis.overrides = overridden;
  }

  // NOTE: there is deliberately no selection-time king-safety backstop any more.
  // An earlier `_kingSafetyGuard` (via _adjustChosenAction) re-sampled the belief
  // and vetoed near-tie moves that hung the king in many worlds, back when the
  // search itself mispriced king-hangs. After the search fixes (infoset
  // action-set invariant, uCond reach weighting, bounded terminals, exact belief,
  // tree carryover) a 24-game / 1787-ply validation measured the with-safe-move
  // king-hang rate at 1.4% with the guard firing on only 0.34% of plies — and
  // batches where it never fired still met the <2% target — so it was removed:
  // play is now genuinely equilibrium-driven (plan doc Phase 4).

  // Power mode, perfect information: score every legal move at full strength, then
  // pick one at random weighted by its score. Scores are converted to win
  // probabilities (a principled, always-positive measure), so a move worth ~twice
  // the win chance of another is played ~twice as often. The power dial only sets
  // the SHARPNESS of that sampling (β): at power 50 the probability is exactly
  // proportional to the win-prob score; higher power sharpens toward the best move,
  // lower power flattens toward uniform. No Skill Level, so no gratuitous blunders.
  async _proportionalPick(state, legalActions, gs) {
    try {
      const us = state.activePlayers[0];
      const fen = toFEN(state.board, state.gameSpecific, us === 'white' ? 'w' : 'b', state.turnNumber ?? 1);
      const t = difficultyToNumber(gs.difficulty) / 100;
      // Score broadly so weak-but-legal moves stay reachable at low power; deeper at
      // higher power for more accurate scores (strength there comes from sharper β too).
      const { multipvCap, depth: depthR, betaAtHalf, betaMax } = chessDial().proportionalPick;
      const multipv = Math.min(legalActions.length, ramp(multipvCap, t));
      const depth = ramp(depthR, t);
      const pv = await multiPV(fen, { multipv, depth });
      if (!pv || !pv.length) return null;

      // cp (mover's perspective) → win probability in (0,1). Mate scores saturate.
      const winProb = cp => (cp >= 90000 ? 1 : cp <= -90000 ? 0 : 1 / (1 + Math.pow(10, -cp / 400)));
      // β: 0 → uniform, betaAtHalf at t=0.5 → probability ∝ win-prob, betaMax at t=1 → near-best.
      const beta = t <= 0.5 ? (t / 0.5) * betaAtHalf : (betaAtHalf + (t - 0.5) / 0.5 * (betaMax - betaAtHalf));

      const scored = [];
      for (const { move, cp } of pv) {
        const a = uciToAction(move, legalActions);
        if (!a) continue;
        const wp = winProb(cp);
        scored.push({ action: a, cp, weight: Math.pow(wp, beta) });
      }
      if (!scored.length) return null;

      // Full power is the true perfect-information special case: it collapses to
      // PURE Stockfish play (deterministic best move), no sampling softness left.
      if (t >= 0.999) {
        let best = scored[0];
        for (const x of scored) if (x.cp > best.cp) best = x;
        for (const x of scored) { x.weight = x === best ? 1 : 0; }
      }
      const total = scored.reduce((s, x) => s + x.weight, 0) || 1;
      for (const x of scored) x.prob = x.weight / total;

      // Sample proportional to weight.
      let r = this._rng() * total, chosen = scored[0].action;
      for (const x of scored) { r -= x.weight; if (r <= 0) { chosen = x.action; break; } }

      const chosenKey = this._key(chosen);
      const rank = scored.findIndex(x => this._key(x.action) === chosenKey);
      this.lastAnalysis = {
        ts: Date.now(),
        player: us,
        engine: 'stockfish',
        mode: 'Stockfish · proportional',
        difficulty: gs.difficulty ?? null,
        depth,
        chosenRank: rank >= 0 ? rank + 1 : null,
        candidates: scored.map(x => ({
          key: this._key(x.action), move: compactAction(x.action),
          cp: x.cp, prob: x.prob, chosen: this._key(x.action) === chosenKey,
        })),
        totalCandidates: legalActions.length,
      };
      return this._matchLegal(chosen, legalActions) ?? chosen;
    } catch { return null; } // any engine hiccup → fall back to the generic search
  }

  // Time mode (perfect information): the strongest move within the movetime budget.
  // The MultiPV ranking is captured purely for the analysis panel.
  async _captureStockfishAnalysis(state, legalActions, chosen, sfOpts, gs) {
    try {
      const us = state.activePlayers[0];
      const fen = toFEN(state.board, state.gameSpecific, us === 'white' ? 'w' : 'b', state.turnNumber ?? 1);
      const { captureMultipv, captureDepth } = analysisDefaults();
      const pv = await multiPV(fen, { multipv: captureMultipv, depth: captureDepth });
      const chosenKey = this._key(chosen);
      let cands = [];
      if (pv && pv.length) {
        cands = pv.map(({ move, cp }) => {
          const a = uciToAction(move, legalActions);
          return a ? { key: this._key(a), move: compactAction(a), cp, chosen: this._key(a) === chosenKey } : null;
        }).filter(Boolean);
      }
      if (!cands.some(c => c.chosen)) {
        cands.push({ key: chosenKey, move: compactAction(chosen), cp: null, chosen: true });
      }
      const rank = cands.findIndex(c => c.chosen);
      this.lastAnalysis = {
        ts: Date.now(),
        player: us,
        engine: 'stockfish',
        mode: 'Stockfish · best',
        difficulty: gs.difficulty ?? null,
        movetimeMs: sfOpts.movetime ?? null,
        chosenRank: rank >= 0 ? rank + 1 : null,
        candidates: cands,
        totalCandidates: cands.length,
      };
    } catch { /* analysis is best-effort; never break move selection */ }
  }
}

export const ObscuroAgent = new ChessObscuroAgent();

// ---------------------------------------------------------------------------
// Inspection helper (used by tests): run Obscuro's search for one position and
// return the candidate moves, the last-iterate distribution over them, the
// chosen move, its value and the solve mode ('minimax' with perfect info, 'cfr'
// under fog). Uses the generic static evaluation (not Stockfish) so it is
// deterministic and does not require the engine.
// ---------------------------------------------------------------------------
export async function obscuroStrategy(state, legalActions, opts = {}) {
  const rng = opts.rng ?? Math.random;
  // Defaults to whoever's actually to move, but a caller may override — e.g.
  // the analysis API always passes the requesting viewer's own colour under
  // fog, so "what's good for my side" stays answerable even when it isn't
  // literally their turn yet (battle-simulator's analysis endpoint does this). When it
  // overrides to a colour that ISN'T state.activePlayers[0], the state itself
  // is patched to match: the generic search's tree-building derives whose
  // move a node represents from activePlayers at every level it touches, not
  // just this root call's `me`, so a root that internally still claims "black
  // to move" while search/tree code is told `me = white` desyncs partway
  // through and returns nonsense (a piece move for a side that isn't even on
  // this board). Presenting a state that honestly says "white to move" keeps
  // the whole tree self-consistent for this hypothetical/counterfactual read.
  const me = opts.color ?? state.activePlayers[0];
  if (me !== state.activePlayers[0]) state = { ...state, activePlayers: [me] };
  const game = await getGame();
  const fog = !!state.gameSpecific.fogOfWar;

  // opts.worlds lets a caller supply the belief cloud explicitly (the batched
  // enumeration cursor passes the next slice of the population — see
  // analyzeObscuroProgressive) instead of sampling a fresh one here.
  let worlds = opts.worlds ?? (fog ? game.sampleWorlds(state, me, opts.particles ?? analysisDefaults().strategy.particles, rng) : null);
  if (!worlds || worlds.length === 0) worlds = [state];

  const hooks = makeHooks(game, me, { rng });
  const opp = (state.players ?? []).find(p => p.id !== me)?.id ?? null;
  // Live "round N/M" progress (lichess-style depth ticks, but for CFR rounds) —
  // purely a side channel; see runObscuroSearch's cfg.onRound.
  const onRound = opts.onProgress
    ? (round, maxRounds, info) => opts.onProgress({ kind: 'round', round, maxRounds, candidates: rankCandidates(info.rows, info.dist) })
    : undefined;
  const strategyDefaults = analysisDefaults().strategy;
  const res = await runObscuroSearch(hooks, worlds, {
    opp, rootActions: legalActions, rng,
    timeBudgetMs: opts.timeBudgetMs ?? 0,
    maxRounds: opts.maxRounds ?? strategyDefaults.maxRounds,
    expandPerRound: opts.expandPerRound ?? strategyDefaults.expandPerRound,
    cfrPerRound: opts.cfrPerRound ?? strategyDefaults.cfrPerRound,
    purifyMax: opts.purifyMax ?? strategyDefaults.purifyMax,
    onRound,
    // So a solve stops mid-flight when the analysis position changes (rather
    // than running out its rounds after the viewer has already moved on).
    isCancelled: opts.isCancelled,
  });

  const k = game.actionKey;
  const action = legalActions.find(a => k(a) === k(res.action)) ?? res.action ?? legalActions[0];
  return { mode: fog ? 'cfr' : 'minimax', action, dist: res.dist, rows: res.rows, value: res.value, particles: worlds.length };
}

// Shared with the onRound progress callback above so a mid-search snapshot and
// the final result are ranked identically. Sorted by probability (how much of
// the equilibrium's mass this move gets) descending, ties — most of them,
// since only a handful of moves ever get nonzero mass — broken by cp
// (highest first) once one's available (see analyzeObscuroProgressive's eval
// ladder below; mid-search progress ticks have no cp yet, so ties there just
// keep whatever order `rows` came in).
function rankCandidates(rows, dist) {
  return (rows ?? [])
    .map((action, i) => ({ move: action, prob: dist?.[i] ?? 0 }))
    .sort((a, b) => (b.prob - a.prob) || ((b.cp ?? -Infinity) - (a.cp ?? -Infinity)));
}

// ---------------------------------------------------------------------------
// Read-only position analysis for the UI's "suggest a move" panel: runs the
// exact same solve as obscuroStrategy (fog-aware via the belief population when
// applicable, perfect-info minimax otherwise) and reshapes its output into
// ranked candidates, without ever selecting/committing a move for real play.
//
// There used to be two disconnected branches here: a perfect-information one
// that asked Stockfish directly (deep, live "Depth N/14" ticks, no belief
// framing) and a fog one that walked belief worlds at a fixed shallow depth
// (wider and wider, never deeper). That split was artificial. PERFECT
// INFORMATION IS JUST A BELIEF POPULATION OF SIZE 1 — nothing is hidden, so
// exactly one world is consistent with the observation — and once
// FogChess.beliefPopulation says so, the one progressive walk below covers both
// regimes: it refines along BOTH axes, more worlds and more depth, and a
// population of 1 simply spends every batch on the single real position.
// ---------------------------------------------------------------------------
export async function analyzeObscuro(state, legalActions, opts = {}) {
  if (!legalActions?.length) return { engine: 'obscuro', mode: 'none', candidates: [] };
  return await analyzeObscuroProgressive(state, legalActions, opts);
}

// Batched Stockfish leaf eval over an EXPLICIT set of belief worlds, all scored
// at the SAME depth (`opts.sfDepth`) so their scores are commensurable: returns
// the per-legal-move MASS-WEIGHTED SUM of cp across the worlds it managed to
// score (Σ w·cp), the total mass `wsum` of those worlds, and their count `n`.
//
// The weighting is the point. Belief worlds are NOT equally likely — each carries
// its posterior probability as `beliefWeight` (see FogChess.enumerateWorlds) —
// so a plain mean would be an average over the wrong measure, giving a world the
// opponent almost certainly did not play into the same say as one they probably
// did. Σ(w·cp)/Σw is the population expectation: exact once the walk is
// exhaustive, and an unbiased running estimate while it is partial. Worlds with
// no weight (the generative fallback, which samples uniformly, and sampled worlds
// generally, whose weight is already in the draw) default to 1 and reduce this to
// the plain mean.
//
// Kept raw (sums + mass, not a mean) so a caller folding many batches together
// forms the exact population expectation instead of averaging per-batch means
// (which would misweight unequal final batches). `n` is reported alongside because
// "did the engine manage to price anything at this depth" is a count question, not
// a mass one. Bails promptly when the caller has moved on or the budget is spent,
// discarding any world whose evaluation was interrupted part-way rather than
// folding a half-searched score into the mean.
export async function cpSumsOverWorlds(game, worlds, color, legalActions, cols, opts = {}) {
  const { sfDepth = maxSfDepth(), isCancelled, deadline, onWorld } = opts;
  const stop = () => (isCancelled?.() ?? false) || (deadline != null && Date.now() > deadline);
  const leafEval = makeChessLeafEval(sfDepth, cols, { isCancelled: stop });
  const sums = new Array(legalActions.length).fill(0);
  let n = 0, wsum = 0;
  for (let w = 0; w < worlds.length; w++) {
    const world = worlds[w];
    if (stop()) break;
    const childStates = legalActions.map(a => game.applyActions(world, [{ playerId: color, action: a }]));
    const scores = await leafEval(world, color, legalActions, childStates);
    if (!scores) continue;
    if (stop()) break; // interrupted mid-world — its scores are partial, drop them
    const pw = world.beliefWeight ?? 1;
    n++; wsum += pw;
    for (let i = 0; i < scores.length; i++) sums[i] += pw * scores[i];
    // Side channel for the UI's per-world view: the aggregate above answers
    // "how good is this move on average", but the panel also lets a viewer ask
    // "which board makes THIS move look best", which needs the individual
    // world's scores kept rather than summed away.
    onWorld?.(w, world, scores);
  }
  return { sums, wsum, n };
}

// Walk order for the batched enumeration: DESCENDING posterior weight, so the
// first batches carry most of the population's mass. The weighted running mean is
// unbiased in any order, but front-loading the mass means a partial walk's numbers
// are close to the final ones within a batch or two instead of after full
// coverage — and the panel's "top N most likely boards" then falls out of the walk
// for free, since first-seen and heaviest coincide (which is also why the
// scoredWorlds cap below can keep first-seen worlds and still be keeping the
// heaviest).
//
// The trade-off is deliberate and must not be misdescribed: partial coverage is
// biased toward heavy worlds. That is what you want from an ESTIMATE of the
// population mean; it is not a uniform sample of the population.
function weightOrder(probs, n) {
  if (!probs || probs.length !== n) return null;
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx;
}

// Fisher-Yates permutation of [0, n), the fallback walk order when no posterior is
// available to sort by — every world covered exactly once, but early batches
// aren't spatially biased toward one region of the position set. Built once per
// analysis session (not per batch); an n-int array for n up to the exact tracker's
// cap (~200k) is a few MB, released when the walk ends.
function shuffledIndices(n, rng) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  return idx;
}

// ---------------------------------------------------------------------------
// The single analysis walk. It refines along TWO axes at once:
//
//   • WIDTH — the belief population. Walk every world consistent with what the
//     viewer can see, in batches, folding each batch into a running estimate.
//   • DEPTH — an iterative-deepening ladder over the Stockfish leaf eval. Score
//     the whole population at depth 1, then re-score it all at depth 2, then 3 …
//     up to MAX_SF_DEPTH, so there is a complete, self-consistent answer at
//     every rung instead of one long opaque wait for a deep one.
//
// Depth is the OUTER loop and the population the inner one, because an average
// is only meaningful over worlds scored at the same depth: mixing a depth-20
// world with a depth-3 world would weight the position by how far down the
// batch queue it happened to land. So each rung is a complete sweep, and its
// cp aggregates are discarded and rebuilt from scratch at the next rung.
//
// The MIXING probabilities are not re-derived per rung: they come from the CFR
// tree, which prices its own leaves with the game's cheap static evaluator and
// never consults Stockfish, so they do not depend on `sfDepth` at all. Each
// batch of NEW worlds contributes its equilibrium once — which for the exact
// population means the first sweep alone, since later sweeps revisit the very
// same worlds, and for the generative fallback means every sweep, since each
// draws fresh samples.
//
// Two population regimes (see FogChess.beliefPopulation):
//   • EXACT belief (perfect information, and the common fog case after the
//     opening): P is a real array, enumerated WITHOUT replacement via a one-time
//     shuffled cursor — every world covered exactly once per rung, `total`
//     known, coverage marching to 100%. Perfect information is this regime with
//     total === 1.
//   • Heuristic fallback (exact tracking lost): belief.js is generative with no
//     enumerable set, so each sweep samples fresh worlds (with replacement),
//     `total` is null, and a sweep is capped at a fixed batch count so the
//     ladder can still climb.
//
// Aggregation (see OBSCURO-UNLIMITED-BELIEF-PLAN.md's "crux"): the cp EVAL per
// move is additive over worlds, so a MASS-weighted running mean — Σ(w·cp)/Σw over
// each world's posterior probability, see cpSumsOverWorlds — converges to the
// exact population expectation. The move PROBABILITY is an ensemble average of
// each batch's own CFR equilibrium (weighted by batch mass) — a
// well-defined blend, but NOT the single joint-equilibrium mixing (that would
// need the KLUSS gadget to grow its world set mid-solve — Design A, not
// attempted). Cancellation is checked between AND within batches, and now also
// inside the engine call itself (stockfish.js's UCI `stop`), so a stale walk
// stops within a round of the position changing rather than a deep rung later.
// maxTotalMs is a safety net for a missed disconnect, not a quality cap.
// ---------------------------------------------------------------------------
export async function analyzeObscuroProgressive(state, legalActions, opts) {
  const maxTotalMs = opts.maxTotalMs ?? analysisDefaults().maxTotalMs;
  const t0 = Date.now();
  const deadline = t0 + maxTotalMs;
  const game = await getGame();
  const k = game.actionKey;
  const rng = opts.rng ?? Math.random;
  const isCancelled = opts.isCancelled;
  const spent = () => (isCancelled?.() ?? false) || Date.now() >= deadline;

  // Analyze the requesting side's move — patch activePlayers up front (mirrors
  // obscuroStrategy) so every enumerated world is built with the right side to
  // move; the players-array identity is preserved, so the maintained belief is
  // still found by the game's beliefPopulation WeakMap lookup.
  const me = opts.color ?? state.activePlayers[0];
  if (me !== state.activePlayers[0]) state = { ...state, activePlayers: [me] };

  const cols = Math.min(legalActions.length, 16);
  const batchSize = opts.batchSize ?? analysisDefaults().batchSize;
  const maxDepth = opts.maxSfDepth ?? maxSfDepth();
  // Generative fallback only: how many batches count as one "sweep" of an
  // unbounded population before the ladder moves up a rung.
  const sweepBatches = opts.sweepBatches ?? analysisDefaults().sweepBatches;

  // cp source: run Stockfish over each batch, wherever it's available — server
  // (Node worker thread) or browser (nested Worker over the vendored WASM
  // build; see stockfish.js). `opts.cpEval` is an optional override some
  // caller can still supply instead. When neither is available, candidates
  // stay prob-only (cp: null) and the depth ladder is meaningless, so the walk
  // does a single sweep.
  const cpEval = opts.cpEval
    // `onWorld` is forwarded so an override can feed the per-world view too (the
    // real evaluator below reports every world it prices through it).
    ? ((worlds, sfDepth, onWorld) => opts.cpEval(worlds, legalActions, sfDepth, onWorld))
    : ((await stockfishAvailable())
        ? ((worlds, sfDepth, onWorld) => cpSumsOverWorlds(game, worlds, me, legalActions, cols, { sfDepth, isCancelled, deadline, onWorld }))
        : null);

  const pop = game.beliefPopulation(state, me);
  const total = pop.exact ? pop.total : null;

  // ── the per-world view (see buildBeliefWorlds) ────────────────────────────
  // Everything above collapses the belief population into ONE ranked move list.
  // The panel additionally lets a viewer look at the population itself: step
  // through the most likely boards, or ask which board makes a particular
  // candidate move look best. Both need individual worlds kept, so gather them
  // alongside the aggregates — bounded, since the population runs to ~200k and
  // this is a payload that crosses a Worker/SSE boundary every few frames.
  //
  // Only under fog. With perfect information there is nothing hidden to guess at
  // — the population is the one board already on screen — so the whole per-world
  // channel would be an empty payload on every frame.
  const perWorldView = !!state.gameSpecific.fogOfWar;
  const likelyCap = opts.likelyWorldsCap ?? analysisDefaults().likelyWorldsCap;
  const scoredCap = opts.scoredWorldsCap ?? analysisDefaults().scoredWorldsCap;
  const hiddenOf = (world) => game.hiddenPiecesOf?.(world, state, me) ?? [];
  const ranked = perWorldView ? (game.rankBeliefWorlds?.(state, me, likelyCap) ?? null) : null;

  // The walk order. `ranked.probs` is the posterior over the whole population, so
  // prefer heaviest-first (see weightOrder); a random permutation is the fallback
  // when there is no posterior to sort by (perfect information, where the
  // population is a single world anyway, or a game whose belief exposes none).
  const order = pop.exact
    ? (weightOrder(ranked?.probs, pop.total) ?? shuffledIndices(pop.total, rng))
    : null;

  // The most-likely boards, materialised once: they don't depend on the
  // engine at all, so the overlay can be on screen before the first rung lands.
  const likelyWorlds = [];
  if (ranked?.top?.length) {
    const idx = ranked.top.map(t => t.index);
    const worlds = game.enumerateWorlds(state, me, idx);
    for (let i = 0; i < worlds.length; i++) {
      likelyWorlds.push({ index: idx[i], prob: ranked.top[i].prob, hidden: hiddenOf(worlds[i]) });
    }
  }
  // index → { index, prob, hidden, cp[] }, for the worlds the engine actually
  // priced at the current rung. Rebuilt per rung like the cp aggregates, so
  // every cp in it was searched to the same depth and the "best world for this
  // move" ordering compares like with like.
  let scoredWorlds = new Map();
  let settledWorlds = new Map();

  // Mixing aggregate — accumulated across every batch of NEW worlds (see above),
  // each weighted by the batch's posterior MASS rather than its world count, so a
  // batch of near-impossible worlds doesn't get an equal vote in the ensemble.
  const probSum = new Map(); let probW = 0;
  // Eval aggregate — rebuilt from scratch at each rung of the ladder. `cpMass` is
  // the denominator of the weighted mean (Σ w, not a world count). `settledCp`
  // holds the deepest rung that actually produced numbers, so the eval column
  // never blanks out while a deeper rung is still being computed (or is being
  // abandoned because the engine can't reach it inside the budget).
  let cpSum = new Map(), cpMass = new Map();
  const settledCp = new Map();
  let settledDepth = 0;

  // Nothing hidden (population of exactly one world) makes the mixing degenerate
  // — purification commits to a single move, so every other move sits at 0% and
  // the probability column carries no ranking information at all. Rank by the
  // engine's evaluation there, which is what the old perfect-information branch
  // showed. Under a real belief cloud the mixing IS the answer to "what should I
  // play", so it stays primary and cp only breaks its (very common) ties.
  const rankByCp = pop.exact && pop.total === 1;
  const buildCandidates = () => legalActions
    .map(a => {
      const key = k(a);
      const mass = cpMass.get(key);
      return {
        move: a,
        // Same identity the per-world cp vectors are indexed by (`moves` in the
        // belief payload below), so the panel can line a candidate row up with
        // its column without re-deriving move equality from the action object.
        key,
        prob: probW ? (probSum.get(key) ?? 0) / probW : 0,
        cp: mass ? Math.round(cpSum.get(key) / mass) : (settledCp.get(key) ?? null),
      };
    })
    .sort(rankByCp
      ? (a, b) => ((b.cp ?? -Infinity) - (a.cp ?? -Infinity)) || (b.prob - a.prob)
      : (a, b) => (b.prob - a.prob) || ((b.cp ?? -Infinity) - (a.cp ?? -Infinity)));

  // The population itself, for the panel's world stepper — the union of the most
  // LIKELY boards (engine-free, so they are on screen immediately) and the
  // boards the engine has actually priced at the current rung (which carry a cp
  // per candidate move, so "which board makes THIS move look best" is
  // answerable). `moves` fixes the column order of every cp vector.
  const buildBeliefWorlds = () => {
    if (!perWorldView) return null;
    const scored = scoredWorlds.size ? scoredWorlds : settledWorlds;
    const byId = new Map();
    for (const w of likelyWorlds) byId.set(w.index, { id: String(w.index), prob: w.prob, hidden: w.hidden, cp: null });
    for (const [id, w] of scored) {
      const prior = byId.get(id);
      if (prior) prior.cp = w.cp;
      else byId.set(id, { id: String(id), prob: w.prob, hidden: w.hidden, cp: w.cp });
    }
    return {
      total, exact: pop.exact, depth: settledDepth || null,
      // A re-acquired superset, not the history-exact set: the panel must not
      // present these boards as certainties, and its weights are uniform rather
      // than a posterior (see ExactBelief.rankByLikelihood).
      approx: ranked?.approx ?? null,
      moves: legalActions.map(k),
      worlds: [...byId.values()],
    };
  };

  let batches = 0, last = null, covered = false, settledCovered = false;
  // The likely boards need nothing from the engine, so hand them over before
  // the first batch's CFR solve + leaf eval (seconds, on a large population)
  // rather than making the viewer stare at bare fog until then.
  if (likelyWorlds.length) opts.onProgress?.({ kind: 'belief', total, beliefWorlds: buildBeliefWorlds() });
  for (let depth = 1; depth <= maxDepth; depth++) {
    // A fresh rung: previous depths' evals are superseded, not blended into.
    cpSum = new Map(); cpMass = new Map();
    scoredWorlds = new Map();
    let cursor = 0, evaluated = 0, sweepCount = 0, rungCp = 0;
    covered = false;
    // Every batch of NEW worlds contributes its equilibrium once; the exact
    // population is only new on the first sweep.
    const foldProb = !pop.exact || depth === 1;

    while (!spent()) {
      // Next batch of belief worlds.
      let worlds;
      // Absolute population indices of `worlds`, so a world the engine prices
      // can be filed under the same id the likelihood ranking uses. Null in
      // the generative regime, which has no enumerable population to index into.
      let batchIdx = null;
      if (pop.exact) {
        if (cursor >= order.length) break; // (unreachable: `covered` breaks below first)
        const idx = order.slice(cursor, cursor + batchSize);
        cursor += idx.length;
        worlds = game.enumerateWorlds(state, me, idx);
        batchIdx = idx;
      } else {
        const w = game.sampleWorlds(state, me, batchSize, rng);
        worlds = (w && w.length) ? w : [state];
      }
      if (!worlds.length) break;

      // Mixing: one CFR equilibrium over this batch, folded in weighted by size.
      let mode = last?.mode ?? (state.gameSpecific.fogOfWar ? 'cfr' : 'minimax');
      if (foldProb) {
        const batchMixing = analysisDefaults().batchMixing;
        const r = await obscuroStrategy(state, legalActions, {
          worlds, color: me, rng, isCancelled,
          maxRounds: opts.maxRounds ?? batchMixing.maxRounds,
          expandPerRound: opts.expandPerRound ?? batchMixing.expandPerRound,
          cfrPerRound: opts.cfrPerRound ?? batchMixing.cfrPerRound,
        });
        if (isCancelled?.()) break; // moved on mid-solve — discard this partial batch
        mode = r.mode;
        // Batch MASS, not batch size: this batch's equilibrium is worth as much as
        // the posterior probability of the worlds it was solved over. Worlds with
        // no weight (generative fallback) default to 1 and recover the old
        // count-weighting exactly.
        let w = 0;
        for (const world of worlds) w += world.beliefWeight ?? 1;
        probW += w;
        for (let i = 0; i < r.rows.length; i++) {
          const key = k(r.rows[i]);
          probSum.set(key, (probSum.get(key) ?? 0) + w * (r.dist?.[i] ?? 0));
        }
      }

      // Eval: raw cp sums over the SAME batch at THIS rung's depth, so the
      // running mean stays exact and every world in it is equally deep.
      if (cpEval) {
        // Keep each world's own scores as they go by, up to the cap — the
        // aggregate below sums them away, but the per-world view needs them.
        const onWorld = !perWorldView ? undefined : (w, world, scores) => {
          if (scoredWorlds.size >= scoredCap) return;
          const id = batchIdx ? batchIdx[w] : `s${batches}:${w}`;
          if (scoredWorlds.has(id)) return;
          scoredWorlds.set(id, {
            prob: (ranked?.probs && typeof id === 'number') ? ranked.probs[id] : null,
            hidden: hiddenOf(world),
            cp: scores.map(s => Math.round(s)),
          });
        };
        // `wsum` defaults to `n` so an `opts.cpEval` override that predates the
        // weighting (returning just { sums, n }) still folds in, as an unweighted
        // mean over its own worlds.
        const { sums = null, n = 0, wsum = n } = (await cpEval(worlds, depth, onWorld)) ?? {};
        if (n > 0 && sums && wsum > 0) {
          rungCp += n;
          for (let i = 0; i < legalActions.length; i++) {
            const key = k(legalActions[i]);
            cpSum.set(key, (cpSum.get(key) ?? 0) + sums[i]);
            cpMass.set(key, (cpMass.get(key) ?? 0) + wsum);
          }
        }
      }
      if (spent()) break;
      // The engine could not finish a single world at this depth. Sweeping the
      // rest of the population at a rung it cannot reach would burn one full
      // per-call timeout per world for nothing, so abandon the rung now and let
      // the ladder top out on the last one that worked.
      if (cpEval && depth > 1 && rungCp === 0) break;

      batches++; evaluated += worlds.length; sweepCount++;
      covered = pop.exact ? cursor >= order.length : sweepCount >= sweepBatches;
      // Fully settled: the whole population, at the top of the ladder. Nothing
      // left to refine on either axis.
      const exhaustive = covered && (depth >= maxDepth || !cpEval) && pop.exact;
      const candidates = buildCandidates();
      last = { engine: 'obscuro', mode, candidates, depth, maxDepth, batches, evaluated, total, exhaustive };
      // The world list is bulky (one hidden-piece layout + one cp vector each)
      // and only meaningfully changes as new worlds get priced, so it rides
      // along on a fraction of the frames rather than every one. The panel keeps
      // the last one it saw.
      const belief = (batches === 1 || covered || batches % 8 === 0) ? buildBeliefWorlds() : null;
      opts.onProgress?.({ kind: 'batch', depth, maxDepth, batch: batches, evaluated, total, exhaustive, candidates,
        ...(belief ? { beliefWorlds: belief } : {}) });
      if (covered) break; // sweep complete — climb to the next rung
    }

    if (rungCp > 0) {
      // This rung produced real numbers: they become the floor the next rung's
      // partial results fall back to while it is still filling in.
      settledDepth = depth;
      settledCovered = covered;
      for (const [key, mass] of cpMass) settledCp.set(key, Math.round(cpSum.get(key) / mass));
      if (scoredWorlds.size) settledWorlds = scoredWorlds;
    } else if (cpEval && depth > 1) {
      break; // engine can't reach this depth inside the budget — the ladder tops out
    }
    if (!cpEval) break; // no engine: depth is meaningless, one sweep is the whole answer
    if (spent() || !covered) break; // cancelled, out of budget, or the sweep was cut short
  }

  if (last) {
    last.depth = settledDepth || last.depth;
    // The ladder may have topped out below maxDepth (a deeper rung simply cannot
    // be searched in the time available). If the population is nonetheless fully
    // covered and we stopped of our own accord, the answer IS settled.
    if (settledCovered && pop.exact && !spent()) last.exhaustive = true;
    last.beliefWorlds = buildBeliefWorlds();
    return last;
  }
  return {
    engine: 'obscuro', mode: state.gameSpecific.fogOfWar ? 'cfr' : 'minimax', candidates: [],
    // Even with no move ranking to show (cancelled before the first batch), the
    // likely-board list is already built and costs nothing to hand over.
    beliefWorlds: buildBeliefWorlds(),
  };
}
