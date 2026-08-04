import { getAllLegalMoves } from './moves.js';
import { applyMoveToBoard, isKingInCheck, fileIndex, rankOf, getVisibleSquares } from './board.js';
import { getBelief } from './belief.js';
import {
  stockfishBestAction, sfOptsForDifficulty, difficultyToNumber, multiPV,
  available as stockfishAvailable,
} from './stockfish.js';
import { toFEN, uciToAction } from './fen.js';
import { PIECE_VALUE, PST } from './pieceTables.js';

const FILES = 'abcdefgh';

function pstIndex(sq, color) {
  const fi = fileIndex(sq);
  const r  = rankOf(sq);
  return color === 'white' ? (8 - r) * 8 + fi : (r - 1) * 8 + fi;
}

function pieceScore(p) {
  const table = PST[p.type];
  return PIECE_VALUE[p.type] + (table ? table[pstIndex(p.position, p.ownerId)] : 0);
}

// Pawn structure: doubled/isolated penalties, passed pawn bonus
function pawnStructure(board, color) {
  const myFiles = new Array(8).fill(0);
  const myPawns = [];
  const oppByFile = Array.from({ length: 8 }, () => []);

  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.type !== 'pawn') continue;
    const fi = fileIndex(sq);
    const r  = rankOf(sq);
    if (p.ownerId === color) {
      myFiles[fi]++;
      myPawns.push([fi, r]);
    } else {
      oppByFile[fi].push(r);
    }
  }

  let score = 0;

  for (const cnt of myFiles) {
    if (cnt >= 2) score -= 20 * (cnt - 1);
  }

  for (let fi = 0; fi < 8; fi++) {
    if (!myFiles[fi]) continue;
    const hasNeighbor = (fi > 0 && myFiles[fi - 1] > 0) || (fi < 7 && myFiles[fi + 1] > 0);
    if (!hasNeighbor) score -= 15;
  }

  for (const [fi, r] of myPawns) {
    let passed = true;
    outer: for (let adj = Math.max(0, fi - 1); adj <= Math.min(7, fi + 1); adj++) {
      for (const oppR of oppByFile[adj]) {
        if ((color === 'white' && oppR > r) || (color === 'black' && oppR < r)) {
          passed = false;
          break outer;
        }
      }
    }
    if (passed) {
      const advance = color === 'white' ? r - 2 : 7 - r;
      score += 15 + advance * 15;
    }
  }

  return score;
}

function bishopPairBonus(board, color) {
  let cnt = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (p && p.ownerId === color && p.type === 'bishop') cnt++;
  }
  return cnt >= 2 ? 30 : 0;
}

function rookFileBonus(board, color) {
  const myPawnFiles  = new Set();
  const oppPawnFiles = new Set();
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.type !== 'pawn') continue;
    if (p.ownerId === color) myPawnFiles.add(fileIndex(sq));
    else                     oppPawnFiles.add(fileIndex(sq));
  }
  let score = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.ownerId !== color || p.type !== 'rook') continue;
    const fi = fileIndex(sq);
    if (!myPawnFiles.has(fi))  score += 10; // semi-open
    if (!oppPawnFiles.has(fi)) score += 10; // fully open
  }
  return score;
}

export function evaluate(board, aiColor) {
  const opp = aiColor === 'white' ? 'black' : 'white';
  let score = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p) continue;
    score += p.ownerId === aiColor ? pieceScore(p) : -pieceScore(p);
  }
  score += pawnStructure(board, aiColor)   - pawnStructure(board, opp);
  score += bishopPairBonus(board, aiColor) - bishopPairBonus(board, opp);
  score += rookFileBonus(board, aiColor)   - rookFileBonus(board, opp);
  return score;
}

// Propagate castling rights and en-passant across a search ply
export function advanceGs(gs, board, action, color) {
  let { castlingRights, halfMoveClock } = gs;
  let enPassantTarget = null;

  if (action.type === 'castle') {
    castlingRights = { ...castlingRights, [color]: { kingSide: false, queenSide: false } };
    halfMoveClock++;
  } else {
    const piece = board[action.from];
    halfMoveClock = (piece?.type === 'pawn' || action.isCapture) ? 0 : halfMoveClock + 1;

    if (action.isDoublePush) {
      const fi = fileIndex(action.from);
      const r  = rankOf(action.from);
      enPassantTarget = 'abcdefgh'[fi] + (r + (color === 'white' ? 1 : -1));
    }

    if (piece?.type === 'king') {
      castlingRights = { ...castlingRights, [color]: { kingSide: false, queenSide: false } };
    } else if (piece?.type === 'rook') {
      const rank = color === 'white' ? 1 : 8;
      const cr = { ...castlingRights[color] };
      if (action.from === 'a' + rank) cr.queenSide = false;
      if (action.from === 'h' + rank) cr.kingSide  = false;
      castlingRights = { ...castlingRights, [color]: cr };
    }

    if (action.isCapture) {
      const captured = board[action.to];
      if (captured?.type === 'rook') {
        const opp  = captured.ownerId;
        const rank = opp === 'white' ? 1 : 8;
        const cr   = { ...castlingRights[opp] };
        if (action.to === 'a' + rank) cr.queenSide = false;
        if (action.to === 'h' + rank) cr.kingSide  = false;
        castlingRights = { ...castlingRights, [opp]: cr };
      }
    }
  }

  return { ...gs, enPassantTarget, castlingRights, halfMoveClock, inCheck: false };
}

// MVV-LVA move ordering: captures first (big victim, small attacker), then promotions
function moveScore(m, board) {
  let s = 0;
  if (m.isCapture) {
    const victim   = board[m.to];
    const attacker = board[m.from];
    s += (PIECE_VALUE[victim?.type] ?? 0) - (PIECE_VALUE[attacker?.type] ?? 0) * 0.1;
  }
  if (m.payload?.promote === 'queen') s += 800;
  return s;
}

function orderMoves(moves, board) {
  return [...moves].sort((a, b) => moveScore(b, board) - moveScore(a, board));
}

// ---------------------------------------------------------------------------
// Transposition table (cleared at the start of each full-info root search)
// ---------------------------------------------------------------------------

let TT = new Map();

// Reset the shared transposition table. The full-info root search resets it
// itself; ObscuroAgent calls this once per move before reusing alphaBeta so the
// table never grows unbounded across a game.
export function clearTT() { TT = new Map(); }

function boardKey(board, color, gs) {
  const cr = gs.castlingRights;
  let key  = color + '|' +
    (cr.white.kingSide  ? 'K' : '') +
    (cr.white.queenSide ? 'Q' : '') +
    (cr.black.kingSide  ? 'k' : '') +
    (cr.black.queenSide ? 'q' : '') + '|' +
    (gs.enPassantTarget ?? '') + '|';
  for (let r = 8; r >= 1; r--) {
    for (let fi = 0; fi < 8; fi++) {
      const p = board[FILES[fi] + r];
      key += p ? p.ownerId[0] + p.type[0] : '.';
    }
  }
  return key;
}

// ---------------------------------------------------------------------------
// Quiescence search: resolve capture sequences at leaf nodes (full-info only)
// ---------------------------------------------------------------------------

function quiesce(board, gs, color, aiColor, alpha, beta) {
  const opp   = color === 'white' ? 'black' : 'white';
  const stand = evaluate(board, aiColor);

  if (color === aiColor) {
    if (stand >= beta) return stand;
    let best = stand;
    if (stand > alpha) alpha = stand;

    const captures = getAllLegalMoves(board, color, gs).filter(m => m.isCapture || m.payload?.promote);
    for (const m of orderMoves(captures, board)) {
      const score = quiesce(applyMoveToBoard(board, m), advanceGs(gs, board, m, color), opp, aiColor, alpha, beta);
      if (score > best) {
        best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
      }
    }
    return best;
  } else {
    if (stand <= alpha) return stand;
    let best = stand;
    if (stand < beta) beta = stand;

    const captures = getAllLegalMoves(board, color, gs).filter(m => m.isCapture || m.payload?.promote);
    for (const m of orderMoves(captures, board)) {
      const score = quiesce(applyMoveToBoard(board, m), advanceGs(gs, board, m, color), opp, aiColor, alpha, beta);
      if (score < best) {
        best = score;
        if (best < beta) beta = best;
        if (alpha >= beta) break;
      }
    }
    return best;
  }
}

// ---------------------------------------------------------------------------
// Search configurations — one per information model
// ---------------------------------------------------------------------------

// Per-particle search config: a particle is a fully-specified board, so we use
// ordinary legal moves with checkmate/stalemate terminals. (Under fog this is an
// approximation — the real game ends on king capture — but it gives strong,
// safe tactical evaluation within each hypothesised world.)
export const FULL_INFO_CFG = {
  getMoves:     getAllLegalMoves,
  evaluate:     evaluate,
  // Returns a terminal score if the position is decided, undefined otherwise.
  hardTerminal: null,
  noMoves(board, color, aiColor, depth) {
    return isKingInCheck(board, color)
      ? (color === aiColor ? -19000 - depth : 19000 + depth)
      : 0; // stalemate
  },
  useTT: true,
};

// ---------------------------------------------------------------------------
// Difficulty configuration
//
// There is a single search (see `search` below): it always evaluates moves
// against a cloud of belief particles. With perfect information that cloud is
// exactly one particle — the true board — so the same code plays ordinary
// chess; with fog of war it is several sampled worlds.
//
//   depth      total plies searched per particle when there is ONE particle
//   noise      random tie-break jitter (centipawns), for weaker difficulties
//   useQuiesce resolve capture sequences at leaves
//   fog        overrides used under fog, where many particles force a shallower
//              per-particle depth and a candidate-move prefilter to bound cost
// ---------------------------------------------------------------------------

// Difficulty is a 0–100 dial (see stockfish.js). Knobs are derived continuously:
// search depth and particle count rise with difficulty, while the random "noise"
// added to scores (which makes weak play blunder) fades out by the mid-range.
function chessConfigForDifficulty(difficulty) {
  const t = difficultyToNumber(difficulty) / 100;
  const lerp = (a, b) => Math.round(a + (b - a) * t);
  return {
    depth: lerp(2, 5),
    noise: Math.round(Math.max(0, 250 * (1 - t / 0.5))), // 250 cp at 0 → 0 by 50
    useQuiesce: t >= 0.2,
    fog: { particles: lerp(4, 18), depth: t < 0.2 ? 1 : 2, topK: lerp(6, 10) },
  };
}

// Weight on tail risk when aggregating a move's score across particles. The
// score is a blend of the mean outcome and the average of the worst-30% of
// particles, so a move that hangs a piece in some plausible world is penalised
// without one paranoid particle vetoing every move.
const PESSIMISM = 0.5;
// Small bonus per square the move would reveal — encourages scouting to shrink
// future uncertainty. Kept well below a pawn (100) so it only breaks ties.
const INFO_WEIGHT = 2;
// Per-particle score clamp (centipawns) under fog. Big enough that losing a
// queen still dominates losing a pawn, small enough that an imagined checkmate
// from phantom hidden pieces can't swamp a concrete material decision (which was
// making the AI keep a hanging queen home rather than expose its king to ghosts).
const FOG_CLAMP = 1200;

// ---------------------------------------------------------------------------
// Minimax with alpha-beta pruning, used to evaluate a move inside a single
// fully-specified particle. cfg controls evaluator / terminal / TT usage.
// ---------------------------------------------------------------------------

export function alphaBeta(board, gs, color, aiColor, depth, alpha, beta, cfg, useQuiesce) {
  // Hard terminal (e.g., a king was captured under fog rules)
  if (cfg.hardTerminal) {
    const t = cfg.hardTerminal(board, aiColor);
    if (t !== undefined) return t;
  }

  // Transposition table lookup
  let origAlpha = alpha, origBeta = beta, key;
  if (cfg.useTT) {
    key = boardKey(board, color, gs);
    const cached = TT.get(key);
    if (cached && cached.depth >= depth) {
      if (cached.flag === 'EXACT') return cached.score;
      if (cached.flag === 'LOWER' && cached.score >= beta)  return cached.score;
      if (cached.flag === 'UPPER' && cached.score <= alpha) return cached.score;
    }
  }

  // Leaf node
  if (depth === 0) {
    const score = useQuiesce
      ? quiesce(board, gs, color, aiColor, alpha, beta)
      : cfg.evaluate(board, aiColor);
    if (cfg.useTT) TT.set(key, { score, depth: 0, flag: 'EXACT' });
    return score;
  }

  const opp   = color === 'white' ? 'black' : 'white';
  const moves = cfg.getMoves(board, color, gs);

  if (moves.length === 0) return cfg.noMoves(board, color, aiColor, depth);

  const sorted    = orderMoves(moves, board);
  let bestScore   = color === aiColor ? -Infinity : Infinity;

  if (color === aiColor) {
    for (const m of sorted) {
      const score = alphaBeta(applyMoveToBoard(board, m), advanceGs(gs, board, m, color), opp, aiColor, depth - 1, alpha, beta, cfg, useQuiesce);
      if (score > bestScore) bestScore = score;
      if (bestScore > alpha) alpha = bestScore;
      if (alpha >= beta) break;
    }
  } else {
    for (const m of sorted) {
      const score = alphaBeta(applyMoveToBoard(board, m), advanceGs(gs, board, m, color), opp, aiColor, depth - 1, alpha, beta, cfg, useQuiesce);
      if (score < bestScore) bestScore = score;
      if (bestScore < beta) beta = bestScore;
      if (alpha >= beta) break;
    }
  }

  if (cfg.useTT) {
    const flag = bestScore <= origAlpha ? 'UPPER' : bestScore >= origBeta ? 'LOWER' : 'EXACT';
    TT.set(key, { score: bestScore, depth, flag });
  }
  return bestScore;
}

// ---------------------------------------------------------------------------
// Full-information root search (no fog).
// ---------------------------------------------------------------------------

// Score one root move within one particle: play it, then run lookahead (with
// quiescence) from the opponent's reply. `depth` is total plies including the
// root move.
export function scoreMoveInParticle(particleBoard, gs, aiColor, move, depth, useQuiesce) {
  const opp = aiColor === 'white' ? 'black' : 'white';
  const nb  = applyMoveToBoard(particleBoard, move);
  const ngs = advanceGs(gs, particleBoard, move, aiColor);
  return alphaBeta(nb, ngs, opp, aiColor, depth - 1, -Infinity, Infinity, FULL_INFO_CFG, useQuiesce);
}

// Blend the mean outcome with the average of the worst-30% of particles. With a
// single particle this is just that particle's score.
function aggregateScores(scores) {
  if (scores.length === 0) return 0;
  if (scores.length === 1) return scores[0];
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const sorted = [...scores].sort((a, b) => a - b);
  const k = Math.max(1, Math.ceil(sorted.length * 0.3));
  let tail = 0;
  for (let i = 0; i < k; i++) tail += sorted[i];
  tail /= k;
  return PESSIMISM * tail + (1 - PESSIMISM) * mean;
}

// ---------------------------------------------------------------------------
// The one and only search.
// Evaluate each candidate move against a cloud of belief particles — each a
// fully-specified guess at where every piece is — and aggregate the outcomes
// pessimistically, so a move that hangs material in some plausible world is
// avoided even when the threatening piece is currently invisible. Under perfect
// information the cloud is a single particle (the true board) and this reduces
// to ordinary alpha-beta search.
// ---------------------------------------------------------------------------

// Scores every candidate move (after the same cheap topK prefilter `search` uses)
// against the belief-particle cloud, without picking one — the read-only half of
// `search`, reused both by move selection and by `analyze` (which must never
// mutate anything, only rank). Returns candidates sorted best-first, each
// `{ move, score }` (no noise applied — noise is a selection-time blunder knob,
// not part of a move's actual evaluation).
function scoreAllCandidates(board, gs, color, legalActions, particles, opts) {
  TT = new Map();
  const { depth, useQuiesce, topK, infoWeight, clamp } = opts;
  // Across many uncertain worlds we clamp each particle's score so that a single
  // fantasy line (e.g. a phantom checkmate from imagined hidden pieces) cannot
  // dominate the aggregate and scare the AI out of saving real material.
  const clip = clamp ? (s => (s > clamp ? clamp : s < -clamp ? -clamp : s)) : (s => s);

  // Under fog (many particles) a candidate move's full evaluation is costly, so
  // first rank moves cheaply against one particle and keep only the best topK.
  let candidates = orderMoves(legalActions, board);
  if (topK && candidates.length > topK) {
    const rep = particles[0];
    candidates = candidates
      .map(m => ({ m, s: scoreMoveInParticle(rep, gs, color, m, 1, false) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, topK)
      .map(x => x.m);
  }

  const scored = candidates.map(m => {
    const scores = particles.map(p => clip(scoreMoveInParticle(p, gs, color, m, depth, useQuiesce)));
    let score = aggregateScores(scores);
    if (infoWeight) score += getVisibleSquares(applyMoveToBoard(board, m), color).size * infoWeight;
    return { move: m, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

function search(board, gs, color, legalActions, particles, opts) {
  const scored = scoreAllCandidates(board, gs, color, legalActions, particles, opts);
  if (!scored.length) return legalActions[0];
  const { noise } = opts;
  if (!(noise > 0)) return scored[0].move;

  let bestScore = -Infinity, bestMove = scored[0].move;
  for (const { move, score } of scored) {
    const jittered = score + (Math.random() * noise * 2 - noise);
    if (jittered > bestScore) { bestScore = jittered; bestMove = move; }
  }
  return bestMove;
}

// The per-game belief store (getBelief) now lives in belief.js and is shared
// with ObscuroAgent.

// ---------------------------------------------------------------------------
// Agent export
// ---------------------------------------------------------------------------

export const ChessAgent = {
  id: 'chess-ai',
  name: 'Chess AI',

  async chooseAction(state, legalActions) {
    if (legalActions.length === 0) return null;
    if (legalActions.length === 1) return legalActions[0];

    // Yield to the event loop before blocking synchronous search
    await new Promise(r => setImmediate(r));

    const color = state.activePlayers[0];
    const { board, gameSpecific } = state;
    const cfg = chessConfigForDifficulty(gameSpecific.difficulty);

    // Build the belief particle cloud. Without fog the AI sees everything, so
    // the cloud is exactly one particle — the true board. With fog it is several
    // sampled worlds and the chosen move is committed back for next turn.
    if (!gameSpecific.fogOfWar) {
      // Prefer the vendored Stockfish for full-information play; fall back to the
      // built-in alpha-beta search if the engine isn't available.
      const sf = await stockfishBestAction(state, legalActions, sfOptsForDifficulty(gameSpecific.difficulty));
      if (sf) return sf;
      return search(board, gameSpecific, color, legalActions, [board], {
        depth: cfg.depth, useQuiesce: cfg.useQuiesce, noise: cfg.noise, topK: 0, infoWeight: 0,
      });
    }

    const belief = getBelief(state, color);
    belief.beginTurn(board, state.turnNumber ?? null);
    let particles = belief.sample(board, cfg.fog.particles);
    if (particles.length === 0) particles = [board]; // fallback: trust the visible board
    const action = search(board, gameSpecific, color, legalActions, particles, {
      depth: cfg.fog.depth, useQuiesce: cfg.useQuiesce, noise: cfg.noise, topK: cfg.fog.topK, infoWeight: INFO_WEIGHT, clamp: FOG_CLAMP,
    });
    belief.commitOurMove(action, board);
    return action;
  },

  // Read-only position analysis: ranks every legal move without committing one,
  // for the UI's "suggest a move" panel. Must never mutate the shared belief
  // store — a viewer opening the analysis panel must not change how the real
  // AI plays its next turn — so under fog it resyncs the belief to the current
  // board (`beginTurn`, which just re-derives from ground truth, same as
  // `chooseAction` does before sampling) but deliberately skips
  // `commitOurMove`, the step that records a move as actually played.
  async analyze(state, legalActions, opts = {}) {
    if (!legalActions?.length) return { engine: 'chess-ai', mode: 'none', candidates: [] };

    // Defaults to whoever's actually to move, but a caller may override — e.g.
    // the analysis API always passes the requesting viewer's own colour under
    // fog, so "what's good for my side" stays answerable even when it isn't
    // literally their turn yet (battle-simulator's analysis endpoint does this).
    const color = opts.color ?? state.activePlayers[0];
    const { board, gameSpecific } = state;
    // Analysis is not gameplay: it must never be hobbled by whatever power level
    // this particular session happens to be playing at (a weak-difficulty game
    // still deserves real suggestions) — always search at the strongest tier the
    // depth/particle dial goes to, ignoring gameSpecific.difficulty entirely.
    const cfg = chessConfigForDifficulty(100);

    if (!gameSpecific.fogOfWar) {
      if (await stockfishAvailable()) {
        try {
          const fen = toFEN(board, gameSpecific, color === 'white' ? 'w' : 'b', state.turnNumber ?? 1);
          const maxDepth = 14;
          const toCandidates = (raw) => raw
            .map(({ move, cp }) => { const a = uciToAction(move, legalActions); return a ? { move: a, cp } : null; })
            .filter(Boolean);
          // Live "depth N/14" progress (lichess-style) — purely a side channel,
          // fired once per completed iterative-deepening depth; see multiPV.
          const onInfo = opts.onProgress
            ? ({ depth, candidates }) => opts.onProgress({ kind: 'depth', depth, maxDepth, candidates: toCandidates(candidates) })
            : undefined;
          const pv = await multiPV(fen, { multipv: Math.min(legalActions.length, 8), depth: maxDepth, onInfo });
          if (pv && pv.length) {
            const candidates = toCandidates(pv);
            if (candidates.length) return { engine: 'chess-ai', mode: 'stockfish', candidates };
          }
        } catch { /* fall through to the built-in evaluator */ }
      }
      const scored = scoreAllCandidates(board, gameSpecific, color, legalActions, [board], {
        depth: cfg.depth, useQuiesce: cfg.useQuiesce, topK: 0, infoWeight: 0,
      });
      return { engine: 'chess-ai', mode: 'minimax', candidates: scored.map(({ move, score }) => ({ move, cp: score })) };
    }

    const belief = getBelief(state, color);
    belief.beginTurn(board, state.turnNumber ?? null);
    let particles = belief.sample(board, opts.particles ?? cfg.fog.particles);
    if (particles.length === 0) particles = [board];
    const scored = scoreAllCandidates(board, gameSpecific, color, legalActions, particles, {
      depth: cfg.fog.depth, useQuiesce: cfg.useQuiesce, topK: cfg.fog.topK, infoWeight: INFO_WEIGHT, clamp: FOG_CLAMP,
    });
    return { engine: 'chess-ai', mode: 'fog-minimax', candidates: scored.map(({ move, score }) => ({ move, cp: score })) };
  },
};
