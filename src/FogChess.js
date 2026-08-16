// ---------------------------------------------------------------------------
// Fog-of-war chess as a plain GameDefinition — the game this package's agents are
// built for, and the one its tests play.
//
// It is the game interface of obscuro-ai (vendor/obscuro/docs/GAME-INTERFACE.md)
// implemented for fog chess: the rules half (createInitialState, getLegalActions,
// applyActions, getResult), the observation half (getVisibleState, identityOf),
// and the belief half that makes the search possible under fog — sampleWorlds
// draws from the exact position set P (src/exactBelief.js), falling back to the
// heuristic particle tracker (src/belief.js) once exact tracking is lost, with
// beliefPopulation/enumerateWorlds/rankBeliefWorlds exposing that same population
// for exhaustive walks and for showing a human what the AI thinks is out there.
//
// THE RULES, which are not standard chess:
//   • Each side sees only the squares its own pieces can reach (src/board.js's
//     getVisibleSquares). Everything else is dark — including, deliberately, the
//     square in front of a blocked pawn.
//   • There is no check, checkmate or stalemate. You win by CAPTURING the enemy
//     king; moving into check is legal, it just loses. Moves are therefore
//     pseudo-legal, and a player's action set depends only on its own
//     observation — which is what lets every position in one information set
//     share one action set (see the note on getLegalActions).
//   • A move can FAIL: a pawn push onto a square that turned out to be occupied
//     leaves the pawn where it was and ends the turn.
//   • 100 half-moves without a capture or pawn move is a draw.
//
// This file is a self-contained implementation of the rules and the belief
// interface, and nothing else — no rendering, no replay, no fog markers, no
// difficulty menu. An embedder whose engine wants those writes its own
// definition against the same src/ modules and keeps this one for reference;
// battle-simulator's games/chess/ChessGame.js is exactly that. When it does, the
// two have to agree on the rules, and it is that embedder's own integration
// tests, not these, that will notice if they stop agreeing.
// ---------------------------------------------------------------------------

import { isKingInCheck, getVisibleSquares } from './board.js';
import { getAllLegalMoves, getAllFogMoves } from './moves.js';
import { evaluate } from './ChessAgent.js';
import { getBelief, impossiblePlacement } from './belief.js';
import { getExactBelief, getBeliefReachWeighting, getBeliefSampleAlpha } from './exactBelief.js';
import { DEFAULT_DIFFICULTY } from './stockfish.js';
import { param } from './config.js';

// ---------------------------------------------------------------------------
// Initial board setup
// ---------------------------------------------------------------------------

function makeUnit(id, ownerId, type, position) {
  return { id, ownerId, type, position, alive: true };
}

export function initialBoard() {
  const board = {};
  const backRank = (color) => color === 'white' ? 1 : 8;
  const pawnRank = (color) => color === 'white' ? 2 : 7;
  const prefix = (color) => color === 'white' ? 'w' : 'b';

  for (const color of ['white', 'black']) {
    const br = backRank(color);
    const pr = pawnRank(color);
    const p = prefix(color);

    const backPieces = [
      ['R', 'rook',   'a'],
      ['N', 'knight', 'b'],
      ['B', 'bishop', 'c'],
      ['Q', 'queen',  'd'],
      ['K', 'king',   'e'],
      ['B', 'bishop', 'f', '2'],
      ['N', 'knight', 'g', '2'],
      ['R', 'rook',   'h', '2'],
    ];
    for (const [sym, type, file, suffix = ''] of backPieces) {
      const sq = file + br;
      board[sq] = makeUnit(p + sym + suffix, color, type, sq);
    }

    for (const file of 'abcdefgh') {
      const sq = file + pr;
      board[sq] = makeUnit(p + 'P' + file, color, 'pawn', sq);
    }
  }
  return board;
}

export function boardToUnits(board) {
  return Object.values(board).filter(Boolean);
}

// ---------------------------------------------------------------------------
// applyActions helpers
// ---------------------------------------------------------------------------

function updateCastlingRights(rights, unit, square) {
  let { white, black } = rights;
  if (unit.type === 'king') {
    if (unit.ownerId === 'white') white = { kingSide: false, queenSide: false };
    else black = { kingSide: false, queenSide: false };
  }
  if (unit.type === 'rook') {
    if (square === 'a1') white = { ...white, queenSide: false };
    if (square === 'h1') white = { ...white, kingSide: false };
    if (square === 'a8') black = { ...black, queenSide: false };
    if (square === 'h8') black = { ...black, kingSide: false };
  }
  return { white, black };
}

// ---------------------------------------------------------------------------
// GameDefinition
// ---------------------------------------------------------------------------

export const FogChess = {
  name: 'Fog Chess',

  createInitialState(players = [{ id: 'white' }, { id: 'black' }], config = {}) {
    const board = config.board ?? initialBoard();
    return {
      gameName: 'Chess',
      turnNumber: config.turnNumber ?? 1,
      activePlayers: [config.toMove ?? 'white'],
      currentPhase: 'action',
      players,
      board,
      units: boardToUnits(board),
      lastActions: null,
      gameSpecific: {
        enPassantTarget: config.enPassantTarget ?? null,
        castlingRights: config.castlingRights ?? {
          white: { kingSide: true, queenSide: true },
          black: { kingSide: true, queenSide: true },
        },
        halfMoveClock: config.halfMoveClock ?? 0,
        inCheck: false,
        fogOfWar: config.fogOfWar ?? true,
        // Exactly one of difficulty (power 0–100) / aiTimeMs (per-move ms) is
        // active; if a time limit is given it wins and difficulty is left null.
        aiTimeMs:   typeof config.aiTimeMs === 'number' ? config.aiTimeMs : null,
        difficulty: typeof config.aiTimeMs === 'number' ? null
          : (config.difficulty ?? param('chess.DEFAULT_DIFFICULTY', DEFAULT_DIFFICULTY)),
        // Optional per-SESSION parameter overrides, in the agent's own opts
        // vocabulary ({ particles, timeBudgetMs, maxRounds, ... }). It rides the
        // state like difficulty does, so a host running several games at once
        // can give each its own knobs without building an agent per game — the
        // production agent is a shared singleton. Outranks a settings file,
        // outranked by an agent's constructor opts. See docs/SETTINGS.md.
        ...(config.obscuro ? { obscuro: config.obscuro } : {}),
      },
    };
  },

  getLegalActions(state, playerId) {
    return state.gameSpecific.fogOfWar
      ? getAllFogMoves(state.board, playerId, state.gameSpecific)
      : getAllLegalMoves(state.board, playerId, state.gameSpecific);
  },

  // NOTE: there is deliberately NO getSearchLegalActions here. The search tree
  // must use the REAL fog action set (pseudo-legal: moving into check is legal,
  // it just loses), because in FoW chess a player's legal-move set is fully
  // determined by its own observation — so every node in one infoset shares one
  // action set, which the Obscuro tree requires. Modelling the tree with
  // CHECK-FILTERED moves instead breaks the invariant: in the belief worlds
  // where OUR move self-checks (exactly the dangerous ones!) the move vanishes
  // from that world's action set and is scored as a neutral pass, so a real
  // king-hang gets priced at material value. Self-check is instead handled by
  // the VALUE model: such children evaluate to −SEARCH_WIN for the mover
  // (src/ObscuroAgent.js), new infosets seed to the best child, and CFR then
  // keeps suicide moves out of both players' strategies.

  applyActions(state, playerActions) {
    const { playerId, action } = playerActions[0]; // chess: always 1 active player
    const opponent = playerId === 'white' ? 'black' : 'white';
    let board = { ...state.board };
    let { castlingRights, halfMoveClock } = state.gameSpecific;
    let enPassantTarget = null; // cleared by default

    if (action.type === 'castle') {
      const king = board[action.from];
      const rook = board[action.rookFrom];
      board[action.from] = undefined;
      board[action.to] = { ...king, position: action.to };
      board[action.rookFrom] = undefined;
      board[action.rookTo] = { ...rook, position: action.rookTo };
      castlingRights = updateCastlingRights(castlingRights, king, action.from);
      halfMoveClock++;
    } else {
      // Regular move / capture / en passant / promotion
      const piece = board[action.from];
      // The action set is generated once against the mover's OWN (fog-limited)
      // view of the board and then replayed against many hidden-state worlds
      // during search/analysis (see the "NO getSearchLegalActions" note above).
      // Those can disagree: a pawn push whose target square looked empty to the
      // mover (a blocked push square stays deliberately hidden — see
      // getVisibleSquares in board.js) can turn out to be occupied in a
      // specific sampled world. Blindly relocating the piece there would
      // fabricate an illegal capture (pawns can't capture by pushing straight)
      // and hand the leaf evaluator a position that could never occur — inflate
      // its score enough and it gets suggested as the best move. Treat that
      // case as the real move failing instead: the piece stays put and nothing
      // else about the position changes.
      let blockedHere = !action.isCapture && !action.isEnPassant && board[action.to] != null;
      // A double push can ALSO fail on the square it jumps over (the pawn can't
      // see past a blocker there either, per getVisibleSquares) even when the
      // final square is genuinely empty in this world — it never gets there.
      let midSquare = null;
      if (!blockedHere && action.isDoublePush) {
        const fi = action.from.charCodeAt(0) - 'a'.charCodeAt(0);
        const fromRank = parseInt(action.from[1], 10);
        const dir = playerId === 'white' ? 1 : -1;
        midSquare = String.fromCharCode('a'.charCodeAt(0) + fi) + (fromRank + dir);
        if (board[midSquare] != null) blockedHere = true;
      }

      if (!blockedHere) {
        board[action.from] = undefined;
        const newType = action.payload?.promote ?? piece.type;
        board[action.to] = { ...piece, position: action.to, type: newType };

        if (action.isEnPassant && action.capturedSquare) {
          board[action.capturedSquare] = undefined;
        }

        halfMoveClock = (piece.type === 'pawn' || action.isCapture) ? 0 : halfMoveClock + 1;
        if (action.isDoublePush) enPassantTarget = midSquare;

        castlingRights = updateCastlingRights(castlingRights, piece, action.from);
        // Also revoke if a rook is captured on its starting square
        if (action.isCapture && action.to) {
          const captured = state.board[action.to];
          if (captured?.type === 'rook') {
            castlingRights = updateCastlingRights(castlingRights, captured, action.to);
          }
        }
      }
    }

    return {
      ...state,
      board,
      units: boardToUnits(board),
      activePlayers: [opponent],
      turnNumber: playerId === 'black' ? state.turnNumber + 1 : state.turnNumber,
      lastActions: playerActions,
      gameSpecific: {
        ...state.gameSpecific,
        enPassantTarget, castlingRights, halfMoveClock,
        inCheck: isKingInCheck(board, opponent),
      },
    };
  },

  getResult(state) {
    if (state.gameSpecific.fogOfWar) {
      // Win by capturing the king; no checkmate or stalemate
      const hasWhiteKing = state.units.some(u => u.ownerId === 'white' && u.type === 'king');
      const hasBlackKing = state.units.some(u => u.ownerId === 'black' && u.type === 'king');
      if (!hasWhiteKing) return { outcome: 'win', winnerId: 'black', reason: 'king-captured' };
      if (!hasBlackKing) return { outcome: 'win', winnerId: 'white', reason: 'king-captured' };
      if (state.gameSpecific.halfMoveClock >= 100) {
        return { outcome: 'draw', winnerId: null, reason: 'fifty-move-rule' };
      }
      return null;
    }
    const [activePlayer] = state.activePlayers;
    const legal = getAllLegalMoves(state.board, activePlayer, state.gameSpecific);
    if (legal.length > 0) {
      if (state.gameSpecific.halfMoveClock >= 100) {
        return { outcome: 'draw', winnerId: null, reason: 'fifty-move-rule' };
      }
      return null;
    }
    if (state.gameSpecific.inCheck) {
      return { outcome: 'win', winnerId: activePlayer === 'white' ? 'black' : 'white', reason: 'checkmate' };
    }
    return { outcome: 'draw', winnerId: null, reason: 'stalemate' };
  },

  getVisibleState(state, playerId) {
    if (!state.gameSpecific.fogOfWar) return state;
    // A player NEVER sees the board through the fog — every requester (human or
    // AI) gets only the squares its own pieces can see.
    const visible = getVisibleSquares(state.board, playerId);
    const filteredBoard = { ...state.board };
    for (const sq of Object.keys(filteredBoard)) {
      const piece = filteredBoard[sq];
      if (piece && piece.ownerId !== playerId && !visible.has(sq)) {
        filteredBoard[sq] = undefined;
      }
    }
    return {
      ...state,
      board: filteredBoard,
      units: boardToUnits(filteredBoard),
      // The opponent's last move is exactly the thing fog is hiding, so it cannot
      // ride along in the state we hand out — spelling out the from/to square of a
      // move whose piece we just stripped from the board would undo the filtering
      // above.
      lastActions: state.lastActions?.filter(pa => pa.playerId === playerId) ?? null,
      // The authoritative set of squares this player can see, computed on the FULL
      // board so hidden enemies still block and occupy. A consumer cannot re-derive
      // visibility from the filtered board, where a stripped piece (e.g. a hidden
      // pawn on e5 blocking our e4 pawn's push) would wrongly look like empty +
      // seen. The exact-belief tracker reads this field directly.
      visibleSquares: [...visible],
      viewerId: playerId,
    };
  },

  // --- Imperfect-information interface (drives the generic ObscuroAgent) -----

  // Which part of the observation above makes two positions the same information
  // set. The rest must NOT key: `units` is derived from the board, and
  // `visibleSquares`/`viewerId` are about presentation, so keying on them would
  // split infosets that are one position.
  //
  // Squares map to {ownerId, type, id}, so Obscuro drops the piece id: a belief
  // sampler invents ids the engine would never produce, and without that a
  // sampled world would never dedupe against the identical carried position.
  identityOf(state) {
    return state.board;
  },

  // Heuristic leaf value of a position to `playerId` (white/black), reusing the
  // chess agent's material + piece-square evaluation.
  evaluateState(state, playerId) {
    return evaluate(state.board, playerId);
  },

  // Canonical identity of a move, so the same opponent reply across different
  // sampled worlds maps to the same payoff-matrix column. from+to(+promotion) is
  // unique per move, including the king's two-square castling hop.
  actionKey(action) {
    const promo = action.payload?.promote ? '=' + action.payload.promote[0] : '';
    return (action.from ?? '') + (action.to ?? '') + promo;
  },

  // Belief sampler. Preferred source is the EXACT position set P (the paper's
  // belief: every position consistent with the full observation history,
  // src/exactBelief.js), sampled IN PROPORTION TO EACH POSITION'S POSTERIOR
  // WEIGHT — while it holds, the belief is perfect, and |P| = 1 means we
  // literally know the board. Note the asymmetry with enumerateWorlds below: a
  // sampled world needs no `beliefWeight`, because the weight is already in the
  // draw, and applying it again would count the posterior twice. If exact
  // tracking has given up (attached mid-game, or P outgrew its cap), we fall back
  // to the heuristic particle tracker (src/belief.js), which is kept in lockstep
  // every turn so the handover is seamless. With fog off there is nothing hidden,
  // so we return [] and the agent treats the position as the single known world.
  sampleWorlds(observation, playerId, n, rng = Math.random) {
    if (!observation.gameSpecific.fogOfWar) return [];
    // OBSCURO_VALIDATE_WORLDS=1 reports any imagined board that could never
    // occur, tagged with the path that produced it. Off by default (it walks
    // every world), on when hunting the next one of these: an illegal board makes
    // Stockfish return nothing, and the leaf evaluator silently guesses instead.
    const validate = (worlds, source) => {
      if (!process.env?.OBSCURO_VALIDATE_WORLDS) return worlds;
      for (const w of worlds) {
        const why = impossiblePlacement(w.board);
        if (why) console.error(`[impossible world via ${source}] ${why}`);
      }
      return worlds;
    };
    // turnNumber keys the updates so re-sampling within one decision (e.g. the
    // king-safety guard) can't advance either belief an extra phantom ply.
    const turnKey = observation.turnNumber ?? null;
    const exact = getExactBelief(observation, playerId);
    exact.beginTurn(observation, turnKey);
    const belief = getBelief(observation, playerId);
    belief.beginTurn(observation.board, turnKey);
    // If exact tracking was lost, information may since have collapsed enough
    // (few hidden pieces, small possible-sets) to re-enumerate P from the
    // heuristic belief — a tight superset, still far better than particles.
    if (!exact.exact) exact.tryReacquire(observation, belief, turnKey);
    if (exact.exact) {
      // Draw indices rather than positions so the POSTERIOR WEIGHT of each pick
      // can ride along. It has to: the search weights every world's
      // counterfactual value by its root reach (vendor/obscuro/src/infoset.js —
      // `cfrDescend(w.node, me, 1, w.prob)`), and with a uniform draw that reach
      // was 1/N for every world, i.e. the AI evaluated positions under a UNIFORM
      // belief over P and the posterior reached play through nothing at all.
      //
      // The correction is importance sampling. Drawing ∝ wᵅ and weighting by
      // w¹⁻ᵅ estimates the w-weighted mean for any α, which keeps both ends
      // honest: at the shipped α=1 the weight is already in the draw (uniform
      // reach), and at α=0 the draw is uniform and the reach carries the whole
      // posterior.
      const idx = exact.sampleIndices(n, rng);
      const picks = idx && exact.positionsAt(idx);
      if (picks && picks.length) {
        const source = exact.approx ? 'exact(reacquired)' : 'exact';
        // Fall back to the CONFIGURED α, not to a literal. A tracker always sets
        // `_alpha` at construction so this should never fire, but the literal 0
        // that used to be here silently meant "paper behaviour" the moment the
        // default stopped being 0 — a stale fallback that agrees with the default
        // is invisible until the default moves.
        const alpha = exact.sampleAlpha ?? getBeliefSampleAlpha();
        const beta = getBeliefReachWeighting(playerId);
        return validate(picks.map(pos => ({
          ...observation,
          board: pos.board,
          units: boardToUnits(pos.board),
          // The importance weight, NOT the raw posterior. runObscuroSearch
          // normalises across the sample, so only relative values matter.
          ...(beta > 0 && alpha < 1 ? { beliefWeight: Math.pow(pos.w ?? 0, beta * (1 - alpha)) } : {}),
          // Position-specific rights/en-passant so in-tree move generation for
          // BOTH sides is exact per world (the heuristic path can't know these).
          gameSpecific: {
            ...observation.gameSpecific,
            castlingRights: pos.cr,
            enPassantTarget: pos.ep,
          },
        })), source);
      }
    }
    const boards = belief.sample(observation.board, n, rng);
    return validate(boards.map(board => ({
      ...observation,
      board,
      units: boardToUnits(board),
    })), 'heuristic-particles');
  },

  // --- Whole-population enumeration (drives the analysis panel's batched,
  // eventually-exhaustive belief walk — see ObscuroAgent.analyzeObscuroProgressive).
  //
  // sampleWorlds draws n DISTINCT worlds at random (Efraimidis–Spirakis in the
  // exact path, rejection-with-a-key-set in the heuristic one — neither can
  // return a duplicate, matching the paper's "sampled at random without
  // replacement from the set of possible states"); these two let a caller
  // instead walk the ENTIRE materialized belief set P once, in batches, in a
  // fixed order, and know when it has been covered exhaustively. Only the exact
  // tracker has a finite materialized population — belief.js is a generative
  // sampler with no enumerable set — so this reports exact:false in the fallback
  // case and the caller keeps sampling there. Prepares the belief trackers for
  // this turn exactly like sampleWorlds (idempotent via turnKey), so the two are
  // interchangeable within one decision.
  beliefPopulation(observation, playerId) {
    // Perfect information is not "no belief" — it is a belief population of
    // EXACTLY ONE world: nothing is hidden, so precisely one position is
    // consistent with the observation, namely the observation itself. Saying so
    // (rather than `{ exact: false, total: 0 }`) is what lets the analysis walk
    // run one code path for both regimes instead of forking on fogOfWar.
    if (!observation.gameSpecific.fogOfWar) return { exact: true, total: 1 };
    const turnKey = observation.turnNumber ?? null;
    const exact = getExactBelief(observation, playerId);
    exact.beginTurn(observation, turnKey);
    const belief = getBelief(observation, playerId);
    belief.beginTurn(observation.board, turnKey);
    if (!exact.exact) exact.tryReacquire(observation, belief, turnKey);
    return (exact.exact && exact.positions?.length)
      ? { exact: true, total: exact.positions.length }
      : { exact: false, total: null };
  },

  // Map absolute indices into the exact set P → observation-shaped belief worlds,
  // the SAME shape sampleWorlds produces (position-specific rights/en-passant per
  // world, so in-tree move generation stays exact), so the search treats an
  // enumerated world identically to a sampled one. Requires a beliefPopulation
  // call earlier this turn to have established P; out-of-range indices are
  // skipped, and the result is empty when exact tracking isn't active.
  enumerateWorlds(observation, playerId, indices) {
    // Perfect information: the single world of beliefPopulation's size-1 set is
    // the observation itself. The exact-belief tracker below is never engaged
    // with fog off, so it would answer with an empty set.
    if (!observation.gameSpecific.fogOfWar) return (indices ?? []).includes(0) ? [observation] : [];
    const exact = getExactBelief(observation, playerId);
    const picks = exact.positionsAt(indices);
    if (!picks || !picks.length) return [];
    return picks.map(pos => ({
      ...observation,
      board: pos.board,
      units: boardToUnits(pos.board),
      // The world's posterior probability. Enumeration is without replacement and
      // ignores the posterior, so every aggregate over these worlds has to carry
      // the weight explicitly or it averages over the wrong measure. (sampleWorlds
      // deliberately omits this — see there.)
      beliefWeight: pos.w,
      gameSpecific: {
        ...observation.gameSpecific,
        castlingRights: pos.cr,
        enPassantTarget: pos.ep,
      },
    }));
  },

  // Which members of the belief population to SHOW a human, ranked best-guess
  // first — the counterpart of enumerateWorlds (same absolute indices) for an
  // analysis panel's "most likely board" stepper. These are real posterior
  // probabilities: see ExactBelief.rankByLikelihood, and note the one case where
  // they are not (a re-acquired set, flagged `approx`, has uniform weights).
  //
  // `probs` (the probability of EVERY index) comes back too, so a caller that
  // already holds indices from a different enumeration — the analysis walk's
  // engine-scored worlds — can label those with the same number.
  rankBeliefWorlds(observation, playerId, limit = 32) {
    // Perfect information: one world, the observation itself, with certainty.
    if (!observation.gameSpecific.fogOfWar) return { total: 1, top: [{ index: 0, prob: 1 }], probs: null };
    return getExactBelief(observation, playerId).rankByLikelihood(limit);
  },

  // The pieces a belief world places on squares the viewer cannot actually see —
  // i.e. exactly what a "here's what the board probably looks like" overlay has
  // to draw on top of the real fog. Derived by DIFFING the world against the
  // (fog-filtered) observation rather than consulting a visibility set: anything
  // the viewer can see is present and identical in every world by construction,
  // so a piece that disagrees with the observation is by definition hidden.
  hiddenPiecesOf(world, observation, playerId) {
    const FILES = 'abcdefgh';
    const SYMS = { king: 'k', queen: 'q', rook: 'r', bishop: 'b', knight: 'n', pawn: 'p' };
    const out = [];
    for (const sq of Object.keys(world.board ?? {})) {
      const pc = world.board[sq];
      if (!pc || pc.ownerId === playerId) continue;
      const seen = observation.board?.[sq];
      if (seen && seen.ownerId === pc.ownerId && seen.type === pc.type) continue; // really on screen
      out.push({ sq, type: SYMS[pc.type] ?? 'p', x: FILES.indexOf(sq[0]), y: 8 - parseInt(sq.slice(1), 10) });
    }
    return out;
  },

  // Let both belief trackers record the move we just chose, so next turn they
  // can advance P / detect our own captured pieces.
  onActionCommitted(observation, playerId, action) {
    if (!observation.gameSpecific.fogOfWar) return;
    getExactBelief(observation, playerId).commitOurMove(action);
    getBelief(observation, playerId).commitOurMove(action, observation.board);
  },
};

export default FogChess;
