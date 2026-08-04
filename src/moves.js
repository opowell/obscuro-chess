import { fileIndex, rankOf, squareAt, isAttackedBy, isKingInCheck, applyMoveToBoard } from './board.js';

function opp(color) { return color === 'white' ? 'black' : 'white'; }

// ---------------------------------------------------------------------------
// Per-piece pseudo-legal move generators (ignore check)
// ---------------------------------------------------------------------------

function pawnMoves(board, unit, enPassantTarget) {
  const { id, ownerId, position } = unit;
  const fi = fileIndex(position);
  const r = rankOf(position);
  const dir = ownerId === 'white' ? 1 : -1;
  const startRank = ownerId === 'white' ? 2 : 7;
  const promoteRank = ownerId === 'white' ? 8 : 1;
  const actions = [];

  const mkMove = (to, extras = {}) => ({ type: 'move', unitId: id, from: position, to, ...extras });

  // Single push
  const pushSq = squareAt(fi, r + dir);
  if (pushSq && !board[pushSq]) {
    if (rankOf(pushSq) === promoteRank) {
      for (const p of ['queen', 'rook', 'bishop', 'knight']) {
        actions.push(mkMove(pushSq, { payload: { promote: p } }));
      }
    } else {
      actions.push(mkMove(pushSq));
      // Double push from starting rank
      if (r === startRank) {
        const push2Sq = squareAt(fi, r + dir * 2);
        if (push2Sq && !board[push2Sq]) {
          actions.push(mkMove(push2Sq, { isDoublePush: true }));
        }
      }
    }
  }

  // Diagonal captures
  for (const dfi of [-1, 1]) {
    const capSq = squareAt(fi + dfi, r + dir);
    if (!capSq) continue;

    // En passant
    if (capSq === enPassantTarget) {
      const capturedSq = squareAt(fi + dfi, r); // the double-pushed pawn's square
      const captured = board[capturedSq];
      if (captured && captured.ownerId === opp(ownerId) && captured.type === 'pawn') {
        actions.push(mkMove(capSq, {
          targetId: captured.id,
          isCapture: true,
          isEnPassant: true,
          capturedSquare: capturedSq,
        }));
      }
    }

    // Normal capture
    const occupant = board[capSq];
    if (occupant && occupant.ownerId === opp(ownerId)) {
      if (rankOf(capSq) === promoteRank) {
        for (const p of ['queen', 'rook', 'bishop', 'knight']) {
          actions.push(mkMove(capSq, { targetId: occupant.id, isCapture: true, payload: { promote: p } }));
        }
      } else {
        actions.push(mkMove(capSq, { targetId: occupant.id, isCapture: true }));
      }
    }
  }

  return actions;
}

function slidingMoves(board, unit, directions) {
  const { id, ownerId, position } = unit;
  const fi = fileIndex(position);
  const r = rankOf(position);
  const actions = [];

  for (const [dfi, dr] of directions) {
    let cfi = fi, cr = r;
    while (true) {
      cfi += dfi; cr += dr;
      const sq = squareAt(cfi, cr);
      if (!sq) break;
      const occupant = board[sq];
      if (occupant) {
        if (occupant.ownerId !== ownerId) {
          actions.push({ type: 'move', unitId: id, from: position, to: sq, targetId: occupant.id, isCapture: true });
        }
        break;
      }
      actions.push({ type: 'move', unitId: id, from: position, to: sq });
    }
  }
  return actions;
}

const ROOK_DIRS   = [[0,1],[0,-1],[1,0],[-1,0]];
const BISHOP_DIRS = [[1,1],[1,-1],[-1,1],[-1,-1]];
const QUEEN_DIRS  = [...ROOK_DIRS, ...BISHOP_DIRS];
const KING_OFFSETS = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]];
const KNIGHT_OFFSETS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];

function rookMoves(board, unit)   { return slidingMoves(board, unit, ROOK_DIRS); }
function bishopMoves(board, unit) { return slidingMoves(board, unit, BISHOP_DIRS); }
function queenMoves(board, unit)  { return slidingMoves(board, unit, QUEEN_DIRS); }

function knightMoves(board, unit) {
  const { id, ownerId, position } = unit;
  const fi = fileIndex(position);
  const r = rankOf(position);
  const actions = [];
  for (const [dfi, dr] of KNIGHT_OFFSETS) {
    const sq = squareAt(fi + dfi, r + dr);
    if (!sq) continue;
    const occupant = board[sq];
    if (occupant && occupant.ownerId === ownerId) continue;
    if (occupant) {
      actions.push({ type: 'move', unitId: id, from: position, to: sq, targetId: occupant.id, isCapture: true });
    } else {
      actions.push({ type: 'move', unitId: id, from: position, to: sq });
    }
  }
  return actions;
}

function kingMoves(board, unit, castlingRights, fogOfWar = false) {
  const { id, ownerId, position } = unit;
  const fi = fileIndex(position);
  const r = rankOf(position);
  const actions = [];
  const opponent = opp(ownerId);

  // Normal one-square moves
  for (const [dfi, dr] of KING_OFFSETS) {
    const sq = squareAt(fi + dfi, r + dr);
    if (!sq) continue;
    const occupant = board[sq];
    if (occupant && occupant.ownerId === ownerId) continue;
    if (occupant) {
      actions.push({ type: 'move', unitId: id, from: position, to: sq, targetId: occupant.id, isCapture: true });
    } else {
      actions.push({ type: 'move', unitId: id, from: position, to: sq });
    }
  }

  // Castling — in fog mode skip all attack checks (you can't see threats)
  const rights = castlingRights[ownerId];
  const backRank = ownerId === 'white' ? 1 : 8;
  const kingSq = 'e' + backRank;
  const canCastle = fogOfWar ? true : !isAttackedBy(board, kingSq, opponent);
  if (position === kingSq && canCastle) {
    // Kingside
    if (rights.kingSide) {
      const f1 = 'f' + backRank, g1 = 'g' + backRank, h1 = 'h' + backRank;
      const rookId = board[h1]?.id;
      const pathSafe = fogOfWar || (!isAttackedBy(board, f1, opponent) && !isAttackedBy(board, g1, opponent));
      if (!board[f1] && !board[g1] && rookId && pathSafe) {
        actions.push({ type: 'castle', unitId: id, side: 'kingside',
          from: kingSq, to: g1, rookId, rookFrom: h1, rookTo: f1 });
      }
    }
    // Queenside
    if (rights.queenSide) {
      const d1 = 'd' + backRank, c1 = 'c' + backRank, b1 = 'b' + backRank, a1 = 'a' + backRank;
      const rookId = board[a1]?.id;
      const pathSafe = fogOfWar || (!isAttackedBy(board, d1, opponent) && !isAttackedBy(board, c1, opponent));
      if (!board[d1] && !board[c1] && !board[b1] && rookId && pathSafe) {
        actions.push({ type: 'castle', unitId: id, side: 'queenside',
          from: kingSq, to: c1, rookId, rookFrom: a1, rookTo: d1 });
      }
    }
  }

  return actions;
}

// ---------------------------------------------------------------------------
// Pseudo-legal → legal (filter moves that leave king in check)
// ---------------------------------------------------------------------------

export function pseudoLegalForUnit(board, unit, gameSpecific, fogOfWar = false) {
  switch (unit.type) {
    case 'pawn':   return pawnMoves(board, unit, gameSpecific.enPassantTarget);
    case 'rook':   return rookMoves(board, unit);
    case 'knight': return knightMoves(board, unit);
    case 'bishop': return bishopMoves(board, unit);
    case 'queen':  return queenMoves(board, unit);
    case 'king':   return kingMoves(board, unit, gameSpecific.castlingRights, fogOfWar);
    default:       return [];
  }
}

/**
 * All legal moves for all pieces belonging to `color`.
 * @param {object} board
 * @param {string} color  'white' | 'black'
 * @param {object} gameSpecific
 * @returns {import('../types.js').Action[]}
 */
export function getAllLegalMoves(board, color, gameSpecific) {
  const actions = [];
  for (const sq of Object.keys(board)) {
    const unit = board[sq];
    if (!unit || unit.ownerId !== color) continue;
    const pseudo = pseudoLegalForUnit(board, unit, gameSpecific);
    for (const action of pseudo) {
      const nextBoard = applyMoveToBoard(board, action);
      if (!isKingInCheck(nextBoard, color)) {
        actions.push(action);
      }
    }
  }
  return actions;
}

/**
 * All pseudo-legal moves for `color` under fog-of-war rules.
 * Moves are not filtered for check — players can move into check since they
 * can't see threats. Castling skips the transit-square attack checks.
 * @param {object} board
 * @param {string} color  'white' | 'black'
 * @param {object} gameSpecific
 * @returns {import('../types.js').Action[]}
 */
export function getAllFogMoves(board, color, gameSpecific) {
  const actions = [];
  for (const sq of Object.keys(board)) {
    const unit = board[sq];
    if (!unit || unit.ownerId !== color) continue;
    actions.push(...pseudoLegalForUnit(board, unit, gameSpecific, true));
  }
  return actions;
}
