export const FILES = 'abcdefgh';

export function fileOf(sq) { return sq[0]; }
export function rankOf(sq) { return parseInt(sq[1], 10); }
export function fileIndex(sq) { return FILES.indexOf(sq[0]); }

/** Convert algebraic square → {x, y} with x in [0,7] and y in [0,7]. */
export function squareToXY(sq) {
  return { x: fileIndex(sq), y: rankOf(sq) - 1 };
}

/** Convert algebraic square → [col, row] grid display coords (rank 8 → row 0, rank 1 → row 7). */
export function squareToGrid(sq) {
  return [fileIndex(sq), 8 - rankOf(sq)];
}

/** Convert 0-based file index + rank (1–8) → algebraic square, or null if off-board. */
export function squareAt(fi, rank) {
  if (fi < 0 || fi > 7 || rank < 1 || rank > 8) return null;
  return FILES[fi] + rank;
}

/**
 * Return true if any piece of `byColor` attacks `square`.
 * Uses ray-casting from the target square backward — no need to iterate all pieces.
 * @param {object} board
 * @param {string} square
 * @param {string} byColor  'white' | 'black'
 */
export function isAttackedBy(board, square, byColor) {
  const fi = fileIndex(square);
  const r = rankOf(square);

  // Pawns attack diagonally: a white pawn at (fi±1, r-1) attacks (fi, r).
  const pawnRankOffset = byColor === 'white' ? -1 : 1;
  for (const dfi of [-1, 1]) {
    const sq = squareAt(fi + dfi, r + pawnRankOffset);
    if (sq) {
      const p = board[sq];
      if (p && p.ownerId === byColor && p.type === 'pawn') return true;
    }
  }

  // Knights
  for (const [dfi, dr] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
    const sq = squareAt(fi + dfi, r + dr);
    if (sq) {
      const p = board[sq];
      if (p && p.ownerId === byColor && p.type === 'knight') return true;
    }
  }

  // Rooks / queens (rank & file rays)
  for (const [dfi, dr] of [[0,1],[0,-1],[1,0],[-1,0]]) {
    let cfi = fi, cr = r;
    while (true) {
      cfi += dfi; cr += dr;
      const sq = squareAt(cfi, cr);
      if (!sq) break;
      const p = board[sq];
      if (p) {
        if (p.ownerId === byColor && (p.type === 'rook' || p.type === 'queen')) return true;
        break;
      }
    }
  }

  // Bishops / queens (diagonal rays)
  for (const [dfi, dr] of [[1,1],[1,-1],[-1,1],[-1,-1]]) {
    let cfi = fi, cr = r;
    while (true) {
      cfi += dfi; cr += dr;
      const sq = squareAt(cfi, cr);
      if (!sq) break;
      const p = board[sq];
      if (p) {
        if (p.ownerId === byColor && (p.type === 'bishop' || p.type === 'queen')) return true;
        break;
      }
    }
  }

  // King
  for (const [dfi, dr] of [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]) {
    const sq = squareAt(fi + dfi, r + dr);
    if (sq) {
      const p = board[sq];
      if (p && p.ownerId === byColor && p.type === 'king') return true;
    }
  }

  return false;
}

/** Return true if the king of `color` is in check on `board`. */
export function isKingInCheck(board, color) {
  const kingSq = Object.keys(board).find(sq => {
    const p = board[sq];
    return p && p.ownerId === color && p.type === 'king';
  });
  if (!kingSq) return false;
  const opp = color === 'white' ? 'black' : 'white';
  return isAttackedBy(board, kingSq, opp);
}

/**
 * Apply a chess action to a board object, returning a new board.
 * Used for check detection only — does not update state.units or gameSpecific.
 */
export function applyMoveToBoard(board, action) {
  const next = { ...board };

  if (action.type === 'castle') {
    const king = board[action.from];
    next[action.from] = undefined;
    next[action.to] = { ...king, position: action.to };
    const rook = board[action.rookFrom];
    next[action.rookFrom] = undefined;
    next[action.rookTo] = { ...rook, position: action.rookTo };
    return next;
  }

  const piece = board[action.from];
  next[action.from] = undefined;
  const newType = action.payload?.promote ?? piece.type;
  next[action.to] = { ...piece, position: action.to, type: newType };

  if (action.isEnPassant && action.capturedSquare) {
    next[action.capturedSquare] = undefined;
  }

  return next;
}

/**
 * Compute the set of squares visible to `color` under fog-of-war rules.
 * A square is visible if a friendly piece occupies it, can move to it, or attacks it.
 * Sliding pieces see along rays up to and including the first blocking piece.
 * Pawns reveal both push squares and diagonal attack squares even when empty.
 */
export function getVisibleSquares(board, color) {
  const visible = new Set();
  const ROOK_DIRS   = [[0,1],[0,-1],[1,0],[-1,0]];
  const BISHOP_DIRS = [[1,1],[1,-1],[-1,1],[-1,-1]];

  for (const sq of Object.keys(board)) {
    const piece = board[sq];
    if (!piece || piece.ownerId !== color) continue;

    visible.add(sq);
    const fi = fileIndex(sq);
    const r  = rankOf(sq);

    if (piece.type === 'pawn') {
      const dir = color === 'white' ? 1 : -1;
      // Forward squares: only visible when not blocked (pawn can't see through blockers)
      const pushSq = squareAt(fi, r + dir);
      if (pushSq && !board[pushSq]) {
        visible.add(pushSq);
        if (r === (color === 'white' ? 2 : 7)) {
          const push2 = squareAt(fi, r + dir * 2);
          if (push2 && !board[push2]) visible.add(push2);
        }
      }
      // Diagonal attack squares: always visible (pawn always watches these)
      for (const dfi of [-1, 1]) {
        const capSq = squareAt(fi + dfi, r + dir);
        if (capSq) visible.add(capSq);
      }
    } else if (piece.type === 'knight') {
      for (const [dfi, dr] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
        const dest = squareAt(fi + dfi, r + dr);
        if (dest) visible.add(dest);
      }
    } else if (piece.type === 'king') {
      for (const [dfi, dr] of [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]) {
        const dest = squareAt(fi + dfi, r + dr);
        if (dest) visible.add(dest);
      }
    } else {
      // Sliding pieces: rook, bishop, queen
      const dirs = piece.type === 'rook' ? ROOK_DIRS
        : piece.type === 'bishop' ? BISHOP_DIRS
        : [...ROOK_DIRS, ...BISHOP_DIRS];
      for (const [dfi, dr] of dirs) {
        let cfi = fi, cr = r;
        while (true) {
          cfi += dfi; cr += dr;
          const dest = squareAt(cfi, cr);
          if (!dest) break;
          visible.add(dest);
          if (board[dest]) break; // can see blocker but not beyond
        }
      }
    }
  }
  return visible;
}

/**
 * Render the board as an ASCII diagram.
 * @param {object} board
 * @returns {string}
 */
export function renderBoard(board) {
  const SYMBOLS = {
    white: { king:'K', queen:'Q', rook:'R', bishop:'B', knight:'N', pawn:'P' },
    black: { king:'k', queen:'q', rook:'r', bishop:'b', knight:'n', pawn:'p' },
  };
  const rows = [];
  rows.push('  a b c d e f g h');
  for (let rank = 8; rank >= 1; rank--) {
    let row = rank + ' ';
    for (const file of FILES) {
      const sq = file + rank;
      const p = board[sq];
      row += p ? SYMBOLS[p.ownerId][p.type] : '.';
      row += ' ';
    }
    rows.push(row.trimEnd());
  }
  rows.push('  a b c d e f g h');
  return rows.join('\n');
}
