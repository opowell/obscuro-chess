// ---------------------------------------------------------------------------
// FEN <-> internal representation helpers, used to talk to Stockfish over UCI.
// ---------------------------------------------------------------------------

const LETTER = { king: 'k', queen: 'q', rook: 'r', bishop: 'b', knight: 'n', pawn: 'p' };
const PROMO = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };
const TYPE = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };

/**
 * Convert an internal board + game-state to a FEN string.
 * @param {object} board            square -> piece ({ ownerId, type })
 * @param {object} gs               gameSpecific (castlingRights, enPassantTarget, halfMoveClock)
 * @param {'w'|'b'} sideToMove
 * @param {number} [fullmove=1]     full-move counter (cosmetic for search)
 */
export function toFEN(board, gs, sideToMove = 'w', fullmove = 1) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank--) {
    let row = '';
    let empty = 0;
    for (const f of 'abcdefgh') {
      const p = board[f + rank];
      if (!p) { empty++; continue; }
      if (empty) { row += empty; empty = 0; }
      const ch = LETTER[p.type] ?? 'p';
      row += p.ownerId === 'white' ? ch.toUpperCase() : ch;
    }
    if (empty) row += empty;
    rows.push(row);
  }
  // Castling rights are emitted only when the BOARD can still support them —
  // king on its home square, rook on the matching corner.
  //
  // This is not belt-and-braces, it is load-bearing under fog. Belief worlds and
  // in-tree positions routinely carry rights that the placement contradicts (a
  // king imagined on d7 while `castlingRights` still says k/q), and a FEN like
  // that is ILLEGAL. Stockfish's answer to an illegal FEN is not an error, it is
  // silence — zero MultiPV lines — and the leaf evaluator then falls through to
  // the static evaluator for every child of that node without saying so. That
  // was the single largest source of lost leaf values in the fog search:
  //
  //   r5p1/pp1kpppp/1bq1P1r1/... b Qkq -   ← black king on d7, still claims k/q
  //
  // Deriving from the board makes every emitted FEN legal by construction, and
  // is exactly right for an imagined world: a king that has wandered has no
  // castling rights, whatever the bookkeeping says.
  const cr = gs?.castlingRights;
  const homeOk = (color, side) => {
    const rank = color === 'white' ? '1' : '8';
    const k = board['e' + rank];
    if (!k || k.type !== 'king' || k.ownerId !== color) return false;
    const r = board[(side === 'kingSide' ? 'h' : 'a') + rank];
    return !!r && r.type === 'rook' && r.ownerId === color;
  };
  let castle = '';
  if (cr) {
    if (cr.white?.kingSide  && homeOk('white', 'kingSide'))  castle += 'K';
    if (cr.white?.queenSide && homeOk('white', 'queenSide')) castle += 'Q';
    if (cr.black?.kingSide  && homeOk('black', 'kingSide'))  castle += 'k';
    if (cr.black?.queenSide && homeOk('black', 'queenSide')) castle += 'q';
  }
  return `${rows.join('/')} ${sideToMove} ${castle || '-'} ${gs?.enPassantTarget || '-'} ${gs?.halfMoveClock ?? 0} ${fullmove}`;
}

/**
 * Parse a FEN into the pieces `FogChess.createInitialState` takes as config:
 * `{ board, toMove, castlingRights, enPassantTarget, halfMoveClock, turnNumber }`.
 *
 * The inverse of `toFEN`, and it exists for the corpus loader: a recorded game
 * may carry a `[FEN]` tag, and a loader that silently ignored it would replay
 * every move of that game against the wrong board — producing a plausible-looking
 * stream of decisions that are all mismatched. Better to be able to read it.
 *
 * Unit ids are synthesised (`wP1`, `bN2`, …) in scan order. Nothing downstream
 * derives meaning from an id beyond uniqueness — the belief works on piece type
 * and owner — but they must be distinct, so the counter runs across the board.
 */
export function fromFEN(fen) {
  const parts = String(fen).trim().split(/\s+/);
  if (parts.length < 2) throw new Error(`fen: not a FEN: ${JSON.stringify(fen)}`);
  const [placement, side, castle = '-', ep = '-', half = '0', full = '1'] = parts;

  const board = {};
  const rows = placement.split('/');
  if (rows.length !== 8) throw new Error(`fen: expected 8 ranks, got ${rows.length}`);
  let n = 0;
  rows.forEach((row, i) => {
    const rank = 8 - i;
    let fileIdx = 0;
    for (const ch of row) {
      if (ch >= '1' && ch <= '8') { fileIdx += Number(ch); continue; }
      const type = TYPE[ch.toLowerCase()];
      if (!type) throw new Error(`fen: unknown piece "${ch}"`);
      if (fileIdx > 7) throw new Error(`fen: rank ${rank} overflows the board`);
      const sq = 'abcdefgh'[fileIdx] + rank;
      const ownerId = ch === ch.toUpperCase() ? 'white' : 'black';
      const id = (ownerId === 'white' ? 'w' : 'b') + ch.toUpperCase() + (++n);
      board[sq] = { id, ownerId, type, position: sq, alive: true };
      fileIdx++;
    }
  });

  return {
    board,
    toMove: side === 'b' ? 'black' : 'white',
    castlingRights: {
      white: { kingSide: castle.includes('K'), queenSide: castle.includes('Q') },
      black: { kingSide: castle.includes('k'), queenSide: castle.includes('q') },
    },
    enPassantTarget: ep === '-' ? null : ep,
    halfMoveClock: Number(half) || 0,
    turnNumber: Number(full) || 1,
  };
}

/**
 * Map a UCI move string (e.g. "e2e4", "e7e8q", "e1g1" for castling) to the
 * matching action from a list of legal actions, or null if none matches.
 */
export function uciToAction(uci, legalActions) {
  if (!uci || uci.length < 4) return null;
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promo = uci[4] ? PROMO[uci[4].toLowerCase()] : null;
  return (
    legalActions.find(a => a.from === from && a.to === to &&
      (promo ? a.payload?.promote === promo : !a.payload?.promote)) ??
    legalActions.find(a => a.from === from && a.to === to) ??
    null
  );
}
