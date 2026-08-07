// ---------------------------------------------------------------------------
// PGN → recorded fog session.
//
// The corpus this repo was fitted on is battle-simulator's own session JSON,
// which is a fine format and one nobody else emits. PGN is what a recorded game
// arrives as from anywhere else, and — the reason this module exists — it is the
// only common format that carries PLAYER RATINGS (`WhiteElo`/`BlackElo`), which
// is what lets π be fitted per strength band instead of pooled over everyone.
//
// The output is deliberately session-SHAPED: `{ params, log }`, exactly what
// `replayBelief` and the fitter already consume. So PGN support costs the rest
// of the codebase nothing — calibrate-belief, move-quality and the prior fit all
// gain it at once, and there is no second replay path to keep in step with the
// first.
//
// SAN IS RESOLVED AGAINST THE FOG MOVE SET, NOT REIMPLEMENTED. Every move is
// matched into `game.getLegalActions(state, seat)` — production's own action
// objects, with their `isCapture` / `isDoublePush` / `capturedSquare` bookkeeping
// already filled in. Two things follow, both load-bearing:
//
//  • The fitter's central assumption (the choice set it trains on is the choice
//    set production serves) holds by construction rather than by agreement
//    between two move generators.
//  • Fog moves that are ILLEGAL in ordinary chess parse fine — moving into
//    check, walking a pinned piece, capturing the king. A conventional SAN
//    parser rejects exactly those, and under fog they are not errors, they are
//    the interesting plies.
//
// The cost is a genuine ambiguity that ordinary SAN does not have: the fog move
// set is bigger, so `Nd2` can have two matches where a check-filtered generator
// would have one (the other knight being pinned). `disambiguate` prefers the
// check-legal reading, since that is the rule whoever wrote the SAN was
// following; a residual tie is reported, never guessed at.
// ---------------------------------------------------------------------------

import { FogChess } from './FogChess.js';
import { getAllLegalMoves } from './moves.js';
import { fromFEN, uciToAction } from './fen.js';

const PIECE_OF_LETTER = { K: 'king', Q: 'queen', R: 'rook', B: 'bishop', N: 'knight' };
const PROMO_OF_LETTER = { Q: 'queen', R: 'rook', B: 'bishop', N: 'knight' };

// Variant tags that mean "this is fog chess". A corpus recorded elsewhere spells
// it several ways; anything else with an explicit Variant tag is a different
// game and is refused rather than replayed under the wrong rules.
const FOG_VARIANTS = new Set(['fog of war', 'fogofwar', 'fog-of-war', 'fog',
  'dark chess', 'darkchess', 'dark', 'kriegspiel']);

const RESULTS = new Set(['1-0', '0-1', '1/2-1/2', '*']);

// ---------------------------------------------------------------------------
// Lexing
// ---------------------------------------------------------------------------

/**
 * Split a PGN file into raw games: `{ tags, tokens, result, lineNo }`.
 *
 * A generator, because a corpus PGN is routinely larger than memory is
 * comfortable with once every game is materialised at once; the caller converts
 * and discards one at a time.
 *
 * Movetext syntax handled: `{ }` comments (multi-line), `;` to end of line,
 * `( )` recursive annotation variations (DISCARDED — a variation is a move that
 * was NOT played, and the whole model is about what was), `$n` NAGs, move
 * numbers, and the four terminators.
 */
export function* parsePgn(text) {
  const src = String(text);
  let i = 0, line = 1;
  const n = src.length;

  while (i < n) {
    const tags = {};
    let gameLine = line;
    let sawAnything = false;

    // --- tag pair section ---
    while (i < n) {
      while (i < n && /\s/.test(src[i])) { if (src[i] === '\n') line++; i++; }
      if (src[i] !== '[') break;
      const close = src.indexOf(']', i);
      if (close < 0) { i = n; break; }
      const raw = src.slice(i + 1, close);
      const m = /^\s*([A-Za-z0-9_]+)\s*"((?:[^"\\]|\\.)*)"/.exec(raw);
      if (m) { tags[m[1]] = m[2].replace(/\\(["\\])/g, '$1'); sawAnything = true; }
      for (let k = i; k <= close; k++) if (src[k] === '\n') line++;
      i = close + 1;
      if (!sawAnything) gameLine = line;
    }

    // --- movetext ---
    const tokens = [];
    let result = null;
    let depth = 0;   // RAV nesting
    while (i < n) {
      const c = src[i];
      if (c === '\n') {
        line++; i++;
        // A blank line after movetext ends the game only once we are outside a
        // variation and have seen a terminator; PGN's real delimiter is the next
        // game's tag section, which the `[` check below catches.
        continue;
      }
      if (/\s/.test(c)) { i++; continue; }
      if (c === '{') { const e = src.indexOf('}', i); for (let k = i; k < (e < 0 ? n : e); k++) if (src[k] === '\n') line++; i = e < 0 ? n : e + 1; continue; }
      if (c === ';') { const e = src.indexOf('\n', i); i = e < 0 ? n : e; continue; }
      if (c === '(') { depth++; i++; continue; }
      if (c === ')') { depth = Math.max(0, depth - 1); i++; continue; }
      if (c === '[' && depth === 0 && (result || tokens.length)) break;   // next game
      if (c === '$') { while (i < n && !/\s/.test(src[i])) i++; continue; }

      let j = i;
      while (j < n && !/[\s{};()]/.test(src[j])) j++;
      const tok = src.slice(i, j);
      i = j;
      if (depth > 0) continue;                       // inside a variation
      if (RESULTS.has(tok)) { result = tok; continue; }
      if (/^\d+\.*$/.test(tok)) continue;            // move number
      if (tok === '.' || tok === '...') continue;
      if (tok) { tokens.push(tok); sawAnything = true; }
    }

    if (sawAnything) yield { tags, tokens, result, lineNo: gameLine };
    else if (i < n) i++;    // no progress made: don't spin
  }
}

// ---------------------------------------------------------------------------
// SAN / LAN → action
// ---------------------------------------------------------------------------

const SAN_RE = /^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([QRBN]))?/;
const LAN_RE = /^([a-h][1-8])([a-h][1-8])([qrbnQRBN])?$/;

/**
 * Narrow several fog-legal candidates to one.
 *
 * Under fog the action set is not check-filtered, so a SAN written by a normal
 * disambiguation rule can match more than one action here. Whoever wrote the
 * token was disambiguating against the ORDINARY legal set, so that reading wins
 * when it is unique. Anything still tied is returned as null and reported.
 */
function disambiguate(cands, state, seat) {
  if (cands.length <= 1) return cands[0] ?? null;
  const strict = getAllLegalMoves(state.board, seat, state.gameSpecific);
  const key = a => `${a.type}:${a.from}:${a.to}:${a.payload?.promote ?? ''}`;
  const ok = new Set(strict.map(key));
  const narrowed = cands.filter(a => ok.has(key(a)));
  return narrowed.length === 1 ? narrowed[0] : null;
}

/**
 * One movetext token → the matching action from `legal`, or null.
 *
 * Accepts SAN (`Nf3`, `exd5`, `e8=Q`, `O-O`), long algebraic with or without a
 * separator (`Ng1-f3`, `e2e4`) and bare UCI (`e7e8q`). Fog exports use all
 * three, and the check/mate suffixes are stripped rather than interpreted —
 * there is no check in this game, and a recorder that emits `+` anyway is not
 * telling us anything we act on.
 */
export function sanToAction(token, legal, state, seat) {
  let t = token.replace(/[+#!?]+$/, '').replace(/^\.+/, '');
  if (!t) return null;

  // Castling. `0-0` (digits) is common in European PGN; both O and 0 appear.
  const castleLike = t.replace(/0/g, 'O');
  if (castleLike === 'O-O' || castleLike === 'O-O-O') {
    const side = castleLike === 'O-O' ? 'kingside' : 'queenside';
    return legal.find(a => a.type === 'castle' && a.side === side) ?? null;
  }

  // Long algebraic / UCI. Checked BEFORE SAN because `e2e4` also matches the
  // SAN regex (as file-disambiguated pawn move "2e4"), and the UCI reading is
  // the correct one — and unambiguous, which the SAN reading is not.
  const lan = LAN_RE.exec(t.replace(/[-x]/g, ''));
  if (lan && !PIECE_OF_LETTER[t[0]]) {
    return uciToAction(lan[1] + lan[2] + (lan[3] ?? '').toLowerCase(), legal);
  }
  // `Ng1-f3` / `Ng1xf3`: a piece letter followed by a full origin and target.
  const pieceLan = /^([KQRBN])([a-h][1-8])[-x]?([a-h][1-8])$/.exec(t);
  if (pieceLan) {
    const type = PIECE_OF_LETTER[pieceLan[1]];
    return legal.find(a => a.type === 'move' && a.from === pieceLan[2] && a.to === pieceLan[3] &&
      state.board[a.from]?.type === type) ?? null;
  }

  const m = SAN_RE.exec(t);
  if (!m) return null;
  const [, letter, dFile, dRank, , to, promoLetter] = m;
  const type = letter ? PIECE_OF_LETTER[letter] : 'pawn';
  const promote = promoLetter ? PROMO_OF_LETTER[promoLetter] : null;

  const cands = legal.filter(a => {
    if (a.type !== 'move' || a.to !== to) return false;
    const piece = state.board[a.from];
    if (!piece || piece.type !== type) return false;
    if (dFile && a.from[0] !== dFile) return false;
    if (dRank && a.from[1] !== dRank) return false;
    // A promotion SAN without `=Q` is malformed but common; when the token
    // names no piece, accept the queen, which is what such recorders mean.
    if (a.payload?.promote) return promote ? a.payload.promote === promote : a.payload.promote === 'queen';
    return !promote;
  });
  return disambiguate(cands, state, seat);
}

// ---------------------------------------------------------------------------
// Game → session
// ---------------------------------------------------------------------------

/** `"1850"` → 1850; `"?"`, `""`, `"-"` and nonsense → null. */
export function parseRating(raw) {
  if (raw == null) return null;
  const n = Number(String(raw).trim());
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * Replay a list of movetext tokens into a session `{ params, log }`.
 *
 * The shared core of every corpus format: PGN and Chess.com's crawl JSON differ
 * only in where the tokens and the player metadata come from, and both end up
 * here so there is exactly one replay path to keep in step with production.
 *
 * Returns `{ sess, plies, skipped }`. `skipped` counts tokens that matched no
 * action, and the game is TRUNCATED at the first one rather than resynced:
 * every ply after a missed move is played against a board that never existed,
 * and silently feeding those to the fitter is how a corpus poisons a model.
 */
export function movesToSession(tokens, { game = FogChess, players, config }) {
  let state;
  try { state = game.createInitialState(players, config); } catch (err) { return { reason: err.message }; }

  const log = [];
  let skipped = 0;
  for (const tok of tokens) {
    const seat = state.activePlayers[0];
    const legal = game.getLegalActions(state, seat);
    const action = sanToAction(tok, legal, state, seat);
    if (!action) { skipped++; break; }
    const playerActions = [{ playerId: seat, action }];
    log.push({ turnNumber: state.turnNumber, phase: 'action', playerActions, events: [] });
    state = game.applyActions(state, playerActions);
    if (game.getResult(state)) break;
  }
  return { sess: { params: { game: 'chess', players, config }, log }, plies: log.length, skipped };
}

/**
 * Clean a move list that arrived as an ARRAY OF STRINGS rather than as movetext.
 *
 * A PGN lexer splits on whitespace, so these two problems cannot arise there;
 * a scraper pulling moves out of a rendered move list produces both, and both
 * were found in a real Chess.com Fog of War crawl:
 *
 *  • INTERNAL WHITESPACE. A disambiguated SAN is rendered as a piece glyph next
 *    to its origin hint, and the scrape keeps the gap: `"R  4g3"` is `R4g3`,
 *    `"N  8d7"` is `N8d7`. Every such token in that corpus was a real move that
 *    would otherwise have truncated its game at that ply.
 *  • TERMINATION GLYPHS. The move list's last cell is how the game ENDED, not a
 *    move — `R` (resigned), `T` (timed out), and a `P` that only ever appeared
 *    directly before an `R`. They carry no destination square, so they cannot be
 *    SAN under any reading, and the crawl counts them in its own ply total.
 *
 * Only a TRAILING RUN of unmovelike tokens is dropped. One appearing mid-list is
 * left in to fail loudly, because there it means the scrape lost a real move and
 * everything after it would be replayed against a board that never existed.
 */
export function normalizeMoveList(moves) {
  const tokens = moves.map(m => String(m).replace(/\s+/g, ''));
  // A move must name a destination square, or be castling. Nothing else is SAN.
  const movelike = t => /[a-h][1-8]/.test(t) || /^[O0]-[O0](-[O0])?$/.test(t.replace(/[+#!?]+$/, ''));
  let end = tokens.length;
  while (end > 0 && !movelike(tokens[end - 1])) end--;
  return { tokens: tokens.slice(0, end), dropped: tokens.slice(end) };
}

/** The `{ fog: true, ... }` config every corpus game is replayed under. */
export function fogConfig(extra = {}) {
  return { fog: true, fogOfWar: true, maxTurns: 1000, ...extra };
}

/** Is this variant name one of the ways "fog chess" gets spelled? */
export function isFogVariant(name) {
  return FOG_VARIANTS.has(String(name ?? '').trim().toLowerCase());
}

/**
 * Convert one parsed PGN game into a session, or `{ reason }` when it cannot be
 * replayed. See `movesToSession` for the replay itself.
 */
export function pgnGameToSession(parsed, {
  game = FogChess, allowAnyVariant = false, defaultActor = 'human',
} = {}) {
  const { tags, tokens } = parsed;
  if (tags.Variant && !allowAnyVariant && !isFogVariant(tags.Variant)) {
    return { reason: `variant "${tags.Variant}" is not fog chess` };
  }

  const ratings = {
    white: parseRating(tags.WhiteElo ?? tags.WhiteRating),
    black: parseRating(tags.BlackElo ?? tags.BlackRating),
  };
  const players = [
    { id: 'white', name: tags.White ?? 'White', agent: tags.WhiteType ?? defaultActor, rating: ratings.white },
    { id: 'black', name: tags.Black ?? 'Black', agent: tags.BlackType ?? defaultActor, rating: ratings.black },
  ];

  const config = fogConfig();
  if (tags.FEN) {
    let start;
    try { start = fromFEN(tags.FEN); } catch (err) { return { reason: `bad FEN tag: ${err.message}` }; }
    Object.assign(config, start);
  }

  const out = movesToSession(tokens, { game, players, config });
  if (out.reason) return out;
  return { ...out, ratings, result: parsed.result ?? tags.Result ?? null };
}

/**
 * Everything above, over a whole PGN file.
 *
 * Yields `{ sess, ratings, plies, skipped, result, index }` per replayable game
 * and pushes `{ index, reason }` onto `rejects` for the rest — the caller
 * reports the count, so a corpus that half-parses says so out loud.
 */
export function* pgnToSessions(text, opts = {}, rejects = []) {
  let index = 0;
  for (const parsed of parsePgn(text)) {
    const out = pgnGameToSession(parsed, opts);
    if (out.reason) rejects.push({ index, reason: out.reason, tags: parsed.tags });
    else yield { ...out, index };
    index++;
  }
}
