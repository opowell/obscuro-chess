// ---------------------------------------------------------------------------
// Reading a corpus back in: the zip reader, the PGN parser, the loader that
// dispatches between them, and the rating bands fitted on top.
//
// The load-bearing test here is the ROUND TRIP: take a real recorded session,
// write its moves out as a PGN, read them back, and assert the replayed action
// stream is identical ply for ply. That is the property the whole corpus path
// rests on — a PGN game that parses into a DIFFERENT sequence of positions is
// worse than one that fails to parse, because the fitter would train on it and
// report a confident number over boards that never occurred.
//
// The zip fixtures are BUILT HERE rather than committed, so the reader is tested
// against archives whose bytes this repo laid out by hand from the spec, both
// STORED and DEFLATEd — not against one blob that could encode the same
// misunderstanding on both sides.
// ---------------------------------------------------------------------------

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateRawSync, gzipSync } from 'node:zlib';

import { openZip } from '../src/zip.js';
import {
  parsePgn, pgnToSessions, pgnGameToSession, sanToAction, parseRating, normalizeMoveList,
} from '../src/pgn.js';
import { loadCorpus, iterCorpus, ratingSpread } from '../src/corpus.js';
import {
  weightsForRating, ratingZ, FITTED_WEIGHTS, RATING_SLOPE, RATING_PIVOT, RATING_SCALE,
} from '../src/movePrior.js';
import { getExactBelief, getDefaultMovePrior } from '../src/exactBelief.js';
import { setOverrides, resetSettings } from '../src/config.js';
import { FogChess } from '../src/FogChess.js';
import { fromFEN, toFEN } from '../src/fen.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');

const PROMO_LETTER = { queen: 'q', rook: 'r', bishop: 'b', knight: 'n' };

/** One recorded session's log as UCI tokens — the input side of the round trip. */
function uciTokens(sess) {
  return sess.log.map(e => {
    const a = e.playerActions[0].action;
    return a.from + a.to + (a.payload?.promote ? PROMO_LETTER[a.payload.promote] : '');
  });
}

function fixtureSessions() {
  return readdirSync(FIXTURES).filter(f => f.endsWith('.json'))
    .map(f => ({ file: f, sess: JSON.parse(readFileSync(join(FIXTURES, f), 'utf8')) }));
}

function pgnFor(tokens, tags = {}) {
  const head = Object.entries({ Variant: 'Fog of War', White: 'w', Black: 'b', ...tags })
    .map(([k, v]) => `[${k} "${v}"]`).join('\n');
  return `${head}\n\n${tokens.join(' ')} *\n`;
}

// ---------------------------------------------------------------------------
// A minimal ZIP WRITER, for the reader to be tested against.
// ---------------------------------------------------------------------------

function makeZip(files, { store = false } = {}) {
  const locals = [], central = [];
  let offset = 0;
  for (const [name, contentStr] of Object.entries(files)) {
    const content = Buffer.from(contentStr);
    const data = store ? content : deflateRawSync(content);
    const nameBuf = Buffer.from(name, 'utf8');
    const crc = crc32(content);

    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4);                       // version needed
    lh.writeUInt16LE(store ? 0 : 8, 8);            // method
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(data.length, 18);
    lh.writeUInt32LE(content.length, 22);
    lh.writeUInt16LE(nameBuf.length, 26);
    locals.push(lh, nameBuf, data);

    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0);
    ch.writeUInt16LE(20, 6);
    ch.writeUInt16LE(store ? 0 : 8, 10);
    ch.writeUInt32LE(crc, 16);
    ch.writeUInt32LE(data.length, 20);
    ch.writeUInt32LE(content.length, 24);
    ch.writeUInt16LE(nameBuf.length, 28);
    ch.writeUInt32LE(offset, 42);
    central.push(ch, nameBuf);

    offset += 30 + nameBuf.length + data.length;
  }
  const localBuf = Buffer.concat(locals);
  const centralBuf = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(Object.keys(files).length, 8);
  eocd.writeUInt16LE(Object.keys(files).length, 10);
  eocd.writeUInt32LE(centralBuf.length, 12);
  eocd.writeUInt32LE(localBuf.length, 16);
  return Buffer.concat([localBuf, centralBuf, eocd]);
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** A scratch directory, cleaned up by the OS rather than by us. */
function scratch() {
  return mkdtempSync(join(tmpdir(), 'obscuro-corpus-'));
}

// ---------------------------------------------------------------------------
// zip
// ---------------------------------------------------------------------------

for (const store of [false, true]) {
  test(`zip: reads ${store ? 'STORED' : 'DEFLATEd'} members back byte-for-byte`, () => {
    const files = {
      'a.txt': 'hello',
      'nested/b.json': JSON.stringify({ x: 1, y: [1, 2, 3] }),
      // Long and repetitive, so DEFLATE actually compresses and the two arms
      // are not accidentally the same code path.
      'nested/deep/c.pgn': 'e2e4 e7e5 '.repeat(500),
    };
    const dir = scratch();
    const path = join(dir, 'a.zip');
    writeFileSync(path, makeZip(files, { store }));

    const z = openZip(path);
    try {
      const seen = new Map();
      for (const e of z.entries()) seen.set(e.name, z.read(e).toString());
      assert.deepEqual([...seen.keys()].sort(), Object.keys(files).sort());
      for (const [name, content] of Object.entries(files)) {
        assert.equal(seen.get(name), content, `${name} round-trips`);
      }
    } finally { z.close(); }
  });
}

test('zip: a file that is not a zip is refused, not half-read', () => {
  const dir = scratch();
  const path = join(dir, 'not.zip');
  writeFileSync(path, 'this is plainly not a zip archive');
  assert.throws(() => openZip(path), /not a zip file/);
});

test('zip: an unsupported compression method names itself', () => {
  const dir = scratch();
  const path = join(dir, 'lzma.zip');
  const buf = makeZip({ 'x.txt': 'abc' });
  // Method 14 (LZMA) in both the local and central headers.
  buf.writeUInt16LE(14, 8);
  const cenAt = buf.indexOf(Buffer.from([0x50, 0x4b, 0x01, 0x02]));
  buf.writeUInt16LE(14, cenAt + 10);
  writeFileSync(path, buf);
  const z = openZip(path);
  try {
    const [e] = [...z.entries()];
    assert.throws(() => z.read(e), /unsupported compression method 14/);
  } finally { z.close(); }
});

// ---------------------------------------------------------------------------
// PGN
// ---------------------------------------------------------------------------

test('pgn: a recorded session survives the PGN round trip ply for ply', () => {
  for (const { file, sess } of fixtureSessions()) {
    const pgn = pgnFor(uciTokens(sess), { WhiteElo: 1850, BlackElo: 2110 });
    const [game] = [...pgnToSessions(pgn)];
    assert.ok(game, `${file} produced a game`);
    assert.equal(game.skipped, 0, `${file}: every move matched a fog-legal action`);
    assert.equal(game.plies, sess.log.length, `${file}: same number of plies`);
    for (let i = 0; i < sess.log.length; i++) {
      const got = game.sess.log[i].playerActions[0];
      const want = sess.log[i].playerActions[0];
      assert.equal(got.playerId, want.playerId, `${file} ply ${i}: same seat`);
      assert.equal(got.action.type, want.action.type, `${file} ply ${i}: same action type`);
      assert.equal(got.action.from, want.action.from, `${file} ply ${i}: same origin`);
      assert.equal(got.action.to, want.action.to, `${file} ply ${i}: same target`);
      assert.equal(got.action.payload?.promote, want.action.payload?.promote,
        `${file} ply ${i}: same promotion`);
    }
  }
});

test('pgn: SAN, castling and disambiguation resolve against the fog move set', () => {
  const pgn = pgnFor([], {}).replace('*\n',
    '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 ' +
    '8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 *\n');
  const [game] = [...pgnToSessions(pgn)];
  assert.equal(game.skipped, 0);
  assert.equal(game.plies, 22);
  const moves = game.sess.log.map(e => {
    const a = e.playerActions[0].action;
    return a.type === 'castle' ? (a.side === 'kingside' ? 'O-O' : 'O-O-O') : a.from + a.to;
  });
  assert.equal(moves[8], 'O-O', 'white castles kingside on move 5');
  assert.equal(moves[9], 'f8e7', 'Be7 is the f8 bishop');
  assert.equal(moves[10], 'f1e1', 'Re1 is the castled rook, not the a1 one');
  // Nbd2: the b1 knight, disambiguated from the f3 one, which can also reach d2.
  assert.equal(moves[20], 'b1d2', 'Nbd2 picks the b-file knight');
});

test('pgn: comments, NAGs, variations and result markers are all discarded', () => {
  const pgn = '[Variant "Fog of War"]\n\n' +
    '1. e4 {a comment\nspanning lines} e5 $1 2. Nf3 (2. Bc4 Nc6) Nc6 ; trailing\n3. Bb5 1-0\n';
  const [game] = [...pgnToSessions(pgn)];
  assert.equal(game.plies, 5, 'the variation contributed no plies');
  assert.deepEqual(game.sess.log.map(e => {
    const a = e.playerActions[0].action; return a.from + a.to;
  }), ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5']);
});

test('pgn: several games in one file are separated at the tag sections', () => {
  const text = pgnFor(['e2e4', 'e7e5'], { White: 'a' }) + '\n' +
    pgnFor(['d2d4', 'd7d5', 'c2c4'], { White: 'b' });
  const games = [...pgnToSessions(text)];
  assert.equal(games.length, 2);
  assert.equal(games[0].plies, 2);
  assert.equal(games[1].plies, 3);
  assert.equal(games[0].sess.params.players[0].name, 'a');
  assert.equal(games[1].sess.params.players[0].name, 'b');
});

test('pgn: ratings are read per seat, and nonsense reads as unrated', () => {
  const [game] = [...pgnToSessions(pgnFor(['e2e4', 'e7e5'], { WhiteElo: '1850', BlackElo: '?' }))];
  assert.deepEqual(game.ratings, { white: 1850, black: null });
  assert.equal(parseRating('2200'), 2200);
  for (const bad of ['?', '', '-', 'unrated', null, undefined, '0']) {
    assert.equal(parseRating(bad), null, `${JSON.stringify(bad)} is not a rating`);
  }
});

test('pgn: a non-fog variant is refused rather than replayed under the wrong rules', () => {
  const rejects = [];
  const games = [...pgnToSessions(pgnFor(['e2e4'], { Variant: 'Atomic' }), {}, rejects)];
  assert.equal(games.length, 0);
  assert.match(rejects[0].reason, /not fog chess/);
  // …unless the caller says the corpus is fog anyway.
  const forced = [...pgnToSessions(pgnFor(['e2e4'], { Variant: 'Atomic' }), { allowAnyVariant: true })];
  assert.equal(forced.length, 1);
});

test('pgn: a game truncates at the first unparseable move instead of resyncing', () => {
  // Nf6 is not available to white on move 2; everything after it would be
  // played against a board that never existed.
  const [game] = [...pgnToSessions(pgnFor(['e2e4', 'e7e5', 'Nf6', 'Nc6', 'Bb5']))];
  assert.equal(game.plies, 2, 'stops at the bad token');
  assert.equal(game.skipped, 1);
});

test('pgn: a [FEN] start position is honoured', () => {
  // A fog endgame: white king and pawn, black king. Black to move.
  const fen = '4k3/8/8/8/8/8/4P3/4K3 b - - 0 40';
  const [game] = [...pgnToSessions(pgnFor(['e8d7'], { SetUp: '1', FEN: fen }))];
  assert.equal(game.plies, 1);
  const state = FogChess.createInitialState(game.sess.params.players, game.sess.params.config);
  assert.equal(state.activePlayers[0], 'black', 'black is to move, per the FEN');
  assert.equal(Object.keys(state.board).length, 3);
  assert.equal(toFEN(state.board, state.gameSpecific, 'b', 40), fen);
});

test('fen: fromFEN inverts toFEN over the fixtures\' own positions', () => {
  const { sess } = fixtureSessions()[0];
  let state = FogChess.createInitialState(sess.params.players, sess.params.config);
  for (const entry of sess.log.slice(0, 12)) {
    const side = state.activePlayers[0] === 'white' ? 'w' : 'b';
    const fen = toFEN(state.board, state.gameSpecific, side, state.turnNumber);
    const back = fromFEN(fen);
    assert.equal(toFEN(back.board, back, side, state.turnNumber), fen, 'FEN round-trips');
    state = FogChess.applyActions(state, entry.playerActions);
  }
});

test('fen: a malformed FEN throws instead of producing half a board', () => {
  assert.throws(() => fromFEN('nonsense'), /not a FEN/);
  assert.throws(() => fromFEN('4k3/8/8/8 w - - 0 1'), /expected 8 ranks/);
  assert.throws(() => fromFEN('4k3/8/8/8/8/8/8/4Y3 w - - 0 1'), /unknown piece/);
});

// ---------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------

test('corpus: a directory of session JSON loads as it always did', () => {
  const { games, stats } = loadCorpus(FIXTURES);
  assert.equal(games.length, 3);
  assert.equal(stats.rejected, 0);
  for (const g of games) {
    assert.ok(g.plies >= 10);
    assert.equal(g.sess.params.game, 'chess');
    assert.deepEqual(g.ratings, { white: null, black: null }, 'session JSON carries no ratings');
    assert.equal(g.actors.black, 'obscuro');
  }
});

test('corpus: directories are walked recursively, and .gz is transparent', () => {
  const dir = scratch();
  mkdirSync(join(dir, 'a', 'b'), { recursive: true });
  const { sess } = fixtureSessions()[0];
  const pgn = pgnFor(uciTokens(sess), { WhiteElo: 1400, BlackElo: 1450 });
  writeFileSync(join(dir, 'a', 'top.pgn'), pgn);
  writeFileSync(join(dir, 'a', 'b', 'deep.pgn.gz'), gzipSync(pgn));

  const { games } = loadCorpus(dir);
  assert.equal(games.length, 2, 'both the nested file and the gzipped one');
  for (const g of games) assert.deepEqual(g.ratings, { white: 1400, black: 1450 });
});

test('corpus: a zip is walked like a directory, mixed formats and all', () => {
  const { sess } = fixtureSessions()[0];
  const dir = scratch();
  const path = join(dir, 'corpus.zip');
  writeFileSync(path, makeZip({
    'games/one.json': JSON.stringify(sess),
    'games/two.pgn': pgnFor(uciTokens(sess), { WhiteElo: 2400, BlackElo: 2380 }),
    'README.md': 'not a corpus file',
  }));

  const { games, stats } = loadCorpus(path);
  assert.equal(games.length, 2);
  assert.equal(stats.ignored, 1, 'the README is ignored, not rejected');
  const rated = games.find(g => g.ratings.white === 2400);
  assert.ok(rated, 'the PGN game kept its ratings through the zip');
  assert.equal(rated.plies, sess.log.length);
});

test('corpus: rejections are counted and explained, never silent', () => {
  const dir = scratch();
  writeFileSync(join(dir, 'short.json'), JSON.stringify({
    params: { game: 'chess', config: { fog: true }, players: [] }, log: [1, 2, 3],
  }));
  writeFileSync(join(dir, 'nofog.json'), JSON.stringify({
    params: { game: 'chess', config: {}, players: [] }, log: new Array(40).fill(null),
  }));
  writeFileSync(join(dir, 'othergame.json'), JSON.stringify({ params: { game: 'civ1' }, log: [] }));
  writeFileSync(join(dir, 'broken.json'), '{ not json');

  const { games, stats } = loadCorpus(dir);
  assert.equal(games.length, 0);
  assert.equal(stats.rejected, 4);
  const reasons = [...stats.reasons.keys()].join('|');
  for (const want of ['plies', 'fog', 'chess', 'JSON']) {
    assert.match(reasons, new RegExp(want), `"${want}" appears among the reasons`);
  }
});

test('corpus: maxGames stops the walk instead of loading everything first', () => {
  let yielded = 0;
  for (const _g of iterCorpus(FIXTURES)) { if (++yielded >= 2) break; }
  assert.equal(yielded, 2);
  assert.equal(loadCorpus(FIXTURES, { maxGames: 1 }).games.length, 1);
});

// ---------------------------------------------------------------------------
// crawl JSON — a scraped archive of many games in one file
// ---------------------------------------------------------------------------

/** The shape a Chess.com Fog of War archive crawl produces. */
function crawlDoc(games) {
  return { exportedAt: '2026-08-06T10:31:06.752Z', gameCount: games.length, games };
}
function crawlGame(moves, extra = {}) {
  return {
    gameId: String(extra.gameId ?? 1), variant: 'Fog of War', result: '1-0',
    plies: moves.length, white: 'alice', black: 'bob',
    players: [{ username: 'bob', rating: 2100 }, { username: 'alice', rating: 1850 }],
    moves, ...extra,
  };
}

test('corpus: a crawl file is detected by content, not by extension', () => {
  const { sess } = fixtureSessions()[0];
  const dir = scratch();
  // Named exactly like a session recording — only the content distinguishes it.
  const path = join(dir, 'session-looking-name.json');
  writeFileSync(path, JSON.stringify(crawlDoc([
    crawlGame(uciTokens(sess), { gameId: 'a' }),
    crawlGame(uciTokens(sess).slice(0, 30), { gameId: 'b' }),
  ])));

  const { games } = loadCorpus(path);
  assert.equal(games.length, 2);
  assert.equal(games[0].plies, sess.log.length);
  // Ratings join by username, NOT by position: `players` is in crawl order,
  // which here is black-then-white.
  assert.deepEqual(games[0].ratings, { white: 1850, black: 2100 });
});

test('corpus: a crawl game met twice is loaded once', () => {
  const { sess } = fixtureSessions()[0];
  const toks = uciTokens(sess);
  const dir = scratch();
  const path = join(dir, 'crawl.json');
  writeFileSync(path, JSON.stringify(crawlDoc([
    crawlGame(toks, { gameId: 'dup' }),
    crawlGame(toks, { gameId: 'dup', seenOnArchiveOf: 'bob' }),
    crawlGame(toks, { gameId: 'other' }),
  ])));

  const { games, stats } = loadCorpus(path);
  assert.equal(games.length, 2, 'the duplicate gameId is dropped');
  assert.equal(stats.reasons.get('duplicate gameId'), 1);
});

test('crawl: a trailing termination glyph is not a move', () => {
  // R (resigned), T (timed out) and the P that precedes R in the source corpus.
  for (const tail of [['R'], ['T'], ['P', 'R']]) {
    const { tokens, dropped } = normalizeMoveList(['e4', 'e5', 'Nf3', ...tail]);
    assert.deepEqual(tokens, ['e4', 'e5', 'Nf3'], `${tail.join('+')} is dropped`);
    assert.deepEqual(dropped, tail);
  }
  // …and a game ending in one still replays every real move.
  const dir = scratch();
  const path = join(dir, 'crawl.json');
  const moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 'O-O', 'Be7', 'R'];
  writeFileSync(path, JSON.stringify(crawlDoc([crawlGame(moves, { plies: moves.length })])));
  const { games, stats } = loadCorpus(path, { minPlies: 5 });
  assert.equal(games.length, 1);
  assert.equal(games[0].plies, 10, 'ten real moves replayed');
  assert.equal(stats.truncated, 0, 'the glyph is not a parse failure');
  assert.equal(games[0].sourcePlies, 10, 'the source ply count is compared net of the glyph');
});

test('crawl: internal whitespace in a disambiguated SAN is repaired', () => {
  // The scraper renders `R4g3` as a piece glyph beside its rank hint and the
  // gap survives the scrape. Every such token in the source corpus was a real
  // move that would otherwise have truncated its game at that ply.
  assert.deepEqual(normalizeMoveList(['R  4g3', 'N  8d7']).tokens, ['R4g3', 'N8d7']);

  // Two black knights, on f6 and f8, both able to reach d7 — which is exactly
  // when Chess.com emits the rank-disambiguated form.
  const moves = ['c3', 'c6', 'b3', 'd5', 'Ba3', 'Bf5', 'Nf3', 'Bg6', 'e3', 'Nd7',
    'Be2', 'Ngf6', 'g3', 'e6', 'Bxf8', 'Nxf8', 'a3', 'N  8d7'];
  const dir = scratch();
  const path = join(dir, 'crawl.json');
  writeFileSync(path, JSON.stringify(crawlDoc([crawlGame(moves)])));
  const { games, stats } = loadCorpus(path, { minPlies: 5 });
  assert.equal(stats.truncated, 0);
  assert.equal(games[0].plies, 18, 'the mangled token replayed as a real move');
  const last = games[0].sess.log[17].playerActions[0].action;
  assert.equal(last.from, 'f8', 'N8d7 is the knight on the 8th rank, not the f6 one');
  assert.equal(last.to, 'd7');
});

test('crawl: an unmovelike token MID-list still truncates, loudly', () => {
  // Only a trailing run is termination. One in the middle means the scrape lost
  // a move, and everything after it would be replayed against a fiction.
  const { tokens, dropped } = normalizeMoveList(['e4', 'R', 'e5', 'Nf3']);
  assert.deepEqual(tokens, ['e4', 'R', 'e5', 'Nf3'], 'nothing is dropped mid-list');
  assert.deepEqual(dropped, []);

  const dir = scratch();
  const path = join(dir, 'crawl.json');
  writeFileSync(path, JSON.stringify(crawlDoc([
    crawlGame(['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'R', 'a6', 'Ba4', 'Nf6', 'O-O']),
  ])));
  const { games, stats } = loadCorpus(path, { minPlies: 3 });
  assert.equal(games[0].plies, 5, 'stops at the stray glyph');
  assert.equal(stats.truncated, 1, 'and is counted as a truncation');
});

test('crawl: a non-fog variant in a crawl is refused', () => {
  const dir = scratch();
  const path = join(dir, 'crawl.json');
  writeFileSync(path, JSON.stringify(crawlDoc([
    crawlGame(['e4', 'e5', 'Nf3', 'Nc6'], { variant: 'Bughouse', gameId: 'x' }),
  ])));
  const { games, stats } = loadCorpus(path, { minPlies: 2 });
  assert.equal(games.length, 0);
  assert.match([...stats.reasons.keys()].join(), /not fog chess/);
});

// ---------------------------------------------------------------------------
// rating as a continuous term
// ---------------------------------------------------------------------------

test('corpus: ratingSpread reports the population a rating fit would cover', () => {
  const g = r => ({ ratings: { white: r, black: r + 100 } });
  const s = ratingSpread([g(1400), g(1800), g(2200), g(1600)]);
  assert.equal(s.n, 8);
  assert.equal(s.min, 1400);
  assert.equal(s.max, 2300);
  assert.equal(ratingSpread([{ ratings: { white: null, black: null } }]), null,
    'an unrated corpus has no spread, rather than a spread of zero');
});

test('movePrior: the shipped slopes are zero, so serving is rating-independent', () => {
  assert.deepEqual(RATING_SLOPE, [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'no corpus has yet shown that rating moves these weights out of sample');
  for (const r of [null, 800, 1500, 2600]) {
    assert.equal(weightsForRating(r), FITTED_WEIGHTS,
      `rating ${r} serves the flat model unchanged`);
  }
});

test('movePrior: z is centred on the pivot and scaled by the Elo unit', () => {
  assert.equal(ratingZ(RATING_PIVOT), 0, 'the pivot is z = 0');
  assert.equal(ratingZ(RATING_PIVOT + RATING_SCALE), 1);
  assert.equal(ratingZ(RATING_PIVOT - RATING_SCALE), -1);
});

test('movePrior: a slope tilts the weights linearly in rating', () => {
  // One term only, so the arithmetic is checkable by hand: castleBonus rises by
  // 100 per RATING_SCALE Elo above the pivot.
  const slope = [0, 0, 0, 0, 0, 0, 0, 0, 100];
  const at = r => weightsForRating(r, { slope }).castleBonus;
  const base = FITTED_WEIGHTS.castleBonus;

  assert.ok(Math.abs(at(RATING_PIVOT) - base) < 1e-9, 'no tilt at the pivot');
  assert.ok(Math.abs(at(RATING_PIVOT + RATING_SCALE) - (base + 100)) < 1e-9);
  assert.ok(Math.abs(at(RATING_PIVOT - RATING_SCALE) - (base - 100)) < 1e-9);
  // Linear in between, not stepped — the whole point of dropping bands.
  assert.ok(Math.abs(at(RATING_PIVOT + RATING_SCALE / 2) - (base + 50)) < 1e-9);
  // Two adjacent ratings differ by a hair, not by a whole model — which is the
  // discontinuity that bucketing had and this does not.
  assert.ok(at(1901) > at(1899), 'monotone in rating');
  assert.ok(Math.abs(at(1901) - at(1899)) < 1, 'and the step across a bucket edge is a hair');

  // Clamped, so an outlier rating cannot extrapolate off the corpus.
  assert.equal(at(RATING_PIVOT + 10 * RATING_SCALE), at(RATING_PIVOT + 1.5 * RATING_SCALE),
    'z is clamped at +1.5');
  assert.equal(at(RATING_PIVOT - 10 * RATING_SCALE), at(RATING_PIVOT - 1.5 * RATING_SCALE));

  // Unknown ratings get the flat model, never z = 0 by accident of arithmetic —
  // it is the same answer here, but for a reason worth pinning.
  assert.equal(weightsForRating(null, { slope }), FITTED_WEIGHTS);
  assert.equal(weightsForRating(NaN, { slope }), FITTED_WEIGHTS);
});

test('belief: a configured slope reaches the belief, keyed on the OPPONENT', () => {
  const slope = [0, 0, 0, 0, 0, 0, 0, 0, 150];
  try {
    setOverrides({ chess: { MOVE_PRIOR_RATING_SLOPE: slope } });
    const flat = getDefaultMovePrior();

    // π models the opponent, so it is BLACK's rating that tilts white's model.
    const rated = FogChess.createInitialState(
      [{ id: 'white', rating: 2500 }, { id: 'black', rating: 1400 }], { fogOfWar: true });
    assert.notEqual(getExactBelief(rated, 'white')._prior, flat,
      'white is tilted by black\'s 1400, not by its own 2500');
    assert.notEqual(getExactBelief(rated, 'white')._prior, getExactBelief(rated, 'black')._prior,
      'the two seats face different opponents, so they run different models');

    // An opponent at the pivot is a no-op tilt, but still a compiled prior —
    // what matters is that it behaves identically, not that it is the same object.
    const unrated = FogChess.createInitialState(
      [{ id: 'white' }, { id: 'black' }], { fogOfWar: true });
    assert.equal(getExactBelief(unrated, 'white')._prior, flat, 'unrated serves the flat model');

    // gameSpecific.opponentRating is the other way a host supplies it.
    const viaConfig = FogChess.createInitialState(
      [{ id: 'white' }, { id: 'black' }], { fogOfWar: true });
    viaConfig.gameSpecific.opponentRating = 1400;
    assert.notEqual(getExactBelief(viaConfig, 'white')._prior, flat);
  } finally { resetSettings(); }
});

test('belief: with the shipped zero slopes, rating changes nothing at all', () => {
  const flat = getDefaultMovePrior();
  for (const rating of [1200, 1800, 2400]) {
    const s = FogChess.createInitialState(
      [{ id: 'white' }, { id: 'black', rating }], { fogOfWar: true });
    assert.equal(getExactBelief(s, 'white')._prior, flat,
      `rating ${rating} is inert while RATING_SLOPE is zero`);
  }
});
