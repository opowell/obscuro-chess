// ---------------------------------------------------------------------------
// Loading a corpus of recorded fog games — from a directory, a zip, or a file,
// in session JSON or PGN, with the players' ratings carried through.
//
// WHY THIS IS ONE MODULE AND NOT FOUR COPIES. Four scripts (calibrate-belief,
// fit-move-prior, move-quality, strength-belief) each carried the same fifteen
// lines: readdirSync, filter *.json, JSON.parse in a try, check it is fog chess,
// check it is long enough. They agreed by accident, which meant a corpus that
// one script accepted and another quietly dropped would show up as two harnesses
// disagreeing about the same model. The filter is a property of the CORPUS, so
// it lives with the corpus.
//
// WHAT COUNTS AS A GAME is unchanged from that shared loop and is deliberately
// strict: fog chess only, at least `minPlies` plies. Nothing here silently
// repairs a game. Anything rejected is COUNTED and reported by reason, because
// the number that matters when fitting on someone else's corpus is not how many
// games loaded but how many did not, and why.
//
// RATINGS are the point of the PGN path. Session JSON records who played
// (`agent: "human" | "obscuro"`) but not how well; PGN's `WhiteElo`/`BlackElo`
// are the only widely-recorded measure of opponent strength, and they are what
// makes π fittable per strength band instead of pooled over everyone. A rating
// is per SEAT, not per game — the two players are usually not the same strength,
// and each seat's decisions belong to that seat's band.
// ---------------------------------------------------------------------------

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { openZip } from './zip.js';
import {
  pgnToSessions, parseRating, movesToSession, normalizeMoveList, fogConfig, isFogVariant,
} from './pgn.js';

const SESSION_EXT = new Set(['.json']);
const PGN_EXT = new Set(['.pgn']);

/** Strip one layer of `.gz`, returning `[bytes, nameWithoutGz]`. */
function gunzipIfNeeded(buf, name) {
  if (!name.toLowerCase().endsWith('.gz')) return [buf, name];
  return [gunzipSync(buf), name.slice(0, -3)];
}

/** `{ sess, plies }` for a parsed session document, or `{ reason }`. */
function readSessionDoc(sess, { minPlies, requireFog }) {
  const p = sess?.params;
  if (p?.game !== 'chess') return { reason: 'not a chess session' };
  const c = p.config ?? {};
  if (requireFog && !(c.fogOfWar || c.fog)) return { reason: 'not a fog game' };
  const plies = sess.log?.length ?? 0;
  if (plies < minPlies) return { reason: `only ${plies} plies` };
  return { sess, plies };
}

/**
 * A CRAWL FILE: one JSON holding many games as SAN move lists.
 *
 *   { games: [ { white, black, variant, moves: ["d4","c5",…],
 *                players: [{ username, rating }, …] }, … ] }
 *
 * This is the shape a Chess.com Fog of War archive crawl produces, and the
 * reason the corpus loader dispatches on CONTENT rather than only on extension:
 * it is a `.json` like a session recording, and nothing about the filename says
 * which. Detection is on the `games` array, so any crawler emitting the same
 * fields is read without a second reader.
 *
 * Games are keyed by `gameId` when they carry one — a crawl that walks several
 * players' archives meets the same game from both sides and would otherwise
 * feed it to the fitter twice, which is not more data, it is one game weighted
 * double.
 */
function* gamesFromCrawl(doc, name, opts, stats) {
  const seenIds = stats.seenIds;
  for (const [i, g] of (doc.games ?? []).entries()) {
    if (g?.variant && !opts.allowAnyVariant && !isFogVariant(g.variant)) {
      stats.reject(`variant "${g.variant}" is not fog chess`); continue;
    }
    if (!Array.isArray(g?.moves) || !g.moves.length) { stats.reject('no moves recorded'); continue; }
    const { tokens: moves, dropped } = normalizeMoveList(g.moves);
    if (!moves.length) { stats.reject('no moves recorded'); continue; }
    if (g.gameId != null) {
      if (seenIds.has(g.gameId)) { stats.reject('duplicate gameId'); continue; }
      seenIds.add(g.gameId);
    }

    // The crawl names the seats by username and lists ratings per username, so
    // the two have to be joined rather than read positionally — `players` is in
    // crawl order, not white-then-black.
    const byName = new Map((g.players ?? []).map(p => [p.username, p]));
    const seat = who => {
      const p = byName.get(who);
      return { name: who ?? null, rating: parseRating(p?.rating) };
    };
    const w = seat(g.white), b = seat(g.black);
    const ratings = { white: w.rating, black: b.rating };
    const players = [
      { id: 'white', name: w.name ?? 'White', agent: 'human', rating: w.rating },
      { id: 'black', name: b.name ?? 'Black', agent: 'human', rating: b.rating },
    ];

    const out = movesToSession(moves, { game: opts.game, players, config: fogConfig() });
    if (out.reason) { stats.reject(out.reason); continue; }
    if (out.plies < opts.minPlies) { stats.reject(`only ${out.plies} plies`); continue; }
    // A crawl records how many plies the server saw. If our replay stopped
    // short, we disagree with the source about the game — count it, because a
    // systematic disagreement is a parser bug, not a corpus quirk.
    if (out.skipped) stats.truncated++;
    yield {
      file: `${name}#${g.gameId ?? i}`, sess: out.sess, plies: out.plies,
      ratings, actors: { white: 'human', black: 'human' },
      // The crawl's own ply count INCLUDES any termination glyph, so compare
      // against it net of what normalizeMoveList dropped.
      sourcePlies: typeof g.plies === 'number' ? g.plies - dropped.length : null,
      result: g.result ?? null,
    };
  }
}

/** Ratings and actor types out of a session's own params. */
function metaFromSession(sess) {
  const ratings = { white: null, black: null };
  const actors = { white: '?', black: '?' };
  for (const pl of sess.params?.players ?? []) {
    const id = pl.id ?? pl.playerId;
    if (id !== 'white' && id !== 'black') continue;
    ratings[id] = parseRating(pl.rating ?? pl.elo ?? pl.Elo ?? null);
    actors[id] = pl.type ?? pl.agent ?? '?';
  }
  return { ratings, actors };
}

/**
 * Every game in one file's bytes, whatever kind of file it is.
 *
 * `name` drives the dispatch, so a zip member and a file on disk take exactly
 * the same path — which is the whole reason zip support is ~20 lines here rather
 * than a parallel implementation.
 */
function* gamesFromBuffer(buf, name, opts, stats) {
  const [bytes, base] = gunzipIfNeeded(buf, name);
  const ext = extname(base).toLowerCase();

  if (SESSION_EXT.has(ext)) {
    // A .json is either one recorded session or a crawl holding many games;
    // only the content says which.
    let doc;
    try { doc = JSON.parse(bytes.toString('utf8')); } catch (err) {
      stats.reject(`unparseable JSON (${err.message})`); return;
    }
    if (Array.isArray(doc?.games)) { yield* gamesFromCrawl(doc, name, opts, stats); return; }
    const r = readSessionDoc(doc, opts);
    if (r.reason) { stats.reject(r.reason); return; }
    yield { file: name, sess: r.sess, plies: r.plies, ...metaFromSession(r.sess) };
    return;
  }

  if (PGN_EXT.has(ext)) {
    const rejects = [];
    let i = 0;
    for (const g of pgnToSessions(bytes.toString('utf8'), opts, rejects)) {
      // Rejects accumulate lazily as the generator runs, so drain them as we go
      // rather than after the loop — `maxGames` may stop the iteration early.
      while (i < rejects.length) stats.reject(rejects[i++].reason);
      if (g.plies < opts.minPlies) { stats.reject(`only ${g.plies} plies`); continue; }
      if (g.skipped) stats.truncated++;
      yield {
        file: `${name}#${g.index}`, sess: g.sess, plies: g.plies,
        ratings: g.ratings,
        actors: { white: g.sess.params.players[0].agent, black: g.sess.params.players[1].agent },
      };
    }
    while (i < rejects.length) stats.reject(rejects[i++].reason);
    return;
  }

  stats.ignored++;
}

/** Recursively yield file paths under `dir`, in a stable order. */
function* walk(dir) {
  for (const name of readdirSync(dir).sort()) {
    const full = join(dir, name);
    let st;
    try { st = statSync(full); } catch { continue; }
    if (st.isDirectory()) yield* walk(full);
    else yield full;
  }
}

/**
 * Lazily yield every game in `paths` (a path or an array of them).
 *
 * A path may be a directory (walked recursively), a `.zip` (walked as one), a
 * `.json` session, a `.pgn`, or any of those with a trailing `.gz`. Lazy so that
 * `--max-games 200` on a million-game archive reads two hundred games' worth of
 * bytes and stops, and so peak memory is one member rather than one corpus.
 */
export function* iterCorpus(paths, options = {}) {
  const opts = {
    minPlies: options.minPlies ?? 10,
    requireFog: options.requireFog ?? true,
    allowAnyVariant: options.allowAnyVariant ?? false,
    game: options.game,
  };
  const stats = options.stats ?? makeStats();
  const list = Array.isArray(paths) ? paths : [paths];

  for (const p of list) {
    let st;
    try { st = statSync(p); } catch { throw new Error(`corpus: cannot read ${p}`); }

    if (st.isDirectory()) {
      for (const f of walk(p)) yield* fromPath(f, opts, stats);
    } else {
      yield* fromPath(p, opts, stats);
    }
  }
}

function* fromPath(file, opts, stats) {
  if (file.toLowerCase().endsWith('.zip')) {
    const z = openZip(file);
    try {
      for (const entry of z.entries()) {
        stats.files++;
        yield* gamesFromBuffer(z.read(entry), `${file}!${entry.name}`, opts, stats);
      }
    } finally { z.close(); }
    return;
  }
  stats.files++;
  yield* gamesFromBuffer(readFileSync(file), file, opts, stats);
}

/** The rejection ledger `iterCorpus` fills in. */
export function makeStats() {
  const reasons = new Map();
  return {
    files: 0, ignored: 0, truncated: 0, rejected: 0, reasons,
    // Corpus-wide, so the same game met in two crawl files counts once.
    seenIds: new Set(),
    reject(reason) {
      this.rejected++;
      // Collapse the numeric tail ("only 4 plies") so the summary has a handful
      // of lines rather than one per game.
      const key = reason.replace(/\d+/g, 'N');
      reasons.set(key, (reasons.get(key) ?? 0) + 1);
    },
  };
}

/**
 * Load a corpus eagerly, honouring `maxGames`.
 *
 * Returns `{ games, stats }` where each game is
 * `{ file, sess, plies, ratings: {white, black}, actors: {white, black} }` —
 * `sess` being session-shaped whatever it was read from, so every existing
 * consumer (`replayBelief`, the fitter's replay) takes it unchanged.
 */
export function loadCorpus(paths, options = {}) {
  const stats = makeStats();
  const maxGames = options.maxGames ?? Infinity;
  const games = [];
  for (const g of iterCorpus(paths, { ...options, stats })) {
    games.push(g);
    if (games.length >= maxGames) break;
  }
  return { games, stats };
}

/** A one-line summary of what loaded and what did not. */
export function describeCorpus(games, stats) {
  const rated = games.filter(g => g.ratings.white != null || g.ratings.black != null).length;
  const bits = [`${games.length} fog games from ${stats.files} file(s)`];
  if (rated) bits.push(`${rated} with ratings`);
  if (stats.rejected) {
    const top = [...stats.reasons].sort((a, b) => b[1] - a[1]).slice(0, 4)
      .map(([r, n]) => `${n}× ${r}`).join(', ');
    bits.push(`${stats.rejected} rejected (${top})`);
  }
  if (stats.truncated) bits.push(`${stats.truncated} truncated at an unparseable move`);
  if (stats.ignored) bits.push(`${stats.ignored} non-corpus files ignored`);
  return bits.join('; ');
}

/**
 * Every rating in the corpus, sorted — for reporting what population a fit was
 * actually estimated over, which is the caveat on any rating-conditioned result.
 */
export function ratingSpread(games) {
  const rs = games.flatMap(g => [g.ratings.white, g.ratings.black])
    .filter(r => r != null).sort((a, b) => a - b);
  if (!rs.length) return null;
  const q = p => rs[Math.min(rs.length - 1, Math.floor(rs.length * p))];
  return { n: rs.length, min: rs[0], p10: q(0.1), median: q(0.5), p90: q(0.9), max: rs[rs.length - 1] };
}
