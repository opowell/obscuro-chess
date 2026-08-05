// ---------------------------------------------------------------------------
// Stockfish backend — a standalone, vendored UCI engine (no install required).
//
// We bundle Stockfish 18 ("lite", single-threaded WASM) under ../vendor/stockfish.
// It is the strong evaluator the Obscuro subgame scores its leaves with. Everything
// degrades gracefully: if the vendored files are missing or the engine fails to
// load, `available()` returns false and the agents fall back to the JS search.
//
// WHY THIS BUILD, of the five upstream ships. Measured on identical midgame
// positions against the Stockfish 11 build this replaced:
//
//   depth 1, multipv 40   3.98 ms → 1.88 ms      depth 7, multipv 30  24 ms → 23 ms
//   depth 4, multipv 40   ~9.3 ms → 5.88 ms      load                 87 ms
//
// ~2× faster at the depths the search actually uses, and a far better evaluator:
// SF11 predates NNUE, so it was the classical hand-written eval. Evaluation
// quality is the largest single lever in the Obscuro paper's own ablations
// (their agent beats the same search with a crude eval 81.9%), which makes this
// the cheapest strength change available to us.
//   • "single" (not the threaded build) because the threaded ones need
//     SharedArrayBuffer, i.e. COOP/COEP headers on every page that loads them —
//     and our workload is thousands of TINY depth-1..4 calls, where more threads
//     per search buy far less than more engines would.
//   • "lite" (7.3 MB) because the full-net build is 113 MB, which is not a
//     download to put in front of a browser.
//
// Runs identically in Node and in the browser — same UCI protocol, same
// multiPV/bestMove/evaluate API — just a different transport underneath:
//   - Node: the engine lives in a worker thread (vendor/stockfish/sf-worker.cjs),
//     a thin bridge that requires the vendored CommonJS loader directly.
//   - Browser: vendor/stockfish/sf18-lite-single.cjs is ALSO the upstream browser
//     build (github.com/nmrugg/stockfish.js) — loaded as a classic (non-module)
//     Worker, it self-detects `importScripts` and bootstraps its own
//     onmessage/postMessage UCI bridge, fetching the .wasm named in its URL
//     hash. Both files are plain static assets, so serving this package's
//     directory byte-for-byte over HTTP is enough to reach the browser — no
//     build step, no bundler. The embedder can then run this module inside a
//     Worker of its own (battle-simulator's analysis worker does exactly that).
// Either way the engine is torn down and respawned periodically (maybeRecycle)
// rather than reused indefinitely: the WASM heap grows with use and eventually
// aborts with "memory access out of bounds", and only tearing down the whole
// worker reclaims that memory (a fresh in-process instance shares the same
// linear memory).
// ---------------------------------------------------------------------------

import { toFEN, uciToAction } from './fen.js';
import { param, ramp } from './config.js';

// This module is imported both server-side (Node) and inside the browser
// analysis Web Worker (apps/design/analysis-worker.js), which pulls in the whole
// chess graph. Node's worker_threads/url/path/fs don't exist in a browser, so
// they're loaded lazily behind an isNode guard.
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
const isBrowser = !isNode && typeof Worker !== 'undefined';
let Worker_, fileURLToPath, path, fs;
if (isNode) {
  ({ Worker: Worker_ } = await import('worker_threads'));
  ({ fileURLToPath } = await import('url'));
  path = (await import('path')).default;
  fs = (await import('fs')).default;
}

const HERE = isNode ? path.dirname(fileURLToPath(import.meta.url)) : '';
// The engine runs inside this worker (see sf-worker.cjs) so it can be terminated
// and respawned to reclaim WASM memory. Both are .cjs to opt out of the repo's
// ESM default and match the vendored CommonJS loader.
const VENDOR_DIR = isNode ? path.join(HERE, '..', 'vendor', 'stockfish') : '';
const WORKER_PATH = isNode ? path.join(VENDOR_DIR, 'sf-worker.cjs') : '';
const WASM_PATH = isNode ? path.join(VENDOR_DIR, 'sf18-lite-single.wasm') : '';
// Sibling URL to this module's own — resolves correctly however this file was
// itself fetched (mounted under a base path, served from /lib/, etc.).
//
// The engine bundle finds its own .wasm from `location.hash` when one is given,
// and otherwise by rewriting a trailing `.js` on its own URL. Ours is vendored
// as `.cjs` (to opt out of the repo's ESM default), so that rewrite would miss —
// hence the explicit, encoded wasm URL in the hash.
const BROWSER_ENGINE_URL = isBrowser
  ? new URL('../vendor/stockfish/sf18-lite-single.cjs', import.meta.url).href
    + '#' + encodeURIComponent(new URL('../vendor/stockfish/sf18-lite-single.wasm', import.meta.url).href)
  : '';

// Bumped whenever the engine changes. The cache stores evaluations keyed by
// position, and SF11's numbers are not SF18's numbers — without this tag the
// 23 MB of cached SF11 output would be served as if this engine had produced it.
// Old rows simply stop matching and age out through the existing LRU.
// Bumped 2026-08-02 with the MultiPV depth-mixing fix below: the cached rows are
// keyed by (fen, multipv, depth) but their CONTENT came from the buggy parser, so
// they have to be invalidated like an engine change. Bump this for any change
// that alters what a given query returns, not just for a new engine.
const ENGINE_TAG = 'sf18l-mpv3';

let worker = null;
let readyPromise = null;
let listeners = [];          // line handlers currently attached to the engine output
let queue = Promise.resolve(); // serialises searches (UCI is single-threaded/stateful)

// The vendored Stockfish WASM accrues heap memory across searches and, in a
// long-lived process, eventually aborts with "memory access out of bounds". An
// in-process reload cannot reclaim it — a fresh instance shares the same linear
// memory — so the engine lives in a worker thread that we terminate + respawn:
// proactively every RECYCLE_AFTER searches (below the observed failure point),
// and reactively if it ever does abort (the worker dies, this process does not).
let callsSinceLoad = 0;
// Exported so src/settings.js can list it alongside every
// other fog-chess default.
export const RECYCLE_AFTER = 400;
const recycleAfter = () => param('chess.RECYCLE_AFTER', RECYCLE_AFTER);

// Abort callbacks for in-flight requests. When the worker dies mid-search we
// call these to resolve each pending request as null immediately, rather than
// leaving it to wait out its (multi-second) timeout.
const pending = new Set();
function failAllPending() { for (const abort of [...pending]) abort(); }

// ---------------------------------------------------------------------------
// Disk-backed LRU cache for multiPV results. multiPV is deterministic given
// (fen, depth, multipv) so results are safe to cache across turns and games.
// bestMove uses movetime (non-deterministic) so is intentionally not cached.
//
// Uses node:sqlite (Node >= 22.5) when available: O(1) reads/writes, proper
// LRU via a monotonic sequence column, no compaction needed.
// Falls back to append-only NDJSON on older Node: each entry is one JSON
// line "[key,value]\n" — a single append is atomic so a killed process can't
// corrupt existing lines. Duplicates are compacted on startup when >50% stale.
// ---------------------------------------------------------------------------
let DatabaseSync = null;
if (isNode) { try { ({ DatabaseSync } = await import('node:sqlite')); } catch {} }

export const CACHE_MAX = 20_000;
const cacheMax = () => param('chess.CACHE_MAX', CACHE_MAX);

// WHERE the cache lives. It defaults next to the engine, but it is derived data,
// not part of the package: an embedder that already has a warm cache (a few tens
// of MB of it, in battle-simulator's case) keeps it in its OWN checkout and points
// us at it, so a `git pull` of this repo never carries someone else's evaluations
// and this repo never has to ship them. Point us at it with `chess.SF_CACHE_DIR`
// in a settings file, the SF_CACHE_DIR env var, or a setCacheDir() call — in
// that order of increasing precedence — before the first search. After the
// cache is open, changing it is ignored, because the entries already read are
// the ones the process is answering from.
export const SF_CACHE_DIR = isNode ? path.join(HERE, '..', 'vendor', 'stockfish') : '';
let explicitCacheDir = '';
export function setCacheDir(dir) {
  if (!isNode || db || sfCache) return;         // already open (or browser) — no-op
  explicitCacheDir = dir instanceof URL ? fileURLToPath(dir) : dir;
}
const cacheDir = () =>
  explicitCacheDir || param('chess.SF_CACHE_DIR', (isNode && process.env.SF_CACHE_DIR) || SF_CACHE_DIR);
const cachePath = (ext) => path.join(cacheDir(), `sf-cache.${ext}`);

// SQLite state (used when DatabaseSync is available)
let db = null, stmtGet, stmtSet, stmtTouch, stmtEvict;
let sqliteSize = 0, lruSeq = 0;

// NDJSON fallback state
let sfCache = null;

function loadCache() {
  if (!isNode) { if (!sfCache) sfCache = new Map(); return; } // browser: in-memory only, no disk
  if (DatabaseSync && !sfCache) {
    if (db) return;
    // Concurrent processes (e.g. parallel test files) share this DB; a busy
    // lock can throw anywhere in here. Never leave `db` half-initialised —
    // fall back to the in-memory/NDJSON path instead of crashing callers.
    try {
      db = new DatabaseSync(cachePath('sqlite'));
      db.exec(`CREATE TABLE IF NOT EXISTS cache (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        lru   INTEGER NOT NULL DEFAULT 0
      )`);
      db.exec('CREATE INDEX IF NOT EXISTS cache_lru ON cache(lru)');
      stmtGet   = db.prepare('SELECT value FROM cache WHERE key = ?');
      stmtSet   = db.prepare('INSERT OR REPLACE INTO cache(key, value, lru) VALUES(?, ?, ?)');
      stmtTouch = db.prepare('UPDATE cache SET lru = ? WHERE key = ?');
      stmtEvict = db.prepare('DELETE FROM cache WHERE key = (SELECT key FROM cache ORDER BY lru LIMIT 1)');
      const row = db.prepare('SELECT COUNT(*) as n, COALESCE(MAX(lru), 0) as m FROM cache').get();
      sqliteSize = row.n; lruSeq = row.m;
      return;
    } catch {
      try { db?.close(); } catch { /* ignore */ }
      db = null; // degrade to the NDJSON/in-memory path below
    }
  }
  {
    if (sfCache) return;
    sfCache = new Map();
    let lineCount = 0;
    try {
      for (const line of fs.readFileSync(cachePath('ndjson'), 'utf8').split('\n')) {
        if (!line) continue;
        try { const [k, v] = JSON.parse(line); sfCache.delete(k); sfCache.set(k, v); lineCount++; }
        catch { /* corrupt line — skip */ }
      }
    } catch { /* missing — start fresh */ }
    while (sfCache.size > cacheMax()) sfCache.delete(sfCache.keys().next().value);
    if (lineCount > sfCache.size * 1.5) compactNdjson();
  }
}

function compactNdjson() {
  try { fs.writeFileSync(cachePath('ndjson'), [...sfCache.entries()].map(e => JSON.stringify(e)).join('\n') + '\n'); }
  catch { /* ignore */ }
}

// Every cache key is namespaced by ENGINE_TAG here, in ONE place, so no call
// site can forget and read another engine's numbers back as this one's.
function cacheGet(rawKey) {
  const key = ENGINE_TAG + '|' + rawKey;
  if (db) {
    // A concurrent writer can make any sqlite op throw (SQLITE_BUSY); a cache
    // miss is always an acceptable answer.
    try {
      const row = stmtGet.get(key);
      if (!row) return undefined;
      stmtTouch.run(++lruSeq, key);
      return JSON.parse(row.value);
    } catch { return undefined; }
  }
  if (!sfCache) return undefined;
  const v = sfCache.get(key);
  if (v === undefined) return undefined;
  sfCache.delete(key); sfCache.set(key, v); // move to end (LRU)
  return v;
}

function cacheSet(rawKey, value) {
  const key = ENGINE_TAG + '|' + rawKey;
  if (db) {
    try {
      const isNew = !stmtGet.get(key);
      if (isNew && sqliteSize >= cacheMax()) { stmtEvict.run(); sqliteSize--; }
      stmtSet.run(key, JSON.stringify(value), ++lruSeq);
      if (isNew) sqliteSize++;
    } catch { /* busy — skip this write */ }
  } else if (sfCache) {
    if (sfCache.size >= cacheMax() && !sfCache.has(key)) sfCache.delete(sfCache.keys().next().value);
    sfCache.delete(key); sfCache.set(key, value);
    try { fs.appendFileSync(cachePath('ndjson'), JSON.stringify([key, value]) + '\n'); } catch { /* ignore */ }
  }
}

function send(cmd) { if (worker) worker.postMessage(cmd); }

// Spawn the engine worker and hand-shake it. Resolves true once usable.
// Node and browser share everything past this point (send/request/multiPV/…);
// only how a line handler gets attached and how the worker is constructed differ.
function init() {
  if (readyPromise) return readyPromise;
  readyPromise = new Promise((resolve) => {
    if (!isNode && !isBrowser) return resolve(false); // neither Node nor a real Worker ctor available

    let w;
    if (isNode) {
      if (!fs.existsSync(WORKER_PATH) || !fs.existsSync(WASM_PATH)) return resolve(false);
      try { w = new Worker_(WORKER_PATH); } catch { return resolve(false); }
    } else {
      // Classic (non-module) Worker: vendor/sf18-lite-single.cjs self-bootstraps
      // its own UCI onmessage/postMessage bridge when it detects it's running as
      // a Worker (typeof importScripts === 'function') — see the file's own
      // tail. It reads the .wasm URL out of its own location.hash, which
      // BROWSER_ENGINE_URL supplies (the bundle's fallback of rewriting a
      // trailing `.js` cannot work on a file vendored as `.cjs`).
      try { w = new Worker(BROWSER_ENGINE_URL); } catch { return resolve(false); }
    }
    worker = w;

    let settled = false;
    const finish = (ok) => { if (settled) return; settled = true; resolve(ok); };

    const onReady = (line) => {
      if (line.startsWith('readyok')) { listeners = listeners.filter(x => x !== onReady); finish(true); }
    };
    listeners.push(onReady);

    const onLine = (line) => {
      if (typeof line !== 'string') return;
      if (line.startsWith('__error__')) { finish(false); return; } // engine failed to construct (Node bridge only)
      for (const l of [...listeners]) l(line);
    };
    // An abort inside the WASM (the "memory access out of bounds" fault) kills
    // the worker — surfaced here as an error/exit — but not this process. Drop
    // the dead worker so the next call respawns a fresh one; any in-flight
    // request falls through to its timeout.
    const die = () => {
      finish(false); // no-op once ready; fails the handshake if still loading
      if (worker === w) { worker = null; readyPromise = null; }
      failAllPending(); // an in-flight search will never complete now
    };
    if (isNode) {
      w.on('message', onLine);
      w.on('error', die);
      w.on('exit', die);
    } else {
      w.onmessage = (e) => onLine(e.data);
      w.onerror = die;
    }

    send('uci'); send('isready');
    setTimeout(() => finish(false), 8000); // load watchdog
  });
  return readyPromise;
}

/** Whether a usable Stockfish engine is loaded (async, memoised). */
export function available() { return init(); }

/** Best-effort shutdown (used by tests so the process can exit cleanly). */
export function quit() {
  const w = worker;
  worker = null; readyPromise = null; listeners = [];
  failAllPending();
  if (w) {
    if (isNode) { w.removeAllListeners(); w.terminate().catch(() => {}); }
    else w.terminate();
  }
}

// Terminate the worker and spawn a fresh one once the search budget is spent, so
// WASM heap growth never reaches the abort. Runs inside the serialised queue
// (between requests), so no search is ever interrupted. Terminating a whole
// worker (vs. reloading in-process) is what actually reclaims the memory.
async function maybeRecycle() {
  if (callsSinceLoad < recycleAfter()) return;
  callsSinceLoad = 0;
  const old = worker;
  worker = null; readyPromise = null; listeners = [];
  if (old) {
    if (isNode) { old.removeAllListeners(); try { await old.terminate(); } catch { /* ignore */ } }
    else old.terminate();
  }
  await init();
}

// How often an in-flight search re-checks its caller's `isCancelled` (see below).
export const STOP_POLL_MS = 100;
const stopPollMs = () => param('chess.STOP_POLL_MS', STOP_POLL_MS);

// Run one UCI request, collecting lines until `isDone(line)` returns a result.
// Serialised behind `queue` so only one search runs at a time.
//
// `opts.isCancelled` lets a caller interrupt a search that is already running.
// Cancellation used to take effect only BETWEEN calls, which was fine when every
// call was a sub-second shallow search; on the iterative-deepening ladder a
// single deep rung can run for tens of seconds, so a position change or an
// expired move clock would otherwise be felt a whole rung late. Sending the UCI
// `stop` command makes the engine emit its current-best `bestmove` immediately —
// which the line handler already resolves on, so there is no new resolution
// path. `opts.onStopped` fires when that happens, so the caller can tell a
// truncated result (shallower than the depth it asked for) from a complete one.
function request(commands, isDone, timeoutMs, { isCancelled, onStopped } = {}) {
  const run = async () => {
    await maybeRecycle();
    if (!(await init())) return null; // ensure a live worker (respawns if it crashed)
    callsSinceLoad++;
    return new Promise((resolve) => {
      let settled = false;
      let timer, poll;
      const done = (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        clearInterval(poll);
        listeners = listeners.filter(x => x !== handler);
        pending.delete(abort);
        resolve(result);
      };
      const handler = (line) => { const r = isDone(line); if (r !== undefined) done(r); };
      const abort = () => done(null); // worker died — give up now, don't wait for the timeout
      listeners.push(handler);
      pending.add(abort);
      try { for (const c of commands) send(c); } catch { /* fall through to timeout */ }
      if (isCancelled) {
        poll = setInterval(() => {
          if (settled || !isCancelled()) return;
          clearInterval(poll); poll = null;
          onStopped?.();
          try { send('stop'); } catch { /* engine gone — the timeout still fires */ }
        }, stopPollMs());
      }
      timer = setTimeout(() => done(null), timeoutMs);
    });
  };
  const p = queue.then(run);
  queue = p.catch(() => {});
  return p;
}

/**
 * Best move for a FEN, as a UCI string (e.g. "e2e4"), or null.
 * @param {string} fen
 * @param {{movetime?:number, skill?:number|null}} [opts]
 */
export async function bestMove(fen, { movetime = 300, skill = null } = {}) {
  if (!(await init())) return null;
  const cmds = ['setoption name MultiPV value 1'];
  if (skill != null) cmds.push(`setoption name Skill Level value ${skill}`);
  cmds.push('position fen ' + fen, 'go movetime ' + movetime);
  const uci = await request(cmds, line => (line.startsWith('bestmove') ? (line.split(/\s+/)[1] || null) : undefined), movetime + 5000);
  return uci && uci !== '(none)' ? uci : null;
}

/**
 * Static-ish evaluation of a FEN in centipawns from the side-to-move's view, or
 * null. (Exposed for future use as a leaf evaluator; the agents currently use
 * bestMove directly.)
 */
export async function evaluate(fen, { movetime = 100 } = {}) {
  if (!(await init())) return null;
  let last = null;
  const val = await request(
    ['setoption name MultiPV value 1', 'position fen ' + fen, 'go movetime ' + movetime],
    (line) => {
      const m = line.match(/score (cp|mate) (-?\d+)/);
      if (m) last = m[1] === 'cp' ? Number(m[2]) : (m[2] > 0 ? 100000 - Number(m[2]) : -100000 - Number(m[2]));
      return line.startsWith('bestmove') ? last : undefined;
    },
    movetime + 5000,
  );
  return val;
}

/**
 * Evaluate the top `multipv` moves of a position in a single call — the paper's
 * batched node heuristic ("MultiPV at low depth gives evaluations for all
 * children at once"). Returns [{ move, cp }] with scores from the side-to-move's
 * perspective, or null. Used to score the fog subgame's leaves cheaply.
 *
 * `onInfo({ depth, candidates })` is optional and fires once per completed
 * iterative-deepening depth (as Stockfish's own `info depth N ...` lines
 * arrive, keyed off the multipv-1 line so it's one tick per depth rather than
 * one per multipv slot), letting a caller show live search progress the way
 * lichess does. Purely a side channel — the awaited return value is unchanged.
 *
 * `isCancelled` interrupts an in-flight search (see request above): the engine
 * returns whatever it had reached instead of running the depth out. A result cut
 * short that way is NOT cached — it is shallower than the `depth` its cache key
 * claims — and `onStopped` fires so the caller can treat it as truncated.
 */
export async function multiPV(fen, { multipv = 10, depth = 2, onInfo, isCancelled, onStopped } = {}) {
  if (!(await init())) return null;
  loadCache();
  const key = `${fen}|${multipv}|${depth}`;
  const cached = cacheGet(key);
  if (cached !== undefined) return cached;

  // Results are keyed BY MOVE, per depth — not by MultiPV index across depths.
  //
  // `go depth N` sweeps depths 1..N and re-emits all MultiPV lines at each, and
  // the ranking REORDERS between sweeps. Keying by index and letting sweeps
  // overwrite each other therefore mixed depths and, whenever a move changed
  // rank, left the same move under two indices while another move disappeared
  // entirely. The caller saw a full-length result and had no way to notice one of
  // its moves was missing — in the Obscuro leaf evaluator that silently dropped
  // ~10% of children onto the static evaluator (ObscuroAgent `leafStats`).
  //
  // So: accumulate the current sweep by move, and keep the previous complete
  // sweep to fill any gap if the deepest one is cut short. Deeper values win;
  // coverage never regresses.
  let curDepth = 0;
  let cur = new Map();   // move -> cp, current depth sweep
  let prev = new Map();  // move -> cp, last depth sweep that finished
  const merged = () => {
    const out = new Map(prev);
    for (const [m, cp] of cur) out.set(m, cp);
    return [...out].map(([move, cp]) => ({ move, cp }));
  };
  let lastReportedDepth = 0;
  let stopped = false;
  const cmds = [`setoption name MultiPV value ${multipv}`, 'position fen ' + fen, 'go depth ' + depth];
  const result = await request(
    cmds,
    (line) => {
      const dm = line.match(/^info depth (\d+)/);
      const mpv = line.match(/ multipv (\d+) /);
      const sc = line.match(/ score (cp|mate) (-?\d+)/);
      const pv = line.match(/ pv (\S+)/);
      if (mpv && sc && pv) {
        const d = dm ? Number(dm[1]) : curDepth;
        if (d > curDepth) { if (cur.size) prev = cur; cur = new Map(); curDepth = d; }
        const cp = sc[1] === 'cp'
          ? Number(sc[2])
          : (Number(sc[2]) > 0 ? 100000 - Number(sc[2]) : -100000 - Number(sc[2]));
        cur.set(pv[1], cp);
      }
      if (onInfo && dm && mpv?.[1] === '1') {
        const d = Number(dm[1]);
        if (d !== lastReportedDepth) { lastReportedDepth = d; onInfo({ depth: d, candidates: merged() }); }
      }
      return line.startsWith('bestmove') ? merged() : undefined;
    },
    depth * 400 + 5000,
    { isCancelled, onStopped: () => { stopped = true; onStopped?.(); } },
  );
  if (result !== null && !stopped) cacheSet(key, result);
  return result;
}

// Difficulty is a 0–100 number (0 = weakest, 100 = strongest). Legacy string
// tiers are mapped onto the scale so old saved sessions keep working.
export const LEGACY_DIFFICULTY = { easy: 10, medium: 35, hard: 65, expert: 90 };

// What "no difficulty given" means. This used to be answered three different
// ways — 25 here, 25 in FogChess.createInitialState, 50 in the generic agent's
// _config — so an observation that had lost its difficulty played at a
// different strength depending on which code path asked. One number now, read
// by all three.
export const DEFAULT_DIFFICULTY = 25;

export function difficultyToNumber(difficulty) {
  const fallback = param('chess.DEFAULT_DIFFICULTY', DEFAULT_DIFFICULTY);
  const legacy = param('chess.LEGACY_DIFFICULTY', LEGACY_DIFFICULTY);
  const n = typeof difficulty === 'number' ? difficulty : (legacy[difficulty] ?? fallback);
  return n < 0 ? 0 : n > 100 ? 100 : n;
}

// Map difficulty (0–100) to engine strength (Skill Level 0–20) and time per
// move, for the perfect-information / legacy agent path.
export const SF_DIFFICULTY_RAMP = {
  movetimeMs: { min: 50, max: 1000, curve: 'linear' },
  skill: { min: 0, max: 20, curve: 'linear' },
};
export function sfOptsForDifficulty(difficulty) {
  const t = difficultyToNumber(difficulty) / 100;
  const { movetimeMs, skill } = param('chess.SF_DIFFICULTY_RAMP', SF_DIFFICULTY_RAMP);
  return { movetime: ramp(movetimeMs, t), skill: ramp(skill, t) };
}

/**
 * Pick the best action for a *fully observed* position using Stockfish, or null
 * if the engine is unavailable / the move can't be mapped. Only valid with
 * perfect information (the board must be complete — never call under fog).
 */
export async function stockfishBestAction(state, legalActions, opts = {}) {
  if (!(await init())) return null;
  const us = state.activePlayers[0];
  const fen = toFEN(state.board, state.gameSpecific, us === 'white' ? 'w' : 'b', state.turnNumber ?? 1);
  const uci = await bestMove(fen, opts);
  return uci ? uciToAction(uci, legalActions) : null;
}
