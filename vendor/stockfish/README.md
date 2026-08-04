# Vendored Stockfish (chess engine)

These are unmodified, prebuilt artifacts copied verbatim (no build step) so the
chess AI ships a strong engine with **no install required**.

- `sf18-lite-single.cjs`  — Stockfish 18 "lite", **single-threaded** WASM build
  (Emscripten loader). Renamed from the upstream `stockfish-18-lite-single.js`
  to `.cjs` only because this repo is `"type":"module"`; the file contents are
  unmodified CommonJS.
- `sf18-lite-single.wasm` — the companion WebAssembly binary (7.3 MB)

**Source:** the npm package [`stockfish@18`](https://www.npmjs.com/package/stockfish)
(`bin/stockfish-18-lite-single.{js,wasm}`) — the
[stockfish.js](https://github.com/nmrugg/stockfish.js) port of
[Stockfish](https://stockfishchess.org/).

**License:** Stockfish is **GPL-3.0**. These files are redistributed under the
GPL. If you distribute this project with these files included, the GPL's terms
apply to that distribution.

Loaded in Node by [`../stockfish.js`](../stockfish.js); used as the
perfect-information move picker and as the Obscuro search's leaf evaluator, with
automatic fallback to the built-in JS search if these files are absent or fail
to load.

## Why this build, of the five upstream ships

Upstream offers large/lite × threaded/single, plus an asm.js fallback.

- **single-threaded**, because the threaded builds need `SharedArrayBuffer` —
  i.e. COOP/COEP headers on every page that loads the engine. Our workload is
  also the wrong shape for threads: thousands of *tiny* depth-1..4 MultiPV
  calls, where more threads inside one search buy far less than more engines
  running side by side would.
- **lite** (7.3 MB), because the full-net build is 113 MB, which is not a
  download to put in front of a browser.

## Replacing Stockfish 11 (2026-08-01)

The previous vendored engine was Stockfish 11 — chosen for being small and
dependency-free, not for strength. Measured on identical midgame positions:

| call | SF11 | SF18-lite | |
|---|---|---|---|
| depth 1, multipv 40 | 3.98 ms | 1.88 ms | 2.1× faster |
| depth 4, multipv 40 | ~9.3 ms | 5.88 ms | ~1.6× faster |
| depth 7, multipv 30 | 24 ms | 23.1 ms | ~equal |

Roughly **2× faster at the depths the search actually uses**, and a far better
evaluator: SF11 predates NNUE, so it ran the classical hand-written evaluation.
Evaluation quality is the single largest lever in the Obscuro paper's own
ablations (their agent beats the same search carrying a crude evaluation 81.9%),
which is what made this the cheapest strength change available.

Two integration details cost real debugging time and are worth knowing before
touching [`sf-worker.cjs`](sf-worker.cjs). Both fail *silently* — as
`available() === false`, which merely drops the AI to its JS fallback:

1. The bundle **refuses to export its factory inside a Node worker thread**: its
   own guard assumes any such thread must be its engine worker, so `require()`
   returns an empty object. We run it in a worker deliberately (respawning is
   the only way to reclaim its WASM heap), so the bridge spoofs
   `worker_threads.isMainThread` for the duration of the `require`.
2. The export is a function that **returns** the Emscripten factory —
   `INIT_ENGINE()(module)`, not `INIT_ENGINE(module)` — and the promise it gives
   back resolves *before* the engine will accept commands, so `_isReady` has to
   be polled. Both mirror the upstream `stockfish/index.js` loader.

`sf-cache.{sqlite,ndjson}` keys are namespaced by engine (`ENGINE_TAG` in
[`../stockfish.js`](../stockfish.js)), so cached SF11 evaluations can never be
served as though this engine had produced them; they simply age out via the LRU.
