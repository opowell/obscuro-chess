// ---------------------------------------------------------------------------
// Worker-thread bridge for the vendored Stockfish WASM.
//
// The WASM engine accrues heap memory across searches and eventually aborts with
// "memory access out of bounds". Running it inside a worker lets the main thread
// terminate() and respawn it to reclaim memory wholesale — an in-process reload
// cannot, because a fresh instance shares the same growing linear memory.
//
// Protocol (plain strings, structured-clone-cheap):
//   main → worker : a UCI command line to forward to the engine
//   worker → main : each engine output line, verbatim; or "__error__ <msg>" if
//                   the engine could not be constructed.
//
// STOCKFISH 18 CHANGED THE EMBEDDING API. The old SF11 build was constructed as
// `STOCKFISH(wasmPath)` and spoke `postMessage`/`onmessage`; the 18 bundle is a
// plain Emscripten module factory:
//
//   • it takes a Module-ish object and returns a promise of that same object;
//   • `locateFile` says where the .wasm is — REQUIRED here, because the bundle's
//     own auto-derivation rewrites a trailing `.js`, and this engine is vendored
//     as `.cjs` to opt out of the repo's ESM default;
//   • output arrives through `Module.listener` (the bundle routes both print and
//     printErr there), NOT through an `onmessage` handler;
//   • commands go in via `ccall('command', …)`, with `async: true` for `go` so a
//     running search doesn't block the thread that has to stream its output.
//
// Commands that arrive before instantiation finishes are queued: the main
// thread's handshake fires `uci` and `isready` the instant the worker exists.
// ---------------------------------------------------------------------------
const { parentPort } = require('worker_threads');
const path = require('path');

const HERE = __dirname;
const JS_PATH = path.join(HERE, 'sf18-lite-single.cjs');
const WASM_PATH = path.join(HERE, 'sf18-lite-single.wasm');

// The Emscripten loader mistakes a defined `fetch` for a browser and tries to
// fetch the .wasm as a URL. WASM instantiation is async, so we must keep `fetch`
// hidden for the whole load — not just around the constructor call. This worker
// never needs fetch, so we simply drop it for the thread's lifetime.
globalThis.fetch = undefined;

// …and the bundle refuses to export its factory at all when it believes it is
// itself an engine worker. Its guard is literally
//
//   … || (typeof global !== 'undefined' && isProcess && !require('worker_threads').isMainThread) || (…export…)
//
// so being inside a worker_thread short-circuits the chain and `require()`
// returns an empty object — which is exactly where we run it. The bundle's
// assumption is that a Node worker must be ITS worker, wired to its own
// messaging; ours is a worker we manage for memory recycling. Spoofing the flag
// for the duration of the require puts us back on the export path, and nothing
// downstream reads it: this file talks to `parentPort` directly.
const wt = require('worker_threads');
const realIsMain = wt.isMainThread;
try { Object.defineProperty(wt, 'isMainThread', { value: true, configurable: true }); } catch { /* fall through */ }

const queued = [];
let engine = null;

function send(cmd) {
  // Mirrors the upstream loader: dispatch off the current tick, and run `go`
  // asyncified so `info`/`bestmove` stream out while the search is in progress.
  setImmediate(() => {
    try { engine.ccall('command', null, ['string'], [cmd], { async: /^go\b/.test(cmd) }); }
    catch { /* engine dying; ignore */ }
  });
}

try {
  const INIT_ENGINE = require(JS_PATH);
  try { Object.defineProperty(wt, 'isMainThread', { value: realIsMain, configurable: true }); } catch { /* ignore */ }
  if (typeof INIT_ENGINE !== 'function') throw new Error('engine bundle exported no factory');

  // Two calls, not one: the export is a function that RETURNS the Emscripten
  // module factory, and only the inner call takes the Module object. Then
  // `_isReady` has to be polled — the promise resolves when the wasm is
  // instantiated, which is before the engine will accept commands. Both of these
  // mirror the upstream loader (stockfish/index.js); getting either wrong fails
  // silently as "engine not available" and drops the AI to the JS fallback.
  const mod = {
    locateFile: (p) => (p.indexOf('.wasm') > -1 ? WASM_PATH : JS_PATH),
    listener: (line) => parentPort.postMessage(String(line == null ? '' : line)),
  };
  INIT_ENGINE()(mod).then(function whenReady() {
    if (mod._isReady && !mod._isReady()) return setTimeout(whenReady, 10);
    engine = mod;
    while (queued.length) send(queued.shift());
  }).catch((e) => {
    parentPort.postMessage('__error__ ' + (e && e.message ? e.message : e));
  });
} catch (e) {
  parentPort.postMessage('__error__ ' + (e && e.message ? e.message : e));
}

parentPort.on('message', (cmd) => {
  if (engine) send(cmd); else queued.push(cmd);
});
