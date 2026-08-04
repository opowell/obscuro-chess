# Obscuro analysis: process the whole belief population, not a sample — plan doc

Standalone working doc. Written 2026-07-19, substantially revised 2026-07-19
after re-checking against the code. Safe to read cold in a new session.
Companion to [FOG-AI-FIX-PLAN.md](FOG-AI-FIX-PLAN.md) (search-side king-safety —
already fixed, unrelated to this doc) and `agents/LEARNED-EVAL-PLAN.md`.

## The ask (restated)

Stop *sampling* belief worlds and instead **eventually process the whole
population** of them — walking it in **batches** and updating the analysis
panel's ranked moves + evals every ~second, until *every* consistent position
has been evaluated ("exhaustive"). Most positions have a population too large
to finish, so in practice the list keeps refining forever; but when the
population is small the answer should actually *converge and stop*. The same
"enumerate rather than sample" idea should extend to the other sampled
quantities where feasible. Side goal: run the analysis **client-side in a Web
Worker** so it never blocks the server or the UI.

## Status: Design B (batched enumeration) LANDED 2026-07-20 (server-side)

Shipped and tested (`test/analysis.test.js`):

- `ExactBelief.positionsAt(indices)` + `ChessGame.beliefPopulation` /
  `ChessGame.enumerateWorlds` — walk the materialized set P once, without
  replacement.
- `analyzeObscuroProgressive` rewritten from resample-with-replacement into a
  one-time shuffled cursor over the population with world-count-weighted
  aggregation (exact evals) and an `{evaluated,total,exhaustive}` signal;
  `cpParticles` folded into the same batch; exhaustion ends the walk.
- Prompt cancellation for the memory-leak case: `runObscuroSearch` now honors
  `cfg.isCancelled` (breaks the round loop AND the final-CFR loop), threaded
  through `obscuroStrategy`; the progressive loop also checks between and within
  batches. Client `EventSource.close()` → server `req.on('close')` → `closed` →
  `isCancelled()` is the live wiring (api-server.js `handleAnalyzeStream`).
- Panel (`AnalysisPanel.vue`) shows "N / M worlds" and a settled "All M worlds
  evaluated" done-state; the raw percentage was dropped (floors to 0% for the
  whole walk against a 17k population).

Still Design A (joint-equilibrium mixing) NOT attempted (remains as written
below). **Web Worker port LANDED 2026-07-20** — see next section.

## Status: Web Worker port LANDED 2026-07-20

The Obscuro fog analysis now runs in a browser Web Worker
(`battle-simulator's apps/design/analysis-worker.js`), off both the server and the UI thread.
Verified end-to-end through the real app (fog session → panel rendered ranked
candidates from the worker) and directly via CDP (belief+CFR in-browser, cp from
the server, refining frames). Pieces:

- **`src/stockfish.js` is browser-safe** — its Node imports
  (worker_threads/url/path/fs) are now lazy behind an `isNode` guard, so the
  whole chess+CFR graph imports in a browser worker with `available()` → false.
  The Node server path is unchanged (still uses the worker-thread engine +
  disk cache). This was the one edit to load-bearing shared code; keep the
  guards if touching that module.
- **`vendor/obscuro/src/search.js` yield is cross-env** — `setImmediate` (Node)
  ↦ `setTimeout(0)` (browser) via a `yieldToLoop` helper. Same macrotask
  boundary the starvation fix needs, now in both runtimes. (Same fallback added
  to `vendor/obscuro/src/ObscuroAgent.js`.)
- **`/lib/*` static route** (battle-simulator's `api-server.js` serveLibModule) serves repo ES
  modules to the worker (JS/WASM only, path-escape-guarded). The worker imports
  `…/lib/…` on the embedder's module route (relative, so a mounted base path resolves too).
- **`POST /sessions/:id/cp-eval`** (handleCpEval → exported `cpSumsOverWorlds`)
  returns the batched Stockfish leaf-eval for a batch of belief worlds. The
  worker runs the ~2s CFR locally (measured: CFR ≈ 2000ms/batch, cp ≈ 32ms) but
  has no browser Stockfish, so it POSTs each batch's worlds here.
- **The worker** fetches the fog view (`GET /sessions/:id/state?player=`),
  derives legal moves locally (same as the server's resolveAnalysisContext),
  and runs the exported `analyzeObscuroProgressive` with an `opts.cpEval`
  override (server default = local Stockfish; worker = the /cp-eval fetch).
  Belief is computed locally — for a human, the server also derives it fresh
  from the current view (belief isn't maintained per-ply), so local == server.
- **`AnalysisPanel.vue`** uses the worker only for live fog + Obscuro
  (`props.fog && agent==='obscuro' && ply==null`); non-fog, historical ply, and
  other engines keep the server SSE path. One reused worker; a new position/
  engine posts `{type:'cancel'}` (checked at batch boundaries) then re-analyzes;
  unmount terminates it.

Not ported: browser-WASM Stockfish (cp stays a cheap server round-trip — a
deliberate choice, since CFR is the expensive part and SF is 32ms). Historical-
ply fog analysis still uses SSE (the worker only fetches the live view state).

### Integration fact that decides how often "exhaustive" actually appears

The exact tracker is advanced turn-by-turn ONLY for a color Obscuro itself
plays — `onActionCommitted` fires solely from `vendor/obscuro/src/ObscuroAgent.js` (the
agent's own move), never from the engine for every ply. So for the headline
case (a HUMAN analyzing their own fog game) the human's exact belief is NOT
maintained across their moves; at analysis time `beliefPopulation` sees its
first `beginTurn` at turnNumber > 1, gives up on exactness, and falls to
`ExactBelief.tryReacquire`. Reacquire rebuilds a finite P only when few pieces
are hidden (product under `REACQUIRE_BOUND`). Net effect — which happens to be
exactly the desired behavior:

- **Many pieces hidden (early/mid game): heuristic fallback, `total = null`,
  refines forever.** ("Most of the time 'completely' won't be possible.")
- **Few pieces hidden (late game / simple positions): reacquire yields a finite
  P → enumeration → exhaustion reachable → exact answer, stop.**

Follow-up if fuller mid-game exhaustion is wanted: maintain the human's exact
belief incrementally (call the belief update for every committed ply, not just
Obscuro's own), so enumeration triggers before the endgame. Bigger change —
touches the engine's move-commit path and needs every ply fed to the tracker in
order — deliberately deferred.

## Headline finding: the population is already materialized

The single most important fact, and the one the earlier draft of this doc
missed: **when exact belief tracking holds, the entire population already
exists in memory as a plain array.**

`src/exactBelief.js`'s `ExactBelief.positions` is `Int8Array(66)[]` —
*every* board position consistent with the full observation history (paper's
belief set P), materialized, capped at `CAP = 200000` (avg ~17k in practice),
and it is a **uniform** belief (`samplePositions` draws uniformly *without
replacement*, `exactBelief.js:551`). `ChessGame.sampleWorlds`
(`ChessGame.js:498`) prefers this exact set and only falls back to the
heuristic generative sampler (`belief.js`) when exact tracking was lost or P
outgrew its cap.

Today `analyzeObscuroOnce` asks for `particles: 24` of them and the cp pass
asks for `cpParticles: 20`, both drawn at random. So "process the whole
population" is, for the exact-belief case, **not a research problem at all** —
it's iterating an array we already hold, in chunks, instead of sampling 24
random elements of it. This is a completely different (and far easier) problem
than the one the rest of this doc's "joint equilibrium" section describes.

The catch is *which quantity* you're aggregating. Read the next section before
building anything — it's the crux.

## The crux: evals are additive over worlds, mixing probabilities are not

The panel shows two numbers per move, and they behave completely differently
under "average over more worlds":

- **cp eval per move = an expectation of a per-world leaf value.** It is
  *linear/additive* across worlds. A prior-weighted mean over the enumerated
  population is, once every world is covered, the **exact** belief-expected
  eval — and every partial batch is an unbiased running estimate of it. This
  column cleanly delivers the user's "exhaustive → exact, otherwise refine
  forever" story. Because the exact belief is uniform, "prior-weighted" is
  just a plain mean; each world is counted once.

- **move probability = an equilibrium mixing weight.** It is a *non-linear*
  functional of the whole world set jointly. The true whole-population answer
  is the equilibrium of one game played over all worlds at once — **not** the
  average of many small per-batch equilibria. Averaging per-batch CFR
  distributions (what `analyzeObscuroProgressive` does today, and what the
  batched-enumeration design below would do) converges to *a* well-defined
  quantity, but not to that joint equilibrium, and the KLUSS Resolve/Maxmargin
  safety guarantee does **not** compose across independently-averaged solves.

So there are genuinely two designs, and they are not the same product:

- **Design B — batched enumeration + weighted aggregation (matches the ask's
  described behavior; tractable).** Walk the population once, in batches;
  aggregate; report coverage; stop when exhausted. Evals converge to the exact
  belief expectation. Mixing probabilities converge to the *ensemble average*
  of per-batch equilibria (an honest, defensible number, but not the joint
  equilibrium). Recommended default.

- **Design A — one joint equilibrium over the whole population (the hard
  path).** The only design under which the *mixing probabilities* become the
  true whole-population equilibrium. Requires reworking the KLUSS gadget to
  grow its world/class set mid-solve, which raises a real research question
  about the paper's safety proof. Fully inventoried in the last section; still
  not attempted.

**Decision needed before implementing:** is "exhaustively evaluated" satisfied
by exact *evals* + ensemble-averaged mixing (Design B), or does it require the
true joint-equilibrium mixing (Design A)? The user's operational description
(batches, per-second updates, refine-forever, "evaluated") reads as Design B,
and Design B is what the rest of this section specs. Design A stays documented
below as the harder alternative if the mixing weights specifically must be the
joint equilibrium.

## Design B: batched enumeration — what to build

### 1. Enumerate the population without replacement (exact-belief case)

Replace the per-batch random resample with a cursor over `exact.positions`:

- Add `enumeratePositions(offset, count)` (or `samplePositions`-alongside) to
  `ExactBelief` that returns a *slice* of `this.positions` in a fixed order,
  plus `positions.length` so the caller knows the total. Order can be the
  natural array order; optionally shuffle *once* per analysis session (seeded)
  so early batches aren't spatially biased, but never reshuffle between
  batches or you lose the without-replacement guarantee.
- The progressive loop keeps a cursor `i`; each batch takes worlds
  `[i, i+B)`, runs one solve/eval over them, folds the result into the running
  aggregate, advances `i`. When `i >= length`, the population is **exhausted**:
  emit a final `exhaustive: true` result and stop the loop (this is the
  "converge and stop" case the current re-sampling loop can never reach).
- Aggregation is a *weighted* running mean keyed by move. Under exact belief
  the weights are uniform (count each world once); keep the weight explicit
  anyway so the heuristic-fallback and non-uniform cases below drop in without
  reworking the accumulator.

This is a direct evolution of `analyzeObscuroProgressive`
(`ObscuroAgent.js:489`), which already folds batch results into `probSum`/
`cpSum`/`cpCount` and emits per-batch. The changes are: (a) draw each batch by
cursor from the materialized set instead of `sampleWorlds(...24)` with
replacement, (b) weight the fold, (c) detect and signal exhaustion.

### 2. Coverage / completion signal to the UI

The panel currently shows `Batch N` (`AnalysisPanel.vue:62`). Extend the
per-batch progress payload with `{ evaluated, total, exhaustive }` so the UI
can show "evaluated 4,096 / 17,233 worlds" and a real done-state instead of an
open-ended batch counter. `total` is `exact.positions.length` (or `null` when
on the heuristic fallback, where the population isn't materialized — see below).
`handleAnalyzeStream` already forwards arbitrary `onProgress` fields verbatim
(`api-server.js:988`), so this is additive on both ends.

### 3. Batch size vs. update cadence

The ask is a UI refresh "every second or so." A batch must therefore be sized
to finish in ~1s, not sized to the whole population. Each world costs one
batched Stockfish `multiPV` call at the leaf (`makeChessLeafEval`), so batch
size trades latency for per-tick coverage. Keep the `setImmediate` yield
discipline (see "what NOT to do") so a batch never starves anything. Note the
solve cost is dominated by the CFR rounds over the batch, not just the leaf
evals — for the *eval-only* column you can run a much larger batch per tick
(pure leaf eval, no CFR), which is another reason evals will exhaust long
before the mixing column would.

### 4. The other sampled quantities ("possibly other things as well")

- **cp-eval particles (`cpParticles: 20`, `ObscuroAgent.js:442`)** — this is
  literally another random draw from the *same* `exact.positions`. It folds
  into the same cursor/enumeration trivially and is the cleanest win: the eval
  column becomes the exact population mean once the cursor completes.
- **Heuristic-fallback population (`belief.js`)** — when exact tracking is lost,
  there is no materialized array; `belief.sample` is a *generative* weighted
  sampler (each hidden piece drawn from its `possible` set, `MAX_POSSIBLE = 48`
  each). "The whole population" here is the combinatorial product of those
  per-piece sets filtered for consistency — enumerable *in principle* (a
  systematic Cartesian-product walk with consistency pruning would replace the
  rejection sampler) but almost always astronomically large, so it stays in
  "refine forever, never exhaust" territory. Enumeration-without-replacement
  still helps (stop re-drawing identical particles), but it needs a real
  systematic generator, not the current `sample()`. Lower priority than the
  exact-belief cursor; `total` stays `null` / "≈" in the UI here.
- **In-solve expansion sampling (`gtcfr.js` `sampleWorld`/`expandRoot`)** —
  this is *inside* one CFR solve and is Design-A territory, not Design B. Leave
  it sampled; Design B's "population" is the outer world set, not the tree.

## Client-side Web Worker (the side goal)

Today analysis is entirely server-side: `handleAnalyzeStream` runs the solve in
the Node process and streams SSE; the event-loop starvation fix
(`setImmediate` yields in `vendor/obscuro/src/search.js`) exists precisely because
that solve shares the server's event loop with unrelated move requests.

Moving it into a browser Web Worker would make it genuinely non-blocking for
both the server and the UI, and the batched design above is a *good* fit for a
worker (post a `{evaluated,total,candidates}` message per batch instead of an
SSE frame). Porting inventory:

- **Portable as-is (plain ESM, no Node deps):** `vendor/obscuro/src/*`
  (search/gtcfr/kluss), `src/exactBelief.js`, `belief.js`, `board.js`,
  `fen.js`, the ranking/aggregation glue. These already run in unit tests
  without a server.
- **The real blocker: Stockfish.** `src/stockfish.js` is a *Node*
  `worker_threads` WASM engine with a disk-backed sqlite/NDJSON cache
  (`stockfish.js:19`, cache at `stockfish.js:53+`). A browser worker needs a
  *browser* WASM Stockfish (the vendored `.wasm` is an Emscripten build and can
  run in a `Worker`, but the loader, the `worker_threads` teardown/recycle
  logic, and the disk cache are all Node-shaped and would need a browser
  equivalent — an in-memory or IndexedDB cache, and `Worker`/`postMessage`
  instead of `worker_threads`). This is the bulk of the work.
- **Belief state ownership.** The exact/heuristic trackers currently live
  server-side keyed per game (`WeakMap` on `state.players`). Client-side
  analysis needs the observation-consistent position set on the client. Either
  (a) ship `exact.positions` (or enough to reconstruct it) to the client, or
  (b) rebuild the tracker client-side from the move history the client already
  has. (b) is cleaner but re-runs the belief update in the browser.
- **Fallback stays.** Keep the server SSE path working; a worker that can't
  load WASM should degrade to the server endpoint, mirroring how the engine
  already degrades to the JS search when the vendored files are missing.

A reasonable staging: land Design B server-side first (small, testable diff on
`analyzeObscuroProgressive` + a `total`/`exhaustive` signal), *then* port the
whole batched loop into a worker. The worker port doesn't change the algorithm,
only where it runs.

## Recommended next steps

1. **Design B, exact-belief cursor, server-side.** Add
   `enumeratePositions`/length to `ExactBelief`; convert
   `analyzeObscuroProgressive` from resample-with-replacement to a cursor with
   weighted aggregation and an `exhaustive` stop; fold `cpParticles` into the
   same cursor; add `{evaluated,total,exhaustive}` to progress + panel. This
   alone delivers the user's visible behavior for the common (exact-belief)
   case, including real convergence-and-stop on small populations.
2. **Heuristic-fallback systematic enumeration** (optional, lower value): only
   if positions where exact tracking was lost matter enough to bother; will
   essentially never exhaust.
3. **Web Worker port** of the batched loop (mostly the Stockfish-in-browser
   work above).
4. **Design A (joint equilibrium)** only if the *mixing probabilities*
   specifically must be the true whole-population equilibrium rather than the
   ensemble average — see below. Do not start here.

---

## Design A: one joint equilibrium over the whole population (the hard path)

Everything below is the harder alternative — needed *only* if the move mixing
weights must be the true joint-population equilibrium (not the ensemble average
Design B produces). It was the entire subject of this doc's first draft and is
kept because the technical inventory is still correct. Nothing here has been
attempted.

### Why the belief-world set is fixed within one solve

The search's safety machinery is the KLUSS Resolve/Maxmargin gadget
(`vendor/obscuro/src/kluss.js`, implementing Zhang & Sandholm 2026 §3.1 / App.
B.2, C.1–3). `buildGadget(tree, hooks, cfg)` runs ONCE per solve, before any
CFR iteration:

- Partitions `tree.worlds` into classes `J` by the opponent's observation
  (`kluss.js:54-61`) — carried worlds keep their true opponent infoset,
  freshly-sampled worlds are each their own singleton class (paper Fig. 9
  line 13).
- For each class, computes an **alternate value** `g.altMe` (what the
  opponent could get by "exiting" to a fixed blueprint value instead of
  entering the subgame — `kluss.js:70-83`) and a fresh `RegretMinimizer(2)`
  (`g.R`, enter-vs-exit) — `kluss.js:85`.
- Computes a **non-uniform Resolve prior** `alpha(J)` for every class, which
  depends on `ySum` — the sum of ALL classes' blueprint mass
  (`kluss.js:66-92`). This is the load-bearing detail: `alpha` for every
  EXISTING class is a function of the TOTAL class set, computed once.

Every subsequent CFR iteration (`runGadgetCFR`, `kluss.js:101-150`) reads
`gadget.J` as a fixed array and accumulates regret into each class's `g.R`
and the shared `maxmargin` selector (`new RegretMinimizer(m)`, sized to the
class count `m` at construction — `kluss.js:94`) iteration over iteration.
`gtcfr.js`'s `expandRoot` similarly builds the shared root infoset from
`tree.worlds` once (`gtcfr.js:289-321`), and `sampleWorld` (`gtcfr.js:350-374`)
samples expansion targets from `tree.gadget.J`. So growing the class set
mid-solve changes `alpha`, the per-class regret minimizers, and the Maxmargin
selector's dimensionality out from under CFR iterations that already ran.

### What real mid-solve injection would touch

1. **An `addWorld(tree, hooks, gadget, newState, rng)` entry point.** Sample a
   new belief world (or take the next from `exact.positions` — note Design A
   can also enumerate rather than sample!), wrap as a `Node` (`makeLeaf`,
   `gtcfr.js:21-26`), expand onto the shared root infoset
   (`expandNode(...rootI)` — calls `hooks.evalChildren`, a real Stockfish
   `multiPV` for chess, so adding a world costs one expansion and needs the
   SAME `setImmediate`-yield discipline the round loop has).
2. **Incremental class-set bookkeeping in the gadget.** A freshly-injected
   world is a new singleton class (Fig. 9 line 13): push a new `g` onto
   `gadget.J`, give it a fresh `RegretMinimizer(2)`. Straightforward.
3. **The open engineering question: renormalizing `alpha` and `maxmargin`
   mid-solve.** `alpha(J) = 0.5·(y_J/ySum + 1/m)` for every class depends on
   `ySum` and `m` — both change when a class is added. Recomputing `alpha` for
   all classes (cheap, `O(m)`) leaves each class's accumulated `g.R` chasing a
   moved target mid-flight; whether CFR's regret bounds still hold under a
   shifting `alpha` is not established by this codebase's comments or tests.
   `maxmargin = new RegretMinimizer(m)` is sized at construction; growing `m`
   needs a resizable minimizer or discarding its accumulated regret.
4. **The open theoretical question: does the paper's Resolve/Maxmargin safety
   proof (bounded exploitability vs. a blueprint) cover a growing class set?**
   Needs checking against Zhang & Sandholm 2026 App. B.2/C.1–3 directly, not
   inferred from code comments. If the proof assumes a fixed subgame root,
   mid-solve injection may need a different gadget construction entirely.

### Safer intermediate for Design A

**Periodic gadget REBUILD with regret-state carryover.** Every N rounds, call
`buildGadget` again over an enlarged `tree.worlds`, but copy over accumulated
state (`q`, `pred`, `avg`, `iters` — mirror `gtcfr.js:74-78`'s
`warmStartInfoset`) for classes that already existed, cold-starting only
genuinely new classes. This is "restart with a warm blueprint" (proven safe
elsewhere — see `FOG-AI-FIX-PLAN.md`'s carryover work), sidesteps the open
theoretical question, and would likely beat the ensemble average in sample
efficiency. Natural first experiment for Design A, not the final answer.

## What NOT to do

- Don't convert the exact-belief cursor to resample-with-replacement "for
  simplicity" — the whole point of Design B over today's
  `analyzeObscuroProgressive` is the *without-replacement* enumeration that
  makes "exhausted → exact eval, then stop" reachable. With-replacement never
  exhausts and re-burns Stockfish calls on worlds already scored.
- Don't present Design B's ensemble-averaged mixing probabilities as the joint
  whole-population equilibrium — they aren't (see the crux section). Label the
  eval column as convergent-to-exact and understand the probability column is
  an ensemble average.
- Don't mutate `tree.worlds.push(...)` and re-run `runGadgetCFR` without
  rebuilding the gadget — `gadget.J`, `alpha`, and `maxmargin`'s
  dimensionality would silently desync from `tree.worlds`; nothing asserts that
  invariant.
- Don't skip the `setImmediate`-yield discipline in any new batch/expansion
  loop — it's easy to reintroduce full-process starvation (see the event-loop
  fix in the search module).
- Don't benchmark only against final answer quality — also watch the gadget's
  own `pmax` safety signal (`kluss.js:223` region in `search.js`'s `safe`
  computation); an `alpha`/class-count desync shows up there first.
