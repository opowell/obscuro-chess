# Obscuro chess: a roadmap for playing strength — plan doc

Standalone working doc. Written 2026-08-17. Safe to read cold in a new session.
Companion to [FOG-AI-FIX-PLAN.md](FOG-AI-FIX-PLAN.md) (search-side king safety —
already fixed), [MOVE-PRIOR-PLAN.md](MOVE-PRIOR-PLAN.md) (the belief and its
prior — most of this doc's Phase 1–2 build directly on its most recent entries),
[UNLIMITED-BELIEF-PLAN.md](UNLIMITED-BELIEF-PLAN.md) (the belief-population
batched walk) and [ANALYSIS-DEPTH-PLAN.md](ANALYSIS-DEPTH-PLAN.md) (iterative
deepening in the analysis panel — not real play, but the TIME-mode machinery
Phase 3 reuses).

## The ask

`NEXT STEPS.md` states the standing open question in one line: *"What is the
key to improving strength? Converting belief accuracy into strength is the open
problem, and a bigger corpus is not on its own the answer."* This doc is the
answer to "what do we do about that" — a prioritized, independently-measurable
roadmap, written after auditing every existing planning doc and the settled/
void measurements in `docs/PARAMETERS.md`.

## Status

Two things landed recently that change what's actually open:

- **`sampleAlpha` now ships at 1** (`258a3c8`). Belief worlds are drawn
  proportional to the posterior instead of uniformly over `P`. This is real
  progress on the open question above, not just another null result — see
  Phase 1–2 below for what it actually settles and what it doesn't.
- **Leaf-net distillation was tried twice and is closed** (`3743d0e`,
  `c7ecbbb`). A 768→32→1 net (v1) and a king-bucketed-feature net (v2) both
  fail to close a ~1000× data gap between this project's ~2M labelled
  positions and what a from-scratch NNUE-scale evaluator needs. See Phase 4 —
  do not reopen without a much larger data source.

One loose end from that work: fixing the leaf-net comment also corrected a
stale claim on `_leafEval` that used to justify capping leaf-search depth at 4
("depth 7 is dominated"). The comment is fixed; the actual dial constant is
not — that's Phase 1, item 1 below.

## Phase 1 — ship what's already measured but not deployed

Near-zero research risk: both items below are "the measurement exists,
someone still needs to move the constant and re-verify at production settings."

### 1. Raise the leaf-search depth ceiling — `src/ObscuroAgent.js`

`CHESS_DIAL.leafEval.sfDepth` still reads `{ min: 2, max: 4, curve: 'linear' }`.
The leaf-depth grid (6,520 paired positions, 120 crawl games, worlds=16,
reference depth 12 — see memory `leaf-depth-grid`, and the corrected comment in
commit `3743d0e`) found:

| config | mean cp | median | ms/move | best-move% |
|---|---|---|---|---|
| depth 1, 24 rounds | 69.0 | 33.0 | 550 | 24.5 |
| depth 2, 12 rounds | 68.0 | 31.0 | 403 | 25.4 |
| depth 4, 6 rounds | 67.2 | 30.0 | 405 | 25.9 |
| depth 7, 3 rounds | 63.9 | 24.0 | 666 | 29.2 |

Depth 7 beats depths 1/2/4 decisively (paired vs depth 4: −3.30 ± 0.98 cp,
z=−3.36); depths 1, 2 and 4 are statistically indistinguishable from each
other. The dial's current ceiling of 4 is therefore not measurably better than
depth 1, and the number that used to justify stopping there is void.

**Before just changing the constant**: the grid above traded depth against
*rounds* as its own experimental design (depth 7 only got 3 rounds). Production
doesn't spend its budget that way — round count comes from
`search.DIAL.power.maxRounds` (a separate knob, range 6–100), independently of
`sfDepth`. Re-run `move-quality.mjs --grid` at the actual production dial
settings (not the grid's isolated frontier) before picking a new ceiling, then
raise `sfDepth`'s `max` (7 is the best-supported starting point) and update
this doc's citation plus `docs/PARAMETERS.md` §2.2 per the parameter-change
checklist in that doc's §3.

### 2. Reduce belief-tracking censoring around `sampleAlpha=1` — `src/exactBelief.js`

`CAP` (200,000) and `TIME_GUARD_MS` (4,000) are unchanged from before the
α flip. The commit that shipped α=1 measured its own defaults working against
it: at `CAP=200,000` / `guard=4,000ms`, exact tracking is abandoned — stickily,
for the rest of the game — on 29% of the high-`|P|` turns where α=1's benefit
concentrates, dragging the measured gain from −2.78 ± 0.59 cp to −1.64 ± 0.63
cp. The α=1 result was itself measured at `CAP=2e6` / `guard=180s` specifically
to avoid this censoring.

Exact-belief updates are cheap on their own (~1µs/candidate after the Tier-1
compact-representation rewrite — see `FOG-AI-FIX-PLAN.md`'s "Third follow-up
round"), so raising these two constants toward production-safe values (not all
the way to the 2e6/180s measurement config, which was chosen for statistical
power, not for serving) may reclaim a real fraction of that 1.1 cp for a real
but likely modest latency cost. Measure per-move wall-clock and give-up rate at
a few intermediate settings (e.g. `CAP` in the low millions, `guard` in the
8–15s range) under actual production round/world budgets before picking new
defaults — this repo's whole history is measure-before-ship, and this constant
in particular (`TIME_GUARD_MS`) trades directly against `beginTurn` latency,
which past incidents (`_giveUp()` firing under load) have shown is not free to
spend carelessly.

## Phase 2 — close the "does it actually win games" gap

This is the highest-priority open question in the whole roadmap, because every
other phase either assumes α=1 is a real strength gain or is orthogonal to it.

α=1 shipped on a **cp-loss-vs-depth-12-reference** proxy — explicitly
acknowledged in the shipping commit as "blind to information value" (it can't
credit a move for managing what the opponent does or doesn't get to see, only
for matching a perfect-information engine's opinion of the position). The only
actual win/loss self-play measurement on record is 15 games and points the
*other* direction (4–11 for α=0) — far too small to trust, and itself run
before the harness's determinism fixes (`FRESH_HASH`, no mid-run engine
recycling — see memory `paired-measurement-determinism`).

**Next step**: run `scripts/strength-belief.mjs --arm alpha` seat-swapped,
at the scale the script's own header says is needed — "hundreds of games," not
15. Throughput reference: self-play measured ~0.35s/ply at difficulty 25, up to
~2.06s/ply at difficulty 100 (`MOVE-PRIOR-PLAN.md`, "The corpus is the binding
constraint" section). A few hundred games × ~30–60 plies × 2 (seat swap) is
multi-hour-to-overnight wall-clock; games are independent, so parallelize
across processes/cores rather than running serially.

Two outcomes to plan for:
- **Confirmed** (α=1 wins more games, not just lower cp-loss): the shipped
  default is validated on the actual target metric, not just the proxy. Done.
- **Not confirmed, or reversed**: this becomes the next real research question
  — *why* does a metric that says "more accurate against a perfect-information
  referee" disagree with actual fog-chess outcomes? One live hypothesis worth
  checking first: does weighting the search's world draw toward the
  most-likely opponent lines under-hedge against surprising-but-plausible ones
  that the depth-12 proxy doesn't distinguish from "the AI made a good
  decision under uncertainty"? Don't guess past this — instrument it and
  measure, the way every other question in this doc's ancestry was resolved.

## Phase 3 — attack the actual resource ceiling

Everything above trades against wall-clock: deeper leaves, a looser belief
cap, more belief worlds, more CFR rounds all cost more time per move. This
phase buys back that budget, and its payoff compounds with every item above.

### 1. Parallelize the Stockfish leaf evaluator across a worker pool

Already scoped, not yet built. Profiling a real move (`MOVE-PRIOR-PLAN.md`,
"Search scale" section) found ~80% of move time inside Stockfish (batched
across belief worlds and node children — embarrassingly parallel), ~20%
single-threaded JS CFR. Amdahl's law caps a worker pool at roughly 5× on that
split; the measurement machine has 14 cores and `src/stockfish.js` uses
exactly one today. This is pure engineering — no open research question — and
it's the single lever that makes every other phase's cost cheaper to spend:
raising leaf depth (Phase 1.1), raising the belief cap (Phase 1.2), and
widening belief-world counts all become more affordable per unit of wall-clock
once leaf evaluation itself is parallel.

Parallelizing the CFR pass itself is a separate, harder problem (sequential
regret updates) — lower priority; only revisit if the worker pool alone
doesn't unlock enough headroom.

### 2. Ship a "strongest" preset — `src/presets.js`

The generic search's own dial (`vendor/obscuro/docs/PARAMETERS.md` §1.2) shows
POWER mode is hard-capped well below what the engine can otherwise do: worlds
1–48, `timeBudgetMs` 30–2000, `maxRounds` 6–100, `maxInfosets` 400–6000. TIME
mode's ceilings are far looser — worlds 4–48, `maxRounds` effectively
unbounded (100000, constant), `maxInfosets` 1000–25000 — and chess's own
iterative-deepening ladder (`ANALYSIS-DEPTH-PLAN.md`) climbs leaf search to
`MAX_SF_DEPTH=30`, bounded only by the actual clock given via `aiTimeMs`.

For "as strong as possible" whenever an opponent's clock allows it, TIME mode
with a generous `aiTimeMs` is very likely already the strongest configuration
this engine can produce today — no new algorithm work, just the right
settings. Mirroring the existing `--preset paper` pattern (`src/presets.js`),
add a named preset that pins TIME mode with a large default budget, so
"play as strong as you can" is a `--preset` flag rather than a hand-assembled
settings file. Measure it against the current difficulty-100 POWER-mode
default with `move-quality.mjs` before shipping it as a recommendation, same
as every other preset in that file.

## Phase 4 — data (longer horizon, lower near-term priority)

### 1. Move-prior opponent-conditioning stays null

Rating slopes and actor-type conditioning have failed to beat the pooled model
three separate times now (`MOVE-PRIOR-PLAN.md`), each time attributed to data
volume rather than a modeling failure — the populations genuinely differ, but
246 games / 14,836 decisions isn't enough to estimate nine per-band parameters
without the added variance costing more than the reduced bias buys back.
**Self-play cannot supply this data**: fitting π on self-play makes it a model
of Obscuro, measurably worse against humans (2026-08-05 finding). Only a
larger *external* human corpus helps (further Chess.com Fog of War crawls,
extending the one already in this checkout at
`fow-crawl-2026-08-06T17-38-19-279Z.json`), and per `NEXT STEPS.md`'s own
framing, a bigger corpus alone is explicitly not the bottleneck on strength.
Deprioritized relative to Phases 1–3.

### 2. Leaf-net distillation is closed — do not reopen without ~1000× more data

Two independent architecture attempts (plain 768→32→1, then king-bucketed
features mirroring NNUE's HalfKP trick) both plateau at the same ceiling: top-1
move agreement ~21%, Spearman ~0.33, against Stockfish depth-1's own 38.2%/
0.641 on identical holdout nodes. The gap is data volume (~2M samples vs. the
billions Stockfish's own NNUE trained on), not capacity or features — both
attempts to fix it that way moved nothing. Self-play throughput can't close
the gap either: a measurement run yields ~1.5M labelled children, so reaching
NNUE-scale data is ~700 runs at hours each; training on game outcomes instead
of search scores is worse off (~1 independent label per game, not ~55). See
memory `distilled-leaf-net` and commits `3743d0e`/`c7ecbbb` for the full
record before considering this again.

## Verification discipline for every phase above

This repo has a long, specific history of confident-wrong numbers from
otherwise-reasonable-looking measurements (see `docs/PARAMETERS.md` §2.4.1,
and memory `paired-measurement-determinism`). Whoever executes a phase above
should follow the same discipline that got every number in this doc to a
trustworthy state:

- **Paired, seat-swapped measurement** — never a raw win/loss tally on
  unswapped seats (white wins ~10–11 of every 12 games regardless of arm; see
  `strength-belief.mjs`'s own header).
- **Run the null control first.** Two identically-configured arms must agree
  100.0% / 0.00 ± 0.00 cp. Anything less means the instrument, not the
  question under test, is what moved.
- **`FRESH_HASH` on, and no mid-run engine worker recycling** — `go depth N`
  is not a pure function of the position otherwise (Stockfish carries its
  transposition table across searches), and a worker respawned on a
  cache-hit-insensitive counter desyncs a paired comparison's two arms.
- **Watch `move-quality.mjs`'s exact-`|P|` health line.** A run under ~80%
  exact tracking is comparing two arms that both collapsed to the heuristic
  fallback, which α and the belief-population items above are not knobs on.
- **Each phase lands in its own git worktree**, branched from `origin/main`,
  per the outer `battle-simulator` repo's `CLAUDE.md` (this is a shared
  alpha-stage checkout with other agents potentially active concurrently).

## Key files

- `src/ObscuroAgent.js` — `CHESS_DIAL.leafEval.sfDepth` (Phase 1.1), the
  Stockfish call sites a worker pool would parallelize (Phase 3.1).
- `src/exactBelief.js` — `CAP`, `TIME_GUARD_MS`, `SAMPLE_ALPHA_DEFAULT`
  (Phase 1.2, already 1 on `origin/main`).
- `src/presets.js` — where a "strongest" preset belongs (Phase 3.2), alongside
  the existing `paper` preset.
- `scripts/move-quality.mjs` — paired cp-loss harness; `--grid` for the depth
  re-measurement in Phase 1.1.
- `scripts/strength-belief.mjs` — seat-swapped self-play harness; the
  properly-powered run in Phase 2.
- `src/stockfish.js` — the single-worker engine backend Phase 3.1 pools.
- `docs/PARAMETERS.md` — every constant named above, with its own citation;
  update it alongside any constant this doc's phases change.
