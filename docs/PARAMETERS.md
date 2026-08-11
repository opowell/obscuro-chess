# Obscuro chess AI parameters

Every tunable number in the **fog-chess** Obscuro AI: the Stockfish leaf
evaluator, the belief trackers, the move prior, and chess's own difficulty dial
(`src/ObscuroAgent.js`, `belief.js`, `exactBelief.js`, `movePrior.js`,
`stockfish.js`).

**Where the numbers actually live.** Two modules aggregate every default by
re-exporting the named constant each file already declares, so this document and
the code can't drift apart silently:

- [`src/settings.js`](../src/settings.js) — the fog-chess
  defaults documented below
- [`vendor/obscuro/src/settings.js`](../vendor/obscuro/src/settings.js) —
  the generic search defaults (see §1)

Both are read-only aggregates, not the source of truth for the *values* — each
constant is still defined next to the code it tunes, with the comment explaining
*why* that number and not another (several have "scar tissue" history: a value
that looked reasonable and measurably made play worse). If you're changing a
default, edit it at its declaration (linked below), not in the aggregate.

**To override one without editing anything**, see [SETTINGS.md](SETTINGS.md).
Every name in the two aggregates is also a settings key — `chess.LEAF_CLAMP`,
`search.DIAL.power.worlds` — settable from a JSON file, the command line, or
per game. A parameter can be **fixed** (the difficulty dial stops moving it) or
left to **scale** with the dial, whose own endpoints and curve are settable too.
A whole configuration can also come under one name: `--preset paper` is the
Zhang & Sandholm setup ([SETTINGS.md § Presets](SETTINGS.md#presets)).

---

## 1. The generic search (any game)

The search itself is no longer part of this repo. It lives in
[github.com/opowell/obscuro-ai](https://github.com/opowell/obscuro-ai),
vendored here as a submodule at `vendor/obscuro`, and its parameters — the
0–100 difficulty dial, the tree/solve budgets, the KLUSS and purification
constants — are documented in
[`vendor/obscuro/docs/PARAMETERS.md`](../vendor/obscuro/docs/PARAMETERS.md).

Everything below is what chess layers *on top* of that.

---

## 2. Fog-chess specifics (`src/`)

Chess adds exactly two things on top of the generic search (see the header of
`src/ObscuroAgent.js`): a **Stockfish leaf evaluator**, and the
**perfect-information shortcut** (nothing hidden ⇒ play Stockfish directly,
skip the fog subgame). Everything else — belief sampling, difficulty scaling,
move selection — is inherited from §1.

### 2.1 Terminal values and clamps (`src/ObscuroAgent.js`)

| Parameter | Default | Meaning |
|---|---|---|
| `LEAF_CLAMP` | 1500 (cp) | material/eval scores are clamped to this so an imagined king capture from phantom hidden pieces can't swamp a real material decision |
| `SEARCH_WIN` | 8000 (cp) | the search's terminal win/loss magnitude — chess's override of the generic `1e6`, deliberately bounded (~5.3× `LEAF_CLAMP`) because fog terminal values are *averaged* across belief worlds; unbounded win lets one phantom world dominate |
| `MAX_SF_DEPTH` | 30 | ceiling of the iterative-deepening ladder (a ceiling to climb toward, not a depth usually reached — see `makeIterativeChessLeafEval`) |
| `REFUSED_CHILD_CAP` | 8 | how many children of a node the engine refuses to score (the side not to move is "in check") get a real evaluation of their own, best static score first. Capped because refused nodes are ~10% of all nodes and pricing every child of each would multiply engine work ~4×. The `OBSCURO_REFUSED_CHILD_CAP` env var, which predates the settings system, is now just this parameter's *declared default* — the ordinary layers outrank it. |

`KING_HANG` (= `SEARCH_WIN`) is the value assigned when a move leaves the
mover's own king capturable — deliberately asymmetric with the `+LEAF_CLAMP`
cap on *capturing the enemy* king at a leaf (see the file's own comment: an
imagined capture is phantom-prone, but exposing our own king is a real,
self-inflicted loss).

### 2.2 The chess difficulty dial (`CHESS_DIAL`, `src/ObscuroAgent.js`)

Layered on top of §1.1 — same `t`, chess's own leaf-eval and sampling knobs.

**`_leafEval` (POWER mode)** — one fixed Stockfish depth/breadth for every
node in the tree:

| Field | Range | Formula |
|---|---|---|
| `leafEval.sfDepth` | 2 – 4 | `round(2 + t·2)` — `move-quality.mjs --grid` found depth 7 *dominated* (it buys tactical leaves at the cost of tree size, and the tree was worth more at this search's scale, ~100× smaller than the paper's). **That run is void** — it went through the same corrupted `replayArm` as every other pre-2026-08-07 measurement (§2.4.1), so it compared depths on a belief that had collapsed to the heuristic fallback by ply 2. Re-run the grid before trusting this range. |
| `leafEval.cols` | 5 – 14 | `round(5 + t·9)` — MultiPV lines requested per node |

**`_leafEval` (TIME mode)** — full breadth always, iterative deepening bounded
by a per-call slice of the move budget:

| Field | Default | Meaning |
|---|---|---|
| `timeModeEvalsPerWorldReserve` | 8 | root expansion (one eval per belief world, before the round loop starts) is budgeted `timeMs / (worlds × 8)` per call — an eighth of the budget, leaving the rest for tree growth |
| `timeModeMinPerCallMs` | 30 | floor on the per-call slice above |

**`_proportionalPick`** (perfect information, POWER mode: score every legal
move at full Stockfish strength, then sample proportional to win probability
sharpened by `β`):

| Field | Range/Value | Formula |
|---|---|---|
| `proportionalPick.multipvCap` | 20 | `min(legalActions.length, 20)` |
| `proportionalPick.depth` | 10 – 16 | `round(10 + t·6)` |
| `proportionalPick.betaAtHalf` | 1 | `β` at `t = 0.5` — probability exactly ∝ win-probability |
| `proportionalPick.betaMax` | 12 | `β` at `t = 1` — collapses toward near-best play (t ≥ 0.999 is exactly pure Stockfish) |

**Perfect-information TIME mode** plays `stockfishBestAction` at full strength —
the user's clock is the only handicap, so there is no Skill Level one on top:

| Field | Default | Meaning |
|---|---|---|
| `timeModeSkill` | 20 | UCI Skill Level (20 = no handicap). `movetime` is the user's limit, clamped to `[1, 600000]` |

### 2.3 Analysis-panel search sizes (`ANALYSIS_DEFAULTS`, `src/ObscuroAgent.js`)

`obscuroStrategy` (the inspection helper used by tests and the analysis panel)
and `analyzeObscuroProgressive` (the panel's belief-population walk) have
their own search-size defaults, independent of a real move's difficulty
setting:

| Parameter | Default | Meaning |
|---|---|---|
| `strategy.particles` | 8 | belief worlds sampled when the caller doesn't supply `opts.worlds` |
| `strategy.maxRounds` / `.expandPerRound` / `.cfrPerRound` / `.purifyMax` | 30 / 8 / 4 / 3 | `obscuroStrategy`'s own search-size fallback |
| `batchMixing.maxRounds` / `.expandPerRound` / `.cfrPerRound` | 100 / 16 / 8 | the mixing solve run once per batch inside the progressive walk — bigger than `strategy`'s defaults since it only runs once per batch, not once per candidate |
| `maxTotalMs` | 5 min | safety net for a missed disconnect, **not** a quality cap |
| `batchSize` | 16 | belief worlds enumerated/sampled per batch |
| `sweepBatches` | 4 | generative-fallback only: batches per ladder rung before climbing depth |
| `likelyWorldsCap` | 32 | size of the "most likely boards" overlay |
| `scoredWorldsCap` | 96 | cap on the per-world cp view |
| `captureMultipv` / `captureDepth` | 8 / 12 | `_captureStockfishAnalysis`'s ranking beside a time-mode perfect-information move — display only, never changes what is played |

### 2.4 Belief tracking

Two trackers, always kept in lockstep (`ChessGame.sampleWorlds`): the **exact**
position-set tracker `P` (`exactBelief.js`) is preferred and, while it holds,
*is* the paper's belief; the **heuristic particle** tracker (`belief.js`) is
the fallback once exactness is lost (P outgrew its cap, a time guard tripped,
or the tracker was attached mid-game).

**Exact belief (`src/exactBelief.js`)**

| Parameter | Default | Meaning |
|---|---|---|
| `CAP` | 200,000 | max `\|P\|` before exactness is abandoned (paper: usually ≤ 10⁶ in C++; this tracker averages ~17k) |
| `TIME_GUARD_MS` | 4000 | per-turn update budget; exceeding it also abandons exactness |
| `REACQUIRE_BOUND` | 60,000 | max cross-product size `tryReacquire` will search when trying to rebuild a lost `P` from the heuristic belief |
| `SAMPLE_ALPHA_DEFAULT` | 0 | exponent applied to the posterior when *sampling* search worlds (`draw ∝ w^α`). **Ships at 0 (uniform over P), deliberately ignoring the posterior**, and as of 2026-08-11 that is measured rather than assumed: **+0.21 ± 1.08 cp (z = 0.20), sign test 48.8%** over 4,124 paired holdout positions, on the first harness whose null control holds (see `FRESH_HASH`). Flat. α=1 measured *better* sample coverage (39.3% vs 36.1%) and *worse* actual play (4–11 in seat-swapped self-play), and both harnesses now agree. Every earlier move-quality number for α is void in both directions — see §2.4.1. |
| `REACH_WEIGHTING_DEFAULT` | 0 | exponent β on how much each sampled world is *worth* to the CFR (reach ∝ `w^β`), as opposed to how it is sampled. **Ships off, and the question is closed as of 2026-08-11**: β=1 is +0.46 ± 1.03 cp (z = 0.45) and β=0.5 is −1.35 ± 1.03 (z = −1.30) over 6,150 paired positions, with β=0.5's lean vanishing (−0.04 ± 1.06) on the exact-`\|P\|` subset. A `\|P\|`-dependent effect was pre-registered on a holdout and refuted (contrast z = 0.50). Every earlier number for this knob is void — three separate harness/engine defects, detailed in the long comment on `setBeliefReachWeighting`. |

#### 2.4.1 Every `move-quality.mjs` number before 2026-08-07 is void

The harness asked the agent for a move, measured it, and then committed the
**recorded** move — but `ObscuroAgent.chooseAction` had already committed the
move *it* chose, so the belief was advanced by two of our own moves per turn, one
of them never played. `vendor/obscuro/src/ObscuroAgent.js` names the consequence
directly: "committing an action other than the one actually played silently
corrupts the belief (fatally so for the exact tracker)."

Measured effect: **P died on ply 2 of every replay**, so both arms ran on the
heuristic particle fallback — which α and β are not knobs on — and the script was
comparing two configurations that had no way to differ. Fixing it moved mean cp
loss from 98.8 to 59.2 on the same positions.

This invalidates every arm the script produced, including the `--grid` leaf-depth
frontier that set `leafEval.sfDepth` in §2.2 (that row's "depth 7 is dominated"
has not been rechecked). `move-quality.mjs` now prints the share of positions
holding an exact |P| on every run, and shouts when it drops below 80% — that
count is what caught this, and it is the guard against it recurring.

Both α and β are the *initial* values: `setBeliefSampleAlpha` /
`setBeliefReachWeighting` (and their per-seat variants, which is how the A/B
harnesses run two models in one process) still win over a settings file.

Both ship at 0, which is also **what the paper does** — it samples worlds
uniformly from `P` and assumes every world in an information set is equally
likely, having no model of the opponent. `--preset paper` pins them there
alongside the rest of that setup ([SETTINGS.md](SETTINGS.md#presets)).

**Heuristic particle belief (`src/belief.js`)** — used once exact
tracking is lost:

| Parameter | Default | Meaning |
|---|---|---|
| `MAX_POSSIBLE` | 48 | cap on a single piece's possible-square set |
| `THREAT_BIAS` | 3 | how strongly to over-sample placements that attack our pieces. **Kept deliberately modest** — a past incident: over-weighting phantom attackers made the AI huddle instead of saving real material. |
| `MAX_LURKERS` | 2 | max invisible pieces per particle allowed to threaten our pieces at once — without this cap, threat-biased sampling hallucinates coordinated mating attacks |
| `RECAPTURE_TYPE_WEIGHT` | `{pawn:9, knight:3, bishop:3, rook:1.5, queen:1, king:0.5}` | relative likelihood a piece of each type made a forced (known) capture — recaptures strongly favour the least valuable capturer |
| `MAX_ATTEMPTS_PER_PARTICLE` | 6 | resample attempts per requested particle before giving up |
| `PHANTOM_CHECK_REJECT_WINDOW` | 4 (× n) | how long `sample()` keeps rejecting particles with a phantom self-check before accepting anything |

### 2.5 Move prior — π(move | position) (`src/movePrior.js`)

The opponent model that turns the exact belief `P` from a set into a
distribution: a softmax over a cheap, O(1)-per-move score (capture value,
promotion value, piece-square-table delta, a castling bonus).

**`FITTED_WEIGHTS`** — the production model, **fitted by conditional-logit MLE
(`fit-move-prior.mjs --write`), not hand-tuned**. Refitted 2026-08-06 on 246
Chess.com Fog of War games / 14,836 decisions by 192 players, replacing a fit on
37 games that were one human plus this engine:

| Field | Value | Note |
|---|---|---|
| `temperature` | 100 | fixes the unit only (logits per centipawn × 100) — sharpness lives per-term in `pstWeight`, not in one global knob |
| `floor` | 0.03 | mixes in the uniform prior: `π = (1-floor)·softmax + floor/\|M\|`; bounds how much damage one confidently-wrong parent can do |
| `captureWeight` | 0.943 | |
| `promoWeight` | 0.753 | |
| `pstWeight` | `[–, 2.887(P), 2.804(N), 6.523(B), 4.509(R), 1.662(Q), 0.032(K)]` | per-piece PST-delta weight. **The king weight is ~0 and must stay there** — see below. |
| `castleBonus` | 245.2 | the single biggest term in the fitted model |

The refit shipped because it won on held-out games of a corpus the old weights
had never seen: move log-loss 2.922 → 2.896 (5 folds of 5), and on the gate that
decides — belief log-loss of the true position — 5.229 → 5.161, with the true
board's median rank improving 33 → 25.

**The king weight used to be −0.853, documented as "negative on purpose".** That
finding did not replicate: across 8 disjoint folds of the larger corpus the term
comes out −0.2, +0.6, 0.0, +0.2, +0.3, −0.2, +0.2, −0.4 — sign-flipping in 5 of
8, mean 0.03, τ_eff ≈ 3000. Every other term keeps its sign and rough magnitude
across the same folds. The old value was one player's habit, and the model now
says nothing about where kings go. `move-prior.test.js` pins it near zero.

Do not hand-tune these — read the file's header (esp. the "SCAR TISSUE"
section) before changing any of them. `belief.js`'s `THREAT_BIAS`/`MAX_LURKERS`
document two earlier times an over-sharp belief made the AI measurably worse;
this model's `floor` exists for the same reason.

**`MOVE_PRIOR_UNIFORM`** — `false`. Serve the model-free baseline instead of the
fitted model: every fog-legal move equally likely (`UNIFORM_PRIOR`). This is the
paper's own setting — it samples uniformly from `P` and models nothing about how
the opponent chooses — and it is the arm every measurement of this model is
against, which is why it is a switch rather than something a caller has to
reconstruct out of weights (`floor` cannot reach 1, and `temperature: Infinity`
is not expressible in JSON). Checked on **both** compile paths, so turning it on
cannot leave a rating-tilted prior serving one seat. Note that at the shipped
α = β = 0 the posterior reaches play through nothing at all, so on its own this
changes the belief the analysis panel and `calibrate-belief.mjs` report, not the
move.

**`RATING_SLOPE`** — `[0,0,0,0,0,0,0,0,0]`, and zeros are the shipped state.
The opponent's rating tilts every weight continuously rather than selecting a
bucket:

```
weight_k(r) = FITTED_WEIGHTS_k + RATING_SLOPE_k · z(r),   z = (r − RATING_PIVOT) / RATING_SCALE
```

with `RATING_PIVOT` 2000, `RATING_SCALE` 400 and `RATING_Z_CLAMP` 1.5, so an
outlier rating cannot extrapolate the line past where the corpus went (±1.5 is
roughly 1400–2600 Elo at the shipped pivot and scale). Fit it with
`fit-move-prior.mjs --rating --write`, which only writes slopes that beat the
flat model on held-out games by `--min-rating-gain` (0.01 nats).

All four are settable — `MOVE_PRIOR_RATING_SLOPE`, `MOVE_PRIOR_RATING_PIVOT`,
`MOVE_PRIOR_RATING_SCALE`, `MOVE_PRIOR_RATING_Z_CLAMP` — because the pivot, scale
and clamp describe **the corpus the slopes were fitted on**: a host serving its
own slopes under this package's pivot evaluates the line in the wrong units.

Continuous, not bucketed, because bucketing estimates each band's nine weights
from a fraction of the data — three bands means each weight sees a third of the
decisions — while the interaction form uses every rated decision for every slope,
has no edges to choose, and cannot serve two adjacent ratings different models.

Measured 2026-08-06 on 14,614 rated decisions: the sloped model scores **2.896**
against the flat model's **2.897** held out — Δ +0.0003 nats. The individual
slopes are not small (rook PST moves −2.7 per 400 Elo, 64% of its base), which is
what fitting noise looks like when you have 9 extra parameters. Serving with
zeros reduces exactly to `FITTED_WEIGHTS` at every rating.

At serve time the **opponent's** rating is what tilts the model
(`state.gameSpecific.opponentRating`, or `rating` on the opposing player);
compiled priors are cached per 25 Elo, which is a cache granularity, not a bin.

### 2.6 Stockfish backend (`src/stockfish.js`)

| Parameter | Default | Meaning |
|---|---|---|
| `RECYCLE_AFTER` | 400 (calls) | the engine worker is torn down and respawned after this many searches, proactively, before the vendored WASM's heap growth aborts it |
| `CACHE_MAX` | 20,000 (entries) | LRU cap on the disk-backed MultiPV cache (SQLite when `node:sqlite` is available, else append-only NDJSON) |
| `STOP_POLL_MS` | 100 | how often an in-flight search re-checks the caller's `isCancelled` and can send UCI `stop` |
| `SF_CACHE_DIR` | `vendor/stockfish/` | where that cache lives. Derived data, not part of the package — an embedder with a warm cache keeps it in its own checkout. Overridden by the `SF_CACHE_DIR` env var, and by a `setCacheDir()` call before the first search. |
| `LEGACY_DIFFICULTY` | `{easy:10, medium:35, hard:65, expert:90}` | maps old string difficulty tiers onto the 0–100 scale, for saved sessions |
| `DEFAULT_DIFFICULTY` | 25 | what "no difficulty given" means. One number, read by `difficultyToNumber`, `FogChess.createInitialState` and the agent's `_config` — it used to be answered 25/25/50 by those three, so an observation that had lost its difficulty played at a different strength depending on which path asked. |
| `SF_DIFFICULTY_RAMP` | `movetimeMs: 50–1000, skill: 0–20` | perfect-information difficulty → `(movetime, Skill Level)`, used by `sfOptsForDifficulty` (the legacy/non-Obscuro chess agent path) |

### 2.7 The alpha-beta agent's dial (`CHESS_AGENT_DIAL`, `src/ChessAgent.js`)

`ChessAgent` is the plain alpha-beta + Stockfish agent, not the Obscuro one. It
has always had its own difficulty ramp; until this table existed it was an
inline literal inside `chessConfigForDifficulty`, which made it the one dial in
this package that no aggregate listed and nothing could override. Same numbers:

| Field | Range/Value | Meaning |
|---|---|---|
| `depth` | 2 – 5 | plies searched per particle |
| `noiseCp` / `noiseZeroAt` | 250 / 0.5 | random score jitter in centipawns at `t = 0`, falling linearly to 0 by `t = 0.5` — this is what makes weak difficulties blunder |
| `quiesceFrom` | 0.2 | resolve capture sequences at leaves from this `t` up |
| `fog.particles` | 4 – 18 | belief particles under fog |
| `fog.topK` | 6 – 10 | candidate-move prefilter width |
| `fog.depthShallow` / `.depthDeep` / `.shallowBelow` | 1 / 2 / 0.2 | per-particle depth under fog: `depthShallow` below `t = 0.2`, `depthDeep` at or above |

`CHESS_AGENT_SCORING` is the same story one step further: how this agent turns a
cloud of particle scores into one number, and how it prices fog. Three `const`
scalars until this table existed, so nothing could reach them either:

| Field | Default | Meaning |
|---|---|---|
| `pessimism` / `tailFraction` | 0.5 / 0.3 | the score is `pessimism` × (mean of the worst `tailFraction` of particles) + `(1−pessimism)` × (mean of all of them), so a move that hangs a piece in some plausible world is penalised without one paranoid particle vetoing every move |
| `infoWeight` | 2 (cp per square) | bonus per square the move would reveal — encourages scouting. Kept well below a pawn so it only breaks ties. |
| `fogClamp` | 1200 (cp) | per-particle score clamp under fog: big enough that losing a queen dominates losing a pawn, small enough that an imagined checkmate from phantom pieces can't swamp a concrete material decision (which was making the AI keep a hanging queen home rather than expose its king to ghosts) |

---

## 3. Adding or changing a parameter

1. Declare the constant where it's used, with a comment explaining *why* that
   value (not just what it does) — this repo has been burned more than once by
   a plausible-looking value that measured worse in actual play (see §2.4,
   §2.5). If there's a measurement backing the number, cite it.
2. Export it, and add it to the aggregate: `src/settings.js`
   for a chess parameter, or `vendor/obscuro/src/settings.js` upstream for a
   generic one (which then needs a submodule bump here).
3. Read it through `param('chess.<NAME>', <NAME>)` (see `src/config.js`) rather
   than reading the constant directly, and add `<NAME>` to `SETTING_PATHS` in
   `src/config.js` so it can be overridden. Read it at *use* time, not at
   module load: the production agent is a singleton constructed at import,
   before any host can configure it. `test/settings.test.js` asserts the key
   space and the aggregate's export list stay identical, so skipping this step
   fails the suite.
4. Add a row to this document — or to `vendor/obscuro/docs/PARAMETERS.md` if
   it's a generic knob.
5. If it affects play strength, measure before and after — `move-quality.mjs`
   and `strength-belief.mjs` are the existing harnesses for exactly this, and
   `--set <NAME>=…` is how to sweep it without editing the default
   ([SETTINGS.md](SETTINGS.md#running-a-sweep)).

**What is deliberately *not* a parameter.** Numbers that define the *model*
rather than tune it stay `const`, because changing one invalidates something
fitted against it:

- `pieceTables.js` (`PIECE_VALUE`, `PST`) and the move prior's own king value of
  1000 are **features**, shared by `fit-move-prior.mjs` and by serving. Making
  one settable would let a host serve the fitted weights against a different
  feature definition from the one they were fitted on — a train/serve skew with
  no error message.
- Encoding and memoisation internals (`exactBelief.js`'s Zobrist tables and its
  25-Elo prior cache step, `stockfish.js`'s engine tag) are bookkeeping: they
  change how a value is stored or keyed, not what it is.

Everything else that changes what the AI plays, or how much work it does per
move, is settable — including the two knobs a preset needs to state the paper's
setup, which used to be reachable only through an env var
(`REFUSED_CHILD_CAP`) or not at all (`CHESS_AGENT_SCORING`).
