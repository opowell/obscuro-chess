# Obscuro chess: playing strength — what's done, what's left

Standalone reference doc, safe to read cold in a new session. Consolidates
the search-correctness work, the belief and move-prior system, and the
analysis-panel work that used to live across four separate planning docs,
alongside the open roadmap for playing strength. `NEXT STEPS.md` points here.

## The ask

`NEXT STEPS.md`'s standing question: *"What is the key to improving
strength? Converting belief accuracy into strength is the open problem, and a
bigger corpus is not on its own the answer."* Part 1 below is everything
already built that the answer has to work with. Part 2 is the prioritized,
independently-measurable roadmap for what's left.

## Part 1 — what's been done

### Search correctness: the equilibrium search plays sound fog chess

The generic search (`vendor/obscuro`) implements the paper's algorithm — GT-CFR
tree growth, PCFR+, the KLUSS Resolve/Maxmargin safety gadget, purification —
and the chess layer's job is to hand it a consistent belief and a real leaf
evaluation. Several structural bugs in how that handoff worked were found and
fixed:

- **Root-infoset fragmentation.** Infoset keys used to be recomputed per
  sampled belief world, so imagined hidden pieces landed each world in its own
  singleton root infoset — the fog search was effectively a single-world
  search. Every root world is now forced into one shared root infoset
  (`gtcfr.js` `expandRoot`).
- **Check-filtered action sets broke the infoset invariant.** An earlier fix
  for a phantom-win bug filtered "in check" moves out of the tree's legal-action
  set, but check depends on hidden pieces, so the legal set differed across
  worlds sharing one observation. The tree now uses the real fog pseudo-legal
  action set and punishes self-check by value (a suicide move evaluates to
  −SEARCH_WIN for the mover) instead of filtering it — which also re-fixes the
  phantom-win bug, correctly this time.
- **`uCond` now uses full reach** `π(h) = reachMe·reachOpp` (paper App. B.1)
  instead of the acting player's own reach alone, so root values, PUCT
  expansion and the analysis panel all see the opponent's actual pressure.
- **Terminal values are bounded** (`SEARCH_WIN = 8000`, vs. unbounded `±10⁶`)
  so one phantom world where the enemy king looks capturable can't swamp
  everything; a certain loss (own king hung) is asymmetrically penalized versus
  a phantom capture of the enemy king, which stays capped at `+LEAF_CLAMP`.
- **Belief-generation fixes**: forced-capture-square inference places a real
  piece (weighted by proximity and inverse value, favoring the least valuable
  recapturing piece) instead of a random one; phantom self-check worlds are
  rejected during sampling; `Belief.beginTurn` is idempotent per turn.
- **Expansion respects the KLUSS gadget** — root-world sampling reads the
  gadget's class reach (`tree.gadget`) instead of raw belief mass, per the
  paper's Fig. 12.
- **Gadget alternate-value calibration** uses the world's own engine-informed
  best-child value (`alt = min{ṽ(h), v*}`) instead of a static heuristic,
  fixing a class-tunnel-vision failure mode where the strategy optimized
  against 1–2 junk worlds.
- **Node-level tree carryover**: the previous move's entire solved tree is
  kept and grafted onto the new root, bringing carried alternate values and
  true opponent-class identity forward; freshly sampled worlds are singleton
  classes with a perfectly-informed opponent, matching the paper's Fig. 9
  line 13.
- **Sequence-scoped opponent infosets**: the opponent's in-tree infosets key
  on their chain-hashed observation sequence; our own infosets deliberately
  stay Markov-keyed (player/turn/board) — a safe restriction on our own
  strategy that buys transposition sharing without underestimating the
  opponent.

Validated end to end by seat-swapped self-play: king-hang-with-a-safe-move-
available rate holds at ~1.4% (95% CI 0.9–1.95%) across power levels,
comfortably under target, with no separate safety backstop needed — play is
equilibrium-driven.

### Leaf evaluation correctness

Two silent-failure modes in the Stockfish leaf evaluator were found and fixed:

- **The engine refuses to score positions that are illegal in standard chess
  but ordinary under fog** (enemy king en prise, kings adjacent, side-to-move
  can capture the king) — a `multiPV` call on such a node returns zero lines,
  and the caller silently fell back to the static evaluator for every child.
  `scoreChildren` now detects a refusal in advance and prices each child
  individually (capped at `REFUSED_CHILD_CAP = 8`, best-static-score first)
  instead.
- **The evaluator was being fed genuinely illegal FEN positions** from two
  sources: castling rights that contradicted the board (now derived from
  board state — a right is only claimed when the king and matching rook are
  actually on their home squares), and impossible piece placements (a pawn on
  its own first rank, or a piece type mismatched with a promotion-rank square)
  surviving belief's contradiction-fallback and particle sampler. Both are now
  filtered at the source.

### Exact belief tracking — the position set P

`src/exactBelief.js` maintains every position consistent with the full
observation history (the paper's belief set P), advanced one opponent ply per
turn and filtered against each new observation. It's built on a compact
representation — `Int8Array(66)` positions, array-based fog move generation
mirroring the rules engine, a fixed-seed Zobrist hash for dedupe — that is
roughly an order of magnitude faster per candidate and smaller per position
than an earlier object-based version, which is what lets the tracked-set cap
sit at 200,000 rather than the tens of thousands it started at. When
exactness is lost (cap exceeded, a time guard trips, or the tracker attaches
mid-game), play falls back seamlessly to a heuristic particle belief
(`src/belief.js`) kept in lockstep, and a re-acquisition path rebuilds a
(superset) exact set once few enough pieces are hidden to make the
cross-product tractable again.

### A weighted belief, and a fitted move prior

The belief set started as literally a *set* — every consistent position
equally likely, because nothing modeled how the opponent actually chooses
moves. Two changes turn it into a real posterior:

- **The mechanism**: `exactBelief.js` carries a weight per position, and
  positions reached by multiple histories accumulate weight (a Map-based
  dedupe that sums on collision) instead of the second arrival being silently
  dropped. This alone — even feeding it a uniform per-move prior — produces a
  non-uniform posterior, because a state reachable from several parents
  accumulates their mass and a parent with fewer legal moves passes more mass
  to each child.
- **The model** (`src/movePrior.js`): π(move | position), a conditional-logit
  softmax over a handful of O(1)-per-move features (capture value, promotion
  value, a piece-square-table delta, a castling bonus), fitted by
  maximum-likelihood on recorded games (`scripts/fit-move-prior.mjs`) rather
  than hand-tuned. Per-term weights matter because the terms want very
  different effective temperatures — one shared temperature can't serve all
  nine parameters at once.

Consumers were updated to actually use the weights: `samplePositions` draws
weighted (without replacement), `rankByLikelihood` (renamed from a
marginal-probability surrogate that existed only because there was nothing
better) sorts by real posterior weight, and the analysis panel's aggregates
are mass-weighted rather than count-weighted.

Belief worlds are now also **drawn by the posterior** rather than uniformly
(`sampleAlpha = 1`) — the search samples the opponent's plausible moves more
often than their implausible ones, rather than treating every consistent
position as equally worth searching. A separate CFR-reach-weighting knob
(drawn-world *importance* inside the solve) is subsumed by this and now inert
by default, since weighting an already-posterior-drawn sample by the same
posterior again would double-count it.

Supporting infrastructure that shipped alongside the model: a corpus loader
handling directories, zips, PGN (with ratings) and session/crawl JSON behind
one interface (`corpus.js`/`pgn.js`/`zip.js`); a rating-conditioning mechanism
that tilts every prior weight continuously by the opponent's rating rather
than bucketing (`RATING_SLOPE`, `RATING_PIVOT`/`SCALE`/`Z_CLAMP`); a
belief-calibration harness (`scripts/calibrate-belief.mjs`) measuring how much
probability the belief puts on the true position; a paired per-move cp-loss
harness against a deep reference search (`scripts/move-quality.mjs`), hardened
for measurement determinism (forces a fresh engine hash per search, and gives
the caller ownership of the engine-worker respawn boundary, since a mid-run
recycle desyncs a paired comparison); a seat-swapped self-play strength
harness (`scripts/strength-belief.mjs`); and a one-command adoption pipeline
(`scripts/adopt-corpus.mjs`) that measures ingest health, fits, and refuses to
ship a refit that doesn't beat the shipped weights on the belief-calibration
gate.

### The analysis panel: exhaustive belief walks and iterative deepening

Scoped to the read-only analysis panel, not real move selection. Perfect
information and fog are now one code path rather than two: a fully-known
position is the degenerate case of a belief population of size 1, handled by
the same batched-walk machinery as a genuinely fogged position with thousands
of consistent worlds. That walk enumerates the belief population without
replacement in batches (rather than resampling with replacement forever),
reports `{evaluated, total, exhaustive}` coverage, and — for the exact-belief
case — actually converges and stops once every consistent position has been
scored. Each world's leaf evaluation runs an iterative-deepening ladder up to
`MAX_SF_DEPTH = 30`, live depth ticks forwarded into the same progress stream,
and an in-flight deep search can be interrupted (`UCI stop`) so cancelling
feels responsive even mid-search. The whole thing also runs client-side in a
browser Web Worker (`src/stockfish.js` is browser-safe behind a Node/browser
guard, a `/lib` static route serves the module graph to the worker, and a
`cp-eval` endpoint supplies the leaf evaluation the browser has no local
Stockfish for) so analysis never blocks the server or the UI thread.

### Settled research findings

Durable conclusions worth not re-deriving:

- **Depth-1 leaf evaluation — the paper's own design point — does not
  transfer at this engine's tree scale.** The paper's strength comes from
  aggregating ~10⁶-node trees over shallow leaves; this engine's trees are
  roughly two orders of magnitude smaller, so leaves have to carry tactics the
  tree itself cannot find. A paired measurement over thousands of positions
  found leaf depth 7 beats depths 1/2/4 decisively, with 1/2/4 statistically
  indistinguishable from each other.
- **A fitted move-prior beats a uniform one substantially** on belief
  calibration (log-loss of the true position), and the belief-set weighting
  mechanism alone (even before any real model) already beats an unweighted
  flat posterior.
- **The capture term contributes little; piece-square-table deltas and a
  castling bonus carry almost the entire signal.** An earlier hypothesis that
  captures would dominate (real players take material) did not hold up.
- **The king piece-square-table term is noise, not signal, at adequate corpus
  size.** An early, small corpus produced a confidently negative king weight
  ("fog players walk their king out") that looked like a real qualitative
  finding; on a much larger corpus the term's sign flips across
  cross-validation folds and carries no consistent signal — it was one
  player's habit, not a property of fog play.
- **Over-sharpening a prior is catastrophic, not just wasteful.** Past a
  certain temperature the belief becomes worse than assuming nothing at all,
  even though a naive ranking metric (median rank of the truth) keeps looking
  like it's improving right through the collapse — log-loss, not rank, is the
  metric that catches this. A small uniform-mixing floor bounds the worst
  case.
- **Opponent-conditioning the move prior — by rating, by actor type, or by
  strength bucket — has not beaten the pooled model, in three separate
  attempts.** Populations plausibly do differ, but there isn't yet enough data
  to estimate the extra parameters without the added variance costing more
  than the reduced bias buys back. Self-play cannot supply the missing data
  either: fitting the prior on self-play games makes it a model of this engine
  rather than of real opponents, which measurably hurts play against humans.
- **A better-calibrated belief does not automatically play better** — it
  reaches move selection through exactly one channel (which worlds the search
  draws), and until that channel is actually turned on (see belief-weighted
  sampling above), sharpening the belief only changes what the analysis panel
  displays, not what the AI plays.
- **Distilling the Stockfish leaf evaluator into a small trained network does
  not work at this project's data scale.** Two independent architecture
  attempts (a small dense net, and a version with king-relative input features
  mirroring NNUE's approach) both plateau far short of even a depth-1 real
  search on ranking quality and move agreement. The gap tracks data volume,
  not architecture — a real leaf search's target evaluator (Stockfish's own
  NNUE) trained on roughly three orders of magnitude more positions than this
  project's largest measurement run produced, and neither self-play throughput
  nor an outcome-based training target closes that gap at a practical
  timescale.
- **Weighted tail-pruning of the belief set at its capacity limit was
  considered and deliberately not implemented.** It would trade the tracker's
  central invariant — the true position is always in the tracked set — for a
  softer failure mode, and the positions a prior would prune are
  disproportionately the surprising ones, which is exactly when the true
  position is most likely to be one of them.

## Part 2 — what's left to do

### Ship what's already measured but not deployed

Two low-risk items where the measurement exists and only the shipped constant
hasn't moved yet:

1. **Raise the leaf-search depth ceiling** (`CHESS_DIAL.leafEval.sfDepth` in
   `src/ObscuroAgent.js`, currently `{min: 2, max: 4}`). The settled finding
   above (depth 7 beats 1/2/4 decisively) argues for raising the ceiling, but
   that measurement traded depth against search-round count as its own
   experimental design, while production spends its round budget independently
   (`search.DIAL.power.maxRounds`). Re-run `move-quality.mjs --grid` at actual
   production dial settings before picking a new ceiling — don't just copy the
   isolated grid number — then raise `sfDepth`'s `max` (7 is the
   best-supported starting point) and update `docs/PARAMETERS.md` §2.2 per
   that doc's own parameter-change checklist.

2. **Reduce belief-tracking censoring around posterior-weighted sampling**
   (`CAP` and `TIME_GUARD_MS` in `src/exactBelief.js`, currently 200,000 /
   4,000ms). At those defaults, exact tracking is abandoned — stickily, for
   the rest of the game — on a meaningful share of the high-`|P|` turns where
   posterior-weighted sampling's benefit concentrates, which measurably drags
   down the benefit versus tracking exactly all the way through. Exact-belief
   updates are cheap on their own (roughly a microsecond per candidate on the
   compact representation), so raising these two constants toward
   production-safe values may reclaim a real fraction of that loss for a
   modest latency cost — measure per-move wall-clock and give-up rate at a few
   intermediate settings under actual production round/world budgets before
   choosing new defaults.

### Close the "does it actually win games" gap

The highest-priority open question: posterior-weighted world sampling shipped
on a cp-loss-vs-deep-reference proxy that is explicitly blind to fog-specific
information value (it can't credit a move for managing what the opponent
does or doesn't get to see, only for matching a perfect-information engine's
opinion of the position). The only actual win/loss self-play measurement
available is small (15 games) and points the opposite direction — far too
underpowered to trust, and predating the harness's determinism fixes.

Next step: a properly-powered seat-swapped self-play run
(`scripts/strength-belief.mjs --arm alpha`), at the scale the script's own
documentation says is needed — hundreds of games, not fifteen. Self-play
throughput is roughly 0.35–2 seconds per ply depending on difficulty, so a few
hundred games at typical game length is multi-hour-to-overnight wall-clock;
games are independent, so parallelize across processes rather than running
serially.

Two outcomes to plan for:
- **Confirmed** (posterior-weighted sampling wins more games, not just lower
  cp-loss): the shipped default is validated on the actual target metric, not
  just the proxy.
- **Not confirmed, or reversed**: the real research question becomes *why* a
  metric that rewards matching a perfect-information referee disagrees with
  actual fog-chess outcomes — one hypothesis worth checking first is whether
  weighting the search's world draw toward the most-likely opponent lines
  under-hedges against surprising-but-plausible ones that the proxy can't
  distinguish from good play under uncertainty. Instrument and measure rather
  than guessing past it.

### Attack the actual resource ceiling

Every item above trades against wall-clock — deeper leaves, a looser belief
cap, more belief worlds, more search rounds all cost more time per move. This
buys back that budget, and its payoff compounds with everything above it.

1. **Parallelize the Stockfish leaf evaluator across a worker pool.**
   Profiling a real move found roughly 80% of move time inside Stockfish
   (batched across belief worlds and node children — embarrassingly
   parallel) and roughly 20% in single-threaded CFR; Amdahl's law caps a
   worker pool at roughly 5× on that split, and the engine backend
   (`src/stockfish.js`) currently uses exactly one worker regardless of how
   many cores are available. Pure engineering, no open research question, and
   it's the single lever that makes every leaf-depth or belief-cap increase
   above cheaper to afford.

2. **Ship a "strongest" preset** (`src/presets.js`, mirroring the existing
   `paper` preset). The generic search's POWER-mode dial is hard-capped well
   below what the engine can otherwise do (worlds, round count and tree-size
   ceilings all bounded fairly low), while TIME mode's ceilings are far
   looser and bounded mainly by the actual clock given — including the
   iterative-deepening ladder the analysis panel already uses, which real move
   selection doesn't currently reuse. For "as strong as possible" whenever an
   opponent's clock allows it, TIME mode with a generous time budget is very
   likely already the strongest configuration this engine can produce — no new
   algorithm work, just the right settings, expressed as a named preset
   instead of a hand-assembled settings file. Measure it against the current
   strongest POWER-mode default before shipping it as a recommendation, same
   as any other preset.

### Belief-tracking scaling and correctness (real play, lower priority)

- **Exact-belief re-acquisition currently yields a superset, not the literal
  history-exact set**, once exactness has been lost and rebuilt from the
  heuristic belief's per-piece possible-squares. If real games show this
  mattering, tightening it is future work, not yet scoped in detail.
- **If the belief-tracking cap (200,000) ever proves tight even after
  parallelizing leaf evaluation**, the follow-on scaling work, cheapest first,
  is: incremental ray-patched visibility, then speculative belief-advance
  during the opponent's thinking time, then a bitboard rewrite of the
  exact-belief representation.
- **Opponent-model-weighted sampling from the belief set beyond today's
  posterior weighting** — the paper's own closing suggestion — remains
  unexplored, as does a learned leaf evaluation for non-chess games built on
  this engine.

### Analysis-panel-only refinements (not real play)

- **A true joint-equilibrium mixing over the whole belief population**
  (rather than today's ensemble average of per-batch equilibria) is a real but
  unresolved research question, not yet attempted: it would need the KLUSS
  safety gadget to grow its opponent-class set mid-solve, which raises an open
  question about whether the paper's own safety proof still holds under a
  growing class set. A safer intermediate worth trying first, if this becomes
  a priority, is periodic gadget rebuild with regret-state carryover
  (warm-starting each rebuild from the previous solve's accumulated regret)
  rather than true incremental injection.
- **The human side of a live game's exact belief is only maintained when
  Obscuro itself is the one moving**, not for every ply a human plays — so
  analysis-panel exhaustion on a human's own game is reachable only late, once
  few enough pieces are hidden. Maintaining it incrementally for every
  committed ply would let mid-game exhaustion happen sooner; deferred because
  it touches the move-commit path and needs every ply fed to the tracker in
  order.

### Data and longer-horizon research

- **Move-prior opponent-conditioning stays unproven, pending more data.** The
  settled finding above (three separate null results) is attributed to data
  volume, not a modeling failure, and self-play cannot supply the missing data
  (it makes the prior a model of this engine rather than of real opponents).
  Only a larger *external* human corpus helps here — further recorded
  fog-chess games beyond what's already loaded — and per the standing open
  question this doc opens with, a bigger corpus alone is explicitly not the
  strength bottleneck, so this stays deprioritized relative to everything
  above.
- **Leaf-net distillation is closed — do not reopen without a data source
  roughly three orders of magnitude larger than what's been tried.** Two
  independent architecture attempts both plateaued at the same data-volume
  wall; capacity and feature changes moved nothing.
- **A "gives check" feature for the move prior was considered and left out**
  — it's not computable in constant time on the typed board representation the
  prior needs, and given that piece-square deltas rather than material carry
  the signal, there's no evidence it would earn back that cost.
- **Explicitly out of scope, not merely deprioritized**: modeling the
  opponent recursively (as a player who models us modeling them, and so on —
  stop at one level, a fixed static opponent model), and correcting the move
  prior's fog asymmetry (it scores from the mover's true position, when a
  principled model would score from what the *mover* could see under their
  own fog — a whole extra belief computation per node, not affordable at this
  budget).

## Verification discipline for every item above

This project has a long, specific history of confident-wrong numbers from
otherwise-reasonable-looking measurements. Anything above that involves a
measurement should follow the same discipline that produced every
trustworthy number already on record:

- **Paired, seat-swapped measurement** — never a raw win/loss tally on
  unswapped seats (white's structural advantage under fog dwarfs any measured
  effect and swamps an unswapped comparison).
- **Run the null control first.** Two identically-configured arms must agree
  essentially 100%. Anything less means the instrument, not the question
  under test, is what moved.
- **A fresh engine hash per search, and no mid-run engine-worker recycling**
  — a chess engine that carries state across searches (transposition tables,
  in this case) is not a pure function of the position, and a worker
  respawned on a cache-insensitive counter desyncs a paired comparison's two
  arms.
- **Watch the exact-belief-tracking health of a run.** A run where a large
  share of positions have already fallen back to the heuristic belief is
  comparing two arms that both collapsed to the same fallback, which several
  of the knobs above are not levers on at all.
- **Each item above lands in its own git worktree**, per the outer
  `battle-simulator` repo's `CLAUDE.md` — this is a shared, alpha-stage
  checkout with other agents potentially active concurrently.

## Key files

- `vendor/obscuro/src/gtcfr.js`, `kluss.js`, `infoset.js`, `purify.js` — the
  generic search's tree growth, safety gadget, CFR value propagation and
  purification; the search-correctness fixes above live here.
- `src/ObscuroAgent.js` — the chess leaf evaluator (`scoreChildren`,
  `makeChessLeafEval`), difficulty scaling, the analysis panel's batched
  population walk (`analyzeObscuroProgressive`), `CHESS_DIAL.leafEval.sfDepth`.
- `src/exactBelief.js` — the exact position set P: representation, `CAP`,
  `TIME_GUARD_MS`, `SAMPLE_ALPHA_DEFAULT`, re-acquisition.
- `src/belief.js` — the heuristic particle-belief fallback.
- `src/movePrior.js` — π(move | position): the fitted model, rating
  conditioning, the uniform baseline.
- `src/FogChess.js` — the game definition tying belief, prior and search
  together; `beliefPopulation`/`enumerateWorlds` for the analysis panel.
- `src/presets.js` — named configurations (the `paper` reference point; where
  a "strongest" preset belongs).
- `src/stockfish.js` — the engine backend; browser/Node dual-mode, the
  single-worker bottleneck a pool would parallelize.
- `scripts/move-quality.mjs` — paired cp-loss harness, incl. `--grid` for
  depth measurements.
- `scripts/strength-belief.mjs` — seat-swapped self-play strength harness.
- `scripts/calibrate-belief.mjs`, `scripts/fit-move-prior.mjs`,
  `scripts/adopt-corpus.mjs` — belief calibration, prior fitting, and the
  one-command adoption pipeline.
- `docs/PARAMETERS.md` — every constant named above, with its own citation;
  update it alongside any constant this doc's items change.
