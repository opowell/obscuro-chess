# Fog-of-war chess AI — king-safety fix plan

Standalone working doc. Written 2026-07-13, substantially executed later the same day.
Safe to read cold in a new session.

## Status: root causes found and fixed (Phases 0–3 done; Phase 4 partially)

The search-side king-safety bug is fixed and regression-tested. Measured on the
`febb71bf` blunder position (power 80, 8 seeds, replayed belief):

| | king-hangs | Kf7 in top candidates | value hallucinations (+1500 in a lost position) |
|---|---|---|---|
| before | 1/8 played (guard rescued 2 more; search ranked Kf7 top in 3/8) | 3/8 | 2/8 |
| after  | **0/8** | **0/8** | **0/8** |

After all fixes the search converges near-deterministically to Ne7/Qe7 (sound
defensive moves) with stable values (−715…−1065). On the `befd4820` bishop position it
now plays Stockfish's best move (fxe6) in 5/6 seeds and SF's #4 (Nf6) in the sixth.

Regression harness: `test/fog-blunders.test.js` (replays both recorded
blunder sessions with the real belief lifecycle and seeded RNG).

## Symptom (historical)

The fog-of-war chess AI (`ChessObscuroAgent`, the Obscuro/CFR equilibrium search) played
bad moves at high power levels — most visibly it **walked its own king onto a square an
unseen enemy pawn attacks** (hanging the king), and **thought it was winning a lost
position** (search value `+1500`) while ignoring a hanging bishop.

Repro sessions (recordings under `sessions/`):
- `2026-07-12T23-27-55-befd4820.json` — power 86, played `b6→b5` with the d7 bishop hanging.
- `2026-07-13T12-59-56-febb71bf.json` — power 80, played `e8→f7` (king) into White's e6 pawn.

## Root causes (as actually found — several differed from the original hypotheses)

0. **Root-infoset fragmentation (the deepest one).** Infoset keys are the acting
   player's observation *recomputed inside the sampled world* — but imagined hidden
   pieces change blocking/visibility, so each belief world recomputed a slightly
   different observation and landed in its OWN singleton root infoset. The strategy the
   agent purified was supported by only the world(s) whose key happened to match world
   0: the fog search had effectively been a **single-world search** all along (measured:
   root visitReach 1/42). Fixed in `gtcfr.js`: `expandRoot` forces every root world into
   the one shared root infoset (they are by construction in the searcher's real
   infoset); `expandNode` takes a `forceInfoset` override. After this fix the root
   values equal the belief-weighted mean and the chosen move matched Stockfish's best
   on the befd position across seeds.

1. **Check-filtered in-tree action sets violated the infoset invariant (THE big one).**
   `getSearchLegalActions` (an earlier fix for a phantom-win bug) gave the tree
   check-filtered moves. But check depends on *hidden* pieces, so the "legal set"
   differed across belief worlds sharing one observation — breaking the rule that every
   node in an infoset has the same action set. Concretely: in exactly the worlds where
   `Kf7` walks into the unseen e6 pawn, `Kf7` vanished from that world's move set, so
   `cfrDescend` scored it as a **neutral pass at material value** instead of −WIN. The
   dangerous worlds never voted. Fix: the tree now uses the REAL fog action set
   (pseudo-legal — observation-determined, as FoW chess rules guarantee), and self-check
   is punished by VALUE: such children evaluate to −SEARCH_WIN for the mover, new
   infosets are seeded to the best child, and CFR keeps suicide out of both players'
   strategies (which also re-fixes the original phantom-win bug, properly this time).

2. **uCond was not the paper's conditional value.** `infoset.js` weighted an infoset's
   nodes by the acting player's own reach only, so root values averaged belief worlds
   uniformly no matter how the KLUSS gadget's opponent concentrated on dangerous
   classes. Now weighted by full reach π(h) = reachMe·reachOpp (paper App. B.1), so
   values, PUCT expansion, and the analysis panel all see the gadget's worst-case
   pressure.

3. **Unbounded terminal values.** In-tree terminals were ±10⁶ while leaf material was
   clamped to ±1500, so one phantom world where the enemy king looked capturable
   swamped everything. The paper bounds utilities to [−1,+1]. Now games can set the
   scale (`ObscuroAgent._winValue` / `game.winValue`); chess uses SEARCH_WIN = 8000,
   and "hung own king" = −SEARCH_WIN exactly (a certain loss, not "down 80 pawns").

4. **Belief inconsistency** (`src/belief.js`):
   - The forced-capture-square inference (piece of ours just captured on a now-unseen
     square) back-filled a phantom piece of random/queen type while also scattering the
     real candidates elsewhere. It now places a REAL unseen piece there first, weighted
     by proximity AND inverse piece value (recaptures use the least valuable piece):
     measured on `febb71bf`, e6 = pawn in 39/50 worlds (truth: pawn), up from 26/50,
     with queens/rooks/kings on e6 down from 14/50 to 3/50. This mattered because a
     wrongly-imagined ROOK/QUEEN on e6 gives *check*, and worlds that start in check
     make every quiet move price like death — pushing king walks to the top.
   - Phantom self-checks (in-check worlds with no capture evidence) are rejected during
     sampling (with an escape hatch when non-check worlds are scarce). In-check worlds:
     12/50 → 3/50.
   - `Belief.beginTurn` is now idempotent per turn (keyed by turnNumber): the king-safety
     guard's re-sample used to advance the belief an extra phantom opponent ply per move.

5. **Expansion ignored the gadget.** One-sided GT-CFR sampled root worlds by prior
   belief mass; per the paper (Fig. 12) the descent starts at the gadget root where the
   opponent selects the class J with its current reach π_▼(J). Fixed in `gtcfr.js`
   (`sampleWorld` now reads `tree.gadget`).

6. **Gadget alternate-value miscalibration → class tunnel vision.** With per-world
   singleton classes, alt values on a different scale from the resolved enter values
   made enter/exit go one-hot: measured, the opponent exited 41/42 classes (piv sum
   0.043) and the whole strategy optimised against 1–2 junk worlds. Fixed in
   `kluss.js`: alt = min{ṽ(h), v*} with ṽ(h) the world's engine-informed best-child
   value (the paper's "Stockfish's evaluation of h"), not a static heuristic.
   (A ½·α reach FLOOR was temporarily added as a second mitigation; after tree
   carryover supplied real carried alternate values it was removed again — the pure
   paper blend validated clean. Same for a since-removed purify shortcut that forced
   pure play at ≥0.9 mass. If tunnel vision ever reappears, fix alt calibration, do
   not re-add the floor.)

Also fixed while in there (paper fidelity, not king-safety):
- **Purification** now shifts ALL excluded probability onto a\* (paper App. C.8) instead
  of renormalising across the kept support.
- **Perfect-info power 100** now collapses to pure (deterministic) Stockfish best-move
  play; below 100 the proportional-sampling dial is unchanged. Time mode already played
  pure best.

## Follow-up round (same day): open items executed

- **Exact position set P** (`src/exactBelief.js`, the paper's belief): all
  positions consistent with the full observation history, advanced one opponent ply
  per turn (fog-pseudo-legal, minus king-captures-us successors), filtered inline
  against the current observation (own pieces exact, visible squares exact, exact
  visibility-set reproduction), deduped as STATES (fn. 21), capped at 30k with a 3s
  time guard. Sampling is uniform (paper §3). On give-up (mid-game attach, cap,
  time, contradiction) play falls back seamlessly to the heuristic particle belief,
  which is kept in lockstep every turn. While |P|=1 the agent literally knows the
  board. Invariant test: `test/exact-belief.test.js` (true position ∈ P at
  every turn of both replayed blunder sessions). Live measurement at power 80: exact
  held ~27/40 opening plies, avg |P| ≈ 2.3k, max ≈ 15k.
- **Guard-vs-belief commit bug**: the generic agent committed its chosen action to
  the belief trackers before `_kingSafetyGuard` could swap it, silently corrupting
  the belief (fatal for exact P). Selection adjustment is now a subclass hook
  (`_adjustChosenAction`) that runs BEFORE `onActionCommitted`; overrides are flagged
  in `lastAnalysis.adjusted`.
- **Turn-scoped infoset keys** (`infoset.js observationKey`): keys now include
  (player, turnNumber), so identical visible boards at different plies no longer
  merge into one strategy; absolute turn numbers keep blueprint warm-starts valid.
- **Minimum root-world expansion** (`gtcfr.js expandRoot`): at least 8 root worlds
  expand even past the deadline, so a cold engine cache can't silently degrade the
  search to near-single-world. Root worlds expanded lazily later also join the
  shared root infoset (rootWorld tag).
- **Leaf depth recalibrated** (`src/ObscuroAgent.js _leafEval`): 2..7 by
  dial (was 2..10). The paper uses depth-1 leaves with ~10⁶-node trees; our trees
  are far smaller so leaves must price short tactics — depth 6–7 is where Stockfish
  folds "quiet move → loses the attacked piece" into parent MultiPV scores — but
  depth 9+ made cold root expansion so slow it reintroduced few-world search.
  Search budget top raised to ~2s (power 100).
- **Alternate values, resolved structurally**: because worlds are re-sampled fresh
  each move (only regrets/values warm-start), every root class is a "newly sampled"
  state in the paper's terms, and the paper's own formula for those — min{ṽ(h), v*},
  perfectly-informed opponent — is exactly what the code does. The carried-tree
  branch u(x,y|J) − ĝ(J) (Fig. 9 l. 8) only becomes meaningful with node-level tree
  carryover (future work below).

## Phase 4 validation (instrumented self-play, one agent instance per side)

4 games at power 50 and 80, 4 at power 100 (two chunks), fog on, seeded:

| power | plies | king-hangs w/ safe move | mixed CFR moves | guard fires | avg latency | exact-P plies |
|---|---|---|---|---|---|---|
| 50  | 234 | 4 (1.7%) | 129/212 (61%) | 1 | 774ms | 130/234 (56%), avg P 793 |
| 80  | 373 | 3 (0.8%) | 186/334 (56%) | 2 | 1508ms | 314/373 (84%), avg P 1008 |
| 100 | 391 | 3 (0.8%) | 187/360 (52%) | 1 | 2060ms | 358/391 (92%), avg P ~920 |

- King-hang rate meets the Phase-0 target (<2%) at every power; the few remaining
  hangs are plausibly calculated risks in lost positions (paper App. E.7), not bugs.
- **Mixing confirmed in live play** (open item resolved): with v* carryover the agent
  plays genuinely mixed strategies on half its CFR moves; fresh-agent single-position
  probes sit in Resolve (pure) because v* = ∞ there — expected, not a defect.
- `_kingSafetyGuard` fired 3 times in ~750 plies (0.4%): play is equilibrium-driven.
  (Initially kept as a backstop; REMOVED after the 24-game validation below showed
  zero-guard batches meet the target on their own — Phase 4 complete.)

## Second follow-up round: the former future-work list, executed

- **Node-level tree carryover** (`gtcfr.js harvestCarried`/`attachCarriedRoot`,
  `search.js`, agent `_carry`): the previous move's ENTIRE solved tree plus the
  action actually played is kept per side; the our-move grandchildren consistent
  with the new observation are grafted in as root worlds with subtrees, infoset
  objects, regrets and values intact (remapped onto the pinned root action set,
  infoset membership rebuilt from reachability). Carried worlds bring the paper's
  carried alternate values u(x,y|J) − ĝ(J) (gift ĝ from the old opponent infoset's
  uCond, App. C.2) and their true opponent-class identity (J′, b). Freshly sampled
  worlds are now SINGLETON classes with a perfectly-informed opponent — exactly
  Fig. 9 line 13. Live: grafts on ~72% of plies, ~7 carried worlds per move
  (test: `test/carryover.test.js`).
- **Sequence-scoped opponent infosets** (`infoset.js chainHash`, `gtcfr.js`): the
  opponent's in-tree infosets key on their chain-hashed observation SEQUENCE
  (own actions + observations at their decision nodes), seeded per root world.
  OUR infosets deliberately stay Markov-keyed (player|turn|board): coarsening
  one's OWN information is a safe strategy restriction that buys transposition
  sharing, while coarsening the OPPONENT's would underestimate their play.
- **Exact-belief longevity** (`exactBelief.js`): cap 30k → 50k, guard 3s → 4s
  (max observed |P| was 26k), and RE-ACQUISITION: when exact tracking was lost,
  P is re-enumerated from the heuristic belief's per-piece possible-sets once
  they are small (cross-product bound 60k), yielding a tight SUPERSET (marked
  `approx`) that is advanced exactly thereafter. Guards: refuses truncated or
  possibly-promoted possible-sets (belief.js now flags truncation and adds
  castle destinations so sets stay supersets); forced-capture squares must be
  occupied. Live at power 80 the exact set now holds ~100% of plies, so
  re-acquisition is a rare-path safety net.
- **b5-class inaccuracy: gone.** With singleton fresh classes + carried values,
  12/12 seeded runs on the befd position choose engine-approved moves (fxe6 ×8,
  Nf6 ×4); febb remains 0/8 king-hangs with stable Ne7/Qe7 play.

Post-carryover self-play validation — CLOSED (24 games / 1787 plies, power 80,
seven seeded batches): king-hangs-with-safe-move 25/1787 = **1.4%** (95% CI
0.9–1.95%), entirely under the 2% target and not significantly different from
the pre-carryover 0.8% (z ≈ 0.9, p ≈ 0.35); the early "2.1%" reading was
small-sample noise. Grafts on ~65–75% of plies (9–13 carried worlds/move),
exact P held 60–95% of plies with re-acquisition firing 67 times (a real
working path), max |P| 40.5k (under the 50k cap), latency stable ~1.5s avg,
mixing ~45–50% of CFR moves. The guard fired 6/1787 plies (0.34%); batches
with ZERO guard fires still measured 1.0–1.3%, so the search meets the target
on its own — `_kingSafetyGuard` was therefore DELETED (2026-07-16): play is
genuinely equilibrium-driven, completing Phase 4. The generic
`_adjustChosenAction` hook it rode on remains as a documented extension point.

## Third follow-up round: exact-P capacity (Tier 1)

`exactBelief.js` was rewritten on a compact representation: positions are
Int8Array(66) (64 signed piece codes + castling bits + en-passant index), with
array-based fog move generation and application (mirroring `moves.js` exactly,
castle quirks and 4-way promotions included), visibility as a 2×32-bit mask
compared with two integer equalities (mirroring `getVisibleSquares`, pawn-block
rule included), and a fixed-seed 53-bit Zobrist hash for state dedupe. ~10×
faster per candidate and ~15× smaller per position → CAP raised 50k → 200k
within the same 4s guard. The public API (beginTurn/commitOurMove/
samplePositions/tryReacquire) is unchanged; sampled worlds synthesize piece ids.

That surfaced a latent identity bug: `observationKey` (and the carried-world
dedupe) serialized boards INCLUDING piece ids, so synthesized-id worlds never
matched engine-id carried nodes and tree carryover silently died. Piece
identity is not observable — keys now use an identity-free canonical board
serialization (`infoset.js canonicalBoardSig`: owner+type per square).

Measured (4 games, power 80): max |P| 128k (old cap clipped at 41k), avg |P|
held 7.2k (was ~1–3k), ~1µs per advance candidate, per-move latency max DOWN to
2.4s (was 3–5.4s); 91 plies re-acquired; hangs-with-safe 0.9%, mixing 83%,
carryover grafting 81% of plies. The remaining exactness ceiling is the 4s time
guard in explosive middlegames, not memory. All rollout games (14/15, `aow`
pre-existing) unaffected by the identity-free keys.

## Future work (remaining)

- Re-acquired P is a superset, not the literal history-exact set.
- Deeper OUR-infosets remain Markov-keyed by design (documented trade-off above).
- Opponent-model-weighted sampling from P (paper's closing suggestion) and a
  learned leaf evaluation for the non-chess games remain unexplored.
- If 200k ever proves tight: incremental ray-patched visibility and speculative
  advance during the opponent's thinking time (Tier 2), bitboards (Tier 3).

## Key files
- `src/ObscuroAgent.js` — chess leaf eval (`makeChessLeafEval`, `LEAF_CLAMP`,
  `SEARCH_WIN`/`KING_HANG`), `_winValue`, difficulty scaling, perfect-info Stockfish
  path (pure at power 100). (The former `_kingSafetyGuard` backstop was removed
  after Phase-4 validation.)
- `vendor/obscuro/src/ObscuroAgent.js` — generic agent, `_config` difficulty knobs, `_winValue` +
  `_adjustChosenAction` hooks (the latter currently unused; `.adjusted` in
  `lastAnalysis` flags an override), `lastAnalysis`.
- `vendor/obscuro/src/search.js` — `makeHooks` (action-set hook caveats), `runObscuroSearch`.
- `vendor/obscuro/src/kluss.js` — Resolve/Maxmargin gadget; engine-informed alternate
  values; opponent reach floored at ½·prior.
- `vendor/obscuro/src/gtcfr.js` — tree growth; gadget-driven root sampling; shared root
  infoset (`forceInfoset` / `rootWorld`); minimum-root-worlds floor.
- `vendor/obscuro/src/infoset.js` — CFR value propagation; `uCond` full-reach weighting;
  turn-scoped `observationKey`.
- `vendor/obscuro/src/purify.js` — excluded mass → a*.
- `src/exactBelief.js` — exact position set P (+ `exact-belief.test.js`).
- `src/belief.js` — heuristic fallback: recapture inference, phantom-check
  rejection, idempotent `beginTurn`; `src/FogChess.js` prefers exact P and
  keeps both trackers in lockstep.
- `test/fog-blunders.test.js` — Phase 0 regression harness (SEEDS env scales).

## Repro / measurement recipe
Replay a recording to a decision point and run the agent (Node ESM, Stockfish auto-loads):
see `test/fog-blunders.test.js` (`replaySession` drives the belief lifecycle:
`sampleWorlds` → `onActionCommitted` per AI move, then `applyActions`). King-hang check:
apply the candidate on the TRUE state and `isAttackedBy(kingSq, enemy)`.
Belief probe: `ChessGame.sampleWorlds(view, "black", N, rng)`, inspect `world.board.e6`
and `isAttackedBy(world.board, "f7", "white")`.
Note: the live game shares ONE agent instance across moves (KLUSS blueprint/prevValue
carryover); a fresh agent per call does not. Belief is per (players-array, colour) and
now advances at most once per turnNumber regardless of how often sampleWorlds is called.

## 2026-08-01 — the leaf evaluator is silently blind in ~10% of positions

Found while chasing an unrelated timing discrepancy, and it is the more important
result: **Stockfish returns NOTHING for positions that are illegal in standard
chess but ordinary under fog**, and `scoreChildren` then quietly substitutes the
static JS evaluator for every child of that node. Measured directly (multipv 30,
depth 4):

| position | engine lines |
|---|---|
| ordinary middlegame | 30 |
| **enemy king en prise** | **0 — engine refuses** |
| **kings adjacent** | **0** |
| **side to move can capture the king** | **0** |

Under fog these are not edge cases: there is no check rule, so "we are attacking
the enemy king" is a normal, frequent state — exactly the tactically sharpest one.
A clean instrumented run of `move-quality.mjs` measured **10.19% of leaf
evaluations (11,106 of 108,944) falling back to the static evaluator**, with no
competing load and zero truncated rungs.

**READ THAT NUMBER WITH ITS CONFIGURATION.** It was measured on the *depth-1*
grid arm, which is NOT what ships — the dial tops out at leaf depth 4. Re-measured
at depth 4: **~1% fallbacks and 1–2 refused nodes per 800 engine calls.** The
refusal is real and the mechanism is real, but its frequency is strongly
depth-dependent, and at the shipped depth it is a small effect, not a tenth of
the search. Do not quote the 10% as a property of the AI.

Why it stayed invisible: `scoreChildren` computes an `engineOk` flag, and the
fixed-depth wrapper `makeChessLeafEval` throws it away (`.scores`). The search
carries on with worse numbers and reports nothing. `getLeafEvalStats()` /
`resetLeafEvalStats()` (ObscuroAgent.js) now count engine leaves, fallback leaves
and truncated rungs, and `move-quality.mjs` prints them under every result — a
run with a nonzero fallback share is not comparable with a clean one.

### The fix, and what it was actually worth

`scoreChildren` now detects a position the engine will refuse (`engineWouldRefuse`:
the side not to move is in check, the kings are adjacent, or a king is already
gone) and, instead of making a doomed MultiPV call on the parent, prices the
CHILDREN individually — each child is legal, since there the opponent is merely in
check — negating cp onto the mover's scale. Capped at `REFUSED_CHILD_CAP` (8,
best-static-score first) because refused nodes would otherwise multiply engine
work several-fold; the tail keeps the static evaluator, which is what the whole
node used to get.

A/B on identical positions at the SHIPPED leaf depth (2 games, both seats, 50
moves, `--grid 4:8`):

| | mean cp loss | median | ms/move | fallbacks | refused nodes |
|---|---|---|---|---|---|
| cap 0 (old behaviour) | 87.4 | 58.0 | 712 | 0.96% | 1 |
| cap 8 (the fix) | 87.6 | 55.0 | **662** | 1.02% | 2 |

**Quality is unchanged within noise, and it is slightly FASTER** — skipping a call
that was always going to fail pays for the per-child calls. So this is kept on
its merits (a silent failure mode replaced by a real evaluation, and one less
wasted engine timeout), not because it measurably strengthened play at the depth
we ship. On the depth-1 arm, where refusals are ~10× more common, it costs real
time — which is one more reason the dial stays at 2..4.

Note the value asymmetry is preserved throughout: our own king hanging is a real,
self-inflicted −SEARCH_WIN, while capturing theirs stays capped at +LEAF_CLAMP
because it is phantom-prone.

### While in there: two measurement lessons

- **Never time a search while anything else heavy runs.** Five identical
  invocations launched concurrently returned five different cp figures (52.3,
  56.0, 67.6, 105.1, 84.4) at identical seeds and positions, and ~1.7× the
  per-move wall clock. Under load the engine calls time out into the same static
  fallback above, so contention does not merely slow a run — it changes what the
  AI plays. An earlier grid run reporting 43 s/move (against ~0.65 s measured
  alone) was this, compounded with a cold cache.
- The `ms/move` column in `move-quality.mjs --grid` is therefore **correct**; the
  environment it was first measured in was not.

## 2026-08-03 — the leaf evaluator was being fed ILLEGAL positions

Stockfish's answer to an illegal FEN is not an error, it is **silence**: zero
MultiPV lines. `scoreChildren` then hands every child of that node to the static
evaluator without saying so. Two independent causes, found by dumping the actual
fallback sites (`OBSCURO_DEBUG_FALLBACK=<n>` samples every nth):

**1. Castling rights contradicting the board — FIXED.** Belief worlds and in-tree
positions routinely carry rights the placement cannot support:

```
r5p1/pp1kpppp/1bq1P1r1/... b Qkq -    ← black king on d7, still claims k/q
```

`toFEN` now emits a right only when the king is on its home square and the
matching rook is on its corner. Deriving from the board makes every FEN legal by
construction and is exactly right for an imagined world: a king that has wandered
has no castling rights, whatever the bookkeeping says. Measured on the same
config: **3.36% → 2.85% static-eval fallbacks.**

**2. Impossible piece placements — FIXED 2026-08-03, TWO producers.**

```
3rkb1r/pppqpp2/6p1/8/1P1P3P/B4P2/PP1R1P2/3PKB2 b k -
                                       ^^^^ white PAWN on d1
```

A white pawn cannot legally stand on rank 1. Both sources were found by doing the
cheap thing: `impossiblePlacement()` (belief.js) checked at every world producer
under `OBSCURO_VALIDATE_WORLDS=1`, which names the culprit in its output.

- **belief.js's contradiction fallback.** When a piece's possible-square set is
  pruned empty, it reset to *every* hidden square — including a pawn's own first
  rank. `tryReacquire` trusts these sets, so the exact belief then built worlds
  from them. Now falls back to `possibleSquaresFor(type, colour)`, which excludes
  a pawn's first and promotion ranks. Still a valid SUPERSET: the truth can never
  be on those squares.
- **belief.js's particle sampler.** `possible` legitimately keeps a pawn's
  promotion-rank squares — the piece really could have gone there — but whatever
  stands there is a QUEEN, so placing a *pawn* on it is impossible. `truncated`
  already flagged the piece (which is what stops re-acquisition trusting it); the
  sampler had no equivalent guard and now filters those squares out.

Measured over 3 games, both seats: **zero impossible worlds**, `engine-said-nothing`
falls from hundreds of nodes to **0**, and static-eval fallbacks drop
**2.85% → 0.57%**. Residual is `lines-but-not-our-moves` (140 nodes) and
`fewer-lines-than-asked` (13) — a different, much smaller phenomenon.

### Also landed

- Leaf health is now reported **per arm** in `move-quality.mjs`. It had been
  accumulating from a per-game-seat reset, so the "20–21% fallbacks" figure quoted
  on 2026-08-02 was one game-seat's, not a run's. Run-wide it is ~3%, and the two
  arms are identical (2.85% vs 2.82%) — so reach weighting does NOT degrade the
  evaluator, which was the other live hypothesis.
- Fallbacks are attributed by cause: `engine-said-nothing`,
  `fewer-lines-than-asked`, `lines-but-not-our-moves`.
