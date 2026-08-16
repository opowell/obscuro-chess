# Obscuro Chess

A **fog-of-war chess AI**. Each side sees only the squares its own pieces can
capture on; there is no check or checkmate, and you win by capturing the enemy king.
The AI does not guess where the enemy is and then best-respond to its guess — it
holds every position still consistent with everything it has seen, weights them
by how likely they are, and plays an equilibrium mixed strategy over that belief.

It is the chess half of [Obscuro](https://github.com/opowell/obscuro-ai), which
implements Zhang & Sandholm's game-independent search and is vendored here as a
submodule. This repository adds everything that knows about chess:

- the **rules and observation model** (`src/board.js`, `src/moves.js`, `src/FogChess.js`)
- a **Stockfish leaf evaluator** — one batched MultiPV call scores every child of
  an expanded node, plus the two fog terminals an ordinary engine cannot see
  (capturing the enemy king; leaving your own capturable)
- the **exact belief** P — every position consistent with the full observation
  history, maintained ply by ply, with |P| = 1 meaning the board is known
- a **move prior** π(move | position), fitted by MLE on recorded games, that
  turns P from a set into a distribution

**Zero dependencies. No build step. No install step.** Plain ES modules on
Node 18+, with Stockfish 18 (lite, single-threaded WASM) vendored in.

```bash
git clone --recurse-submodules https://github.com/opowell/obscuro-chess.git
cd obscuro-chess
node bin/obscuro-chess.js demo   # two agents play fog chess, each seeing only its own view
npm test                         # 158 tests
```

## Use it

```js
import { ChessObscuroAgent, FogChess } from 'obscuro-chess';

const agent = new ChessObscuroAgent();
const observation = FogChess.getVisibleState(state, 'white');   // fog applied here
const legal = FogChess.getLegalActions(observation, 'white');
const action = await agent.chooseAction(observation, legal);
```

`FogChess` is a complete GameDefinition in the sense obscuro-ai means it
([the contract](vendor/obscuro/docs/GAME-INTERFACE.md)). If your engine has its
own definition — a renderer, replay, per-square fog markers — write it against
the same `src/` modules and hand it over once at startup:

```js
import { setGame } from 'obscuro-chess';
setGame(MyChessGame);   // the search then applies exactly the rules your engine will
```

The agent scales through one **difficulty dial** (`gameSpecific.difficulty`,
0–100) or a **per-move time limit** (`gameSpecific.aiTimeMs`). There is one
algorithm at every setting; the dial only slides continuous knobs. Every fog-chess
number is documented in [docs/PARAMETERS.md](docs/PARAMETERS.md), and the generic
search's own knobs upstream in
[vendor/obscuro/docs/PARAMETERS.md](vendor/obscuro/docs/PARAMETERS.md).

## Settings

Any of those numbers can be **fixed**, so the dial stops moving it — or the
**dial itself reshaped**, so everything still scales together but over a
different range. Both are the same thing: a dial entry is either a `{min, max,
curve}` range or a bare number the dial leaves alone.

```sh
obscuro-chess config                                    # what am I running?
obscuro-chess demo --difficulty 60 --set search.DIAL.power.worlds=32
obscuro-chess move-quality --set search.DIAL.power.worlds.max=96
```

```jsonc
// obscuro-chess.settings.json, picked up from the working directory
{
  "search": { "DIAL": { "power": { "worlds": 32 } } },   // fixed
  "chess":  { "EXACT_BELIEF_CAP": 500000 }
}
```

The keys are the export names of the two `settings.js` aggregates — the same
names as the PARAMETERS.md tables — and a key that isn't one of them is an
error rather than a silent no-op. A settings file, `$OBSCURO_CHESS_SETTINGS`,
`--set`, a per-game `gameSpecific.obscuro` bag and an agent's constructor opts
stack in that order. The `search.*` half is forwarded to
[obscuro-ai's own settings layer](vendor/obscuro/docs/SETTINGS.md), which owns
those parameters. Full reference: [docs/SETTINGS.md](docs/SETTINGS.md).

A whole configuration can also come under one name:

```sh
obscuro-chess config --preset paper --print-changed   # the Zhang & Sandholm setup
```

`--preset paper` puts back every parameter this engine measured *away* from the
paper's design point at JavaScript tree sizes — depth-1 leaf evaluation, bounded
utilities, `|P| ≤ 10⁶`, no opponent model, hundreds of belief worlds — each with
the paper claim it rests on cited at its value in
[`src/presets.js`](src/presets.js). It is the arm to measure against, not a
recommendation: several of those choices are measurably worse here, which is why
the defaults are what they are.

## The belief is the interesting part

Under fog the search is only as good as the belief feeding it, so the belief is
measured rather than asserted. `scripts/calibrate-belief.mjs` replays recorded
games and reports how much probability the belief put on **what actually
happened** — against the flat-posterior baseline log|P|, which is what a prior
that models nothing scores by definition:

```
$ node scripts/calibrate-belief.mjs

uniform-π          logloss 5.891  baseline log|P| 6.200  Δ 0.309  medRank 151
FITTED (shipped)   logloss 4.882  baseline log|P| 6.200  Δ 1.318  medRank  29
```

Δ is nats better than flat; medRank is where the true board sits in the belief's
own ordering. The shipped prior's weights were fitted by
`scripts/fit-move-prior.mjs`, not hand-tuned — on 246 Chess.com Fog of War games
by 192 players, which makes the run above (over three *other* games) an
out-of-sample one. See [docs/MOVE-PRIOR-PLAN.md](docs/MOVE-PRIOR-PLAN.md) for
what the fit found: castling carries the model, and the one qualitative claim the
earlier 37-game fit made — that fog players walk their kings out — turned out to
be a single player's habit and vanished on the larger corpus.

The other two scripts answer the question calibration cannot: whether a better
belief converts into better play. `scripts/move-quality.mjs` scores chosen moves
against a deep reference search; `scripts/strength-belief.mjs` plays seat-swapped
self-play pairs. `move-quality` defaults to the three recorded games in
`test/fixtures/`; point `--sessions` at a real corpus for anything conclusive.

## Fitting it on your own corpus

**The nine numbers this package ships are the pre-built belief** — π is the only
part of the belief that *can* be shipped, since P itself is per-game state. They
were fitted on 246 Chess.com Fog of War games (14,836 decisions, 192 players).
Refit them on your own games:

```bash
obscuro-chess fit-prior --sessions games.zip          # dir, .zip, .pgn or .json
obscuro-chess fit-prior --sessions games.zip --e2e    # …gated on belief log-loss
obscuro-chess fit-prior --sessions games.zip --write  # replace FITTED_WEIGHTS
```

`--sessions` takes a directory (walked recursively), a `.zip`, a PGN, a session
JSON or a crawl JSON holding many games, any of them `.gz`. Games are filtered to
fog chess and everything rejected is counted and explained — a corpus that
half-loads says so rather than quietly producing a confident number over a third
of the data. The same flag works for `calibrate` and `move-quality`.

The `shipped` column in the CV table is the weights currently in the package,
scored on the same held-out folds. It is the arm that decides whether a refit is
worth taking, since `fitted` has been trained on the corpus and `shipped` has
not.

**Ratings.** PGN's `WhiteElo`/`BlackElo` (and a crawl's per-player Elo) carry how
strong each player was, and `--rating` lets the opponent's rating tilt every
weight *continuously* — filed by the rating of the seat that **made each
decision**, since the two players are usually not the same strength:

```
weight_k(r) = FITTED_WEIGHTS_k + RATING_SLOPE_k · z(r),   z = (r − 2000) / 400
```

Continuous rather than bucketed on purpose: every rated decision informs every
slope, instead of each bucket's nine weights being fitted on a third of the data,
and there are no edges to choose and no jump between a 1899- and a 1901-rated
opponent.

```
=== opponent rating as a continuous term — held-out, sloped vs flat ===
  14614 of 14836 decisions carry a rating; ratings 1426–2466 (median 2001)
  n=14614  uniform 3.405  flat 2.897  sloped 2.896  Δ +0.0003  no better than flat
```

**That is the real result on the corpus π is fitted on, and it is a null.** The
fitted slopes are not small — the rook PST term moves −2.7 per 400 Elo, 64% of
its own base — but they buy **0.0003 nats** out of sample, which means they are
fitting noise. `RATING_SLOPE` ships as zeros because of this run, so serving
reduces exactly to the flat model, and the machinery is here so the next corpus
can overturn it.

Per-band weights are a strictly larger model, so they fit training data better by
construction and that fact is worth nothing. The gate is the only number that
counts: on **held-out games of that band**, does the band's own π beat the pooled
π? `--write` emits `RATING_WEIGHTS` for the bands that pass and leaves out the
ones that don't, and serving falls back to the pooled model for any band the
table lacks. `RATING_WEIGHTS` ships **empty** — no corpus has yet earned it.

At runtime the opponent's rating tilts the model, since π models the opponent:

```js
FogChess.createInitialState([{ id: 'white' }, { id: 'black', rating: 1500 }], config);
// …or state.gameSpecific.opponentRating = 1500
```

## Adopting a corpus, in one command

Everything above — ingest health, the fit, the gate, the rating test — in the
order that stops you shipping on a number you did not check:

```bash
obscuro-chess adopt-corpus games.zip           # measure, change nothing
obscuro-chess adopt-corpus games.zip --write   # …and ship it if it wins
```

It refuses to go past ingest if the corpus does not mostly parse (the failure
mode that produced a confident number over half a corpus once already), refuses
to `--write` unless the refit beats the shipped weights on the **belief** gate,
and prints the caveat below every time.

## What a better prior does and does not buy

**A better π is a better belief, and — at the shipped defaults — not yet better
play.** π reaches move selection through exactly one channel: which worlds the
search draws. Both switches that would let it are zero:

| | ships at | meaning |
|---|---|---|
| `SAMPLE_ALPHA_DEFAULT` | **1** | worlds are drawn **∝ the posterior** (since 2026-08-16 — see below) |
| `REACH_WEIGHTING_DEFAULT` | 0 | each drawn world gets flat 1/N reach, not its importance weight |

So refitting π now reaches the played move through α, as well as sharpening the
belief that `calibrate` measures and the analysis panel displays. β = 0 still
blocks the other channel.

**Turning α on is unresolved, not settled.** It used to be "measured twice and
lost both times", quoting **+2.96 ± 2.62 cp in favour of α=0** from
`move-quality --arm alpha`. That number is void: the harness committed the
agent's own unplayed pick on top of the recorded move, which killed the exact
belief on ply 2 and left both arms sampling the heuristic fallback — so α was
never actually varied. Every pre-2026-08-07 `move-quality` number has the same
defect, including the leaf-depth grid. Remeasured on a live P, 2,044 paired
positions over 40 corpus games:

| arm | paired cp (A − B) | sign test | reading |
|---|---|---|---|
| α=1 vs α=0 | −0.86 ± 1.66 | 52.4% for α=1 (z = 1.53) | mild lean **toward** α=1 |
| reach β=1 vs β=0 | +2.25 ± 1.61 | 47.1% for β=1 (z = −1.83) | lean toward shipped β=0 |

Neither reaches 2σ, so both defaults stay where they are — but the evidence that
justified α=0 now points the other way, and the seat-swapped self-play result
(4–11 for α=0, unaffected by the bug since it plays what it picks) disagrees with
it. That is the honest state.

**2026-08-16: α is now resolved, and α=1 ships.** The table above, and the flat
`+0.21 ± 1.08` that followed it, were both taken at the default `CAP = 200,000` /
`TIME_GUARD_MS = 4,000` — which abandons exactness, stickily for the rest of the
game, on precisely the high-|P| turns where α does its work. Remeasured at
`CAP = 2e6` / 180 s on the crawl games π was *not* fitted to, 300 discovery + 300
**holdout** games with the |P| cut points registered before the holdout ran:

| subset | discovery | holdout |
|---|---|---|
| all exact positions | −1.81 ± 0.57 | **−2.78 ± 0.59** (z = −4.70) |
| \|P\| ≤ 20 — predicted ≈ 0 | −0.27 ± 0.75 | +0.93 ± 0.77 |
| \|P\| ≥ 10³ | −2.84 ± 0.94 | **−4.20 ± 0.90** (z = −4.65) |

Pooled: **−2.28 ± 0.41 cp**, and −3.55 ± 0.65 where |P| ≥ 10³. Replaying the
holdout under the old cap reproduces the earlier null (−2.78 → −1.64 ± 0.63,
1.5σ of power against SE 1.08), so the runs differ by z = 1.48 once censoring is
matched — an underpowered null, not a contradiction. Unlike the dilution story
below, this one is not a tail artefact: it survives dropping the top 25% of |P|
(−1.91 ± 0.63) and its sign test is stronger than its mean test.

**α=1 is a departure from the paper, not a reproduction of it.** Zhang &
Sandholm sample uniformly because they have no opponent model to weight with;
α exists here only because of the fitted π. `--preset paper` therefore pins
α = 0 explicitly, and that line is load-bearing.

The caveat that no sample size removes: cp-loss is scored against a depth-12
search of the *true* board, which is blind to information value. This says α=1
plays more accurately by that referee — not yet that it plays better fog chess.
A properly powered `strength-belief` run is the outstanding work.

**And it is not a dilution artefact.** The obvious explanation — a belief knob can
only pay where something is hidden, and the average is swamped by turns where
almost nothing is — was tested by recording |P| at every decision and regressing
the paired difference on it. There is no relationship for either knob: the fits
on rank(|P|) and log10|P| are null, and the one significant raw-|P| slope
(t = −3.47) is carried entirely by the largest 5% of |P| and flips sign when they
are dropped. How much is hidden does not predict where these knobs help.

Converting belief accuracy into playing strength is therefore an open problem in
this repo, and a bigger corpus is not on its own the answer to it.

**One caveat worth more than the rest.** Fog chess and ordinary chess are
different games, and π is fitted on *fog* behaviour: the king PST weight comes out
negative because under fog players walk kings toward the centre, and capture is
nearly worthless because the observation filter has already priced captures you
can see. Fitting on a full-information PGN corpus will get both of those
backwards — confidently. `--e2e` is the check that catches it; the `floor` term
bounds what it can cost if you ship anyway.

## Layout

```
src/
  ObscuroAgent.js       the chess specialisation: Stockfish leaf evaluator,
                        perfect-information shortcut, analysis entry points
  ChessAgent.js         a plain alpha-beta + Stockfish agent (and the static
                        evaluator the search falls back to)
  exactBelief.js        the exact position set P and its posterior
  belief.js             heuristic particle belief, for when P is lost
  movePrior.js          π(move | position), the fitted opponent model, and the
                        optional per-rating-band table over it
  beliefCalibration.js  the replay walk both the tests and the scripts use
  corpus.js pgn.js zip.js
                        reading recorded games back in: directories, zips, PGN
                        (with ratings) and session JSON, behind one loader
  stockfish.js          the vendored engine (Node worker / browser Worker)
  FogChess.js           the GameDefinition tying it together
  board.js moves.js fen.js pieceTables.js
                        rules, move generation, FEN
  playMatch.js          a minimal self-play loop for the scripts and the demo
  settings.js           every fog-chess default, in one place
  config.js             settings resolution: fix a parameter, or reshape the dial
  presets.js            named configurations, notably the Zhang & Sandholm setup
  cli.js                shared --preset / --settings / --set / --print-config handling
bin/obscuro-chess.js    demo, the tuning harnesses and `config`, in one command
vendor/obscuro/         the generic search (submodule: opowell/obscuro-ai)
vendor/stockfish/       Stockfish 18 lite, single-threaded WASM
test/                   158 tests, incl. three real recorded games as fixtures
docs/                   parameters, how to change one, and the design plans
scripts/                calibration, prior fitting, move quality, strength
```

## In the browser

`src/stockfish.js` runs unchanged in a browser: same UCI protocol, same
`multiPV`/`bestMove` API, a classic Worker instead of a Node worker thread. The
vendored `.cjs` and `.wasm` are plain static assets, so serving this directory
byte-for-byte is enough — no bundler, no build. The disk cache turns itself off
(in-memory only) when there is no filesystem.

In Node, that cache lives in `vendor/stockfish/` unless you say otherwise:

```js
import { setCacheDir } from 'obscuro-chess/stockfish';
setCacheDir('/path/to/my/cache');   // or set SF_CACHE_DIR
```

## Stability

`ChessObscuroAgent`, `ChessAgent`, `FogChess`, `setGame` and the belief classes
(`ExactBelief`, `Belief`, `makeMovePrior`) are the supported API, along with the
exported settings. Everything else `src/index.js` exports is internal, exported
for tests and advanced hosts, and may change.

## License

MIT.
