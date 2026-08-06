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
npm test                         # 128 tests
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

uniform-π                    logloss 5.891  baseline log|P| 6.200  Δ 0.309  medRank 151
FITTED (shipped, in-sample)  logloss 4.839  baseline log|P| 6.200  Δ 1.361  medRank  23
```

Δ is nats better than flat; medRank is where the true board sits in the belief's
own ordering. The shipped prior's weights were fitted by
`scripts/fit-move-prior.mjs`, not hand-tuned — see
[docs/MOVE-PRIOR-PLAN.md](docs/MOVE-PRIOR-PLAN.md) for what that found (the
king-safety weight comes out *negative*, and castling carries the model).

The other two scripts answer the question calibration cannot: whether a better
belief converts into better play. `scripts/move-quality.mjs` scores chosen moves
against a deep reference search; `scripts/strength-belief.mjs` plays seat-swapped
self-play pairs. Both default to the three recorded games in `test/fixtures/`;
point `--sessions <dir>` at a real corpus for anything conclusive.

## Layout

```
src/
  ObscuroAgent.js       the chess specialisation: Stockfish leaf evaluator,
                        perfect-information shortcut, analysis entry points
  ChessAgent.js         a plain alpha-beta + Stockfish agent (and the static
                        evaluator the search falls back to)
  exactBelief.js        the exact position set P and its posterior
  belief.js             heuristic particle belief, for when P is lost
  movePrior.js          π(move | position), the fitted opponent model
  beliefCalibration.js  the replay walk both the tests and the scripts use
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
test/                   128 tests, incl. three real recorded games as fixtures
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
