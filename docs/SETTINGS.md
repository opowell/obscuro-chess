# Settings

Two ways to change what the AI does. They are the same mechanism.

1. **Fix an individual parameter.** Give it a value. The difficulty dial stops
   moving it.
2. **Scale everything at once.** Leave the parameters alone and turn the
   difficulty dial — or reshape the dial itself, by overriding the `{min, max,
   curve}` endpoints it ramps over, or the exponent of its convex ramp
   (`search.DIAL_CONVEX_EXPONENT`).

A **[preset](#presets)** is a named bundle of the first: `--preset paper` loads
the whole Zhang & Sandholm configuration through this same layer.

```jsonc
// obscuro-chess.settings.json
{
  "search": {
    "DIAL": {
      "power": {
        "worlds": 32,                                  // 1. FIXED at 32
        "maxInfosets": { "min": 400, "max": 20000 }    // 2. same ramp, higher ceiling
      }
    }
  },
  "chess": { "LEAF_CLAMP": 900 }
}
```

A dial entry is either a range the dial ramps over or a bare number the dial
does not move — the convention `DIAL.power.purifyMax` already used. So "pin
this knob" is just "replace the range with a number", and nothing else needs to
know which of the two you did.

For what each parameter *means*, see [PARAMETERS.md](PARAMETERS.md) (fog chess)
and [../vendor/obscuro/docs/PARAMETERS.md](../vendor/obscuro/docs/PARAMETERS.md)
(the generic search). This document is only about how to change one.

---

## The keys are the aggregate export names

There is no second vocabulary. Every key under `chess.` is an export of
[`src/settings.js`](../src/settings.js); every key under `search.` is an export
of [`vendor/obscuro/src/settings.js`](../vendor/obscuro/src/settings.js). Those
are the same names as the tables in the two PARAMETERS.md files.

```
chess.LEAF_CLAMP                       search.DIAL.power.worlds
chess.CHESS_DIAL.leafEval.sfDepth      search.DIAL.time.maxInfosets
chess.EXACT_BELIEF_CAP                 search.DIAL_CONVEX_EXPONENT
chess.MOVE_PRIOR_FITTED_WEIGHTS        search.SEARCH_DEFAULTS.safePmaxThreshold
```

`obscuro-chess config` prints the full list with current values. A key that
isn't one of them is an error, not a silent no-op:

```
$ obscuro-chess config --set chess.LEAF_CLAM=900
settings: unknown key "chess.LEAF_CLAM" — did you mean "LEAF_CLAMP"?
```

Nested defaults are **deep-merged**, so naming one field leaves its siblings
alone — you can set `chess.CHESS_DIAL.leafEval.cols` without restating
`sfDepth`, or one weight of `chess.MOVE_PRIOR_FITTED_WEIGHTS` without restating
the fitted model.

---

## Presets

A preset is a whole configuration under one name, written in exactly the keys
above. They live in [`src/presets.js`](../src/presets.js) — a module rather than
a JSON file, so each value can carry the citation that justifies it.

```sh
obscuro-chess config --list-presets
obscuro-chess config --preset paper --print-changed     # what does it actually set?
obscuro-chess demo --preset paper-design --difficulty 60
node scripts/move-quality.mjs --preset paper-design --games 40
```

| Preset | What it is |
|---|---|
| `zhang-sandholm` (alias `paper`) | the paper's setup: its design points **and** its scale — hundreds of belief worlds, a ~10⁶-node tree cap, seconds per move |
| `zhang-sandholm-design` (alias `paper-design`) | the same design points at this engine's own search budget — the arm to measure, since it changes what the engine believes and how it prices leaves without changing how much search it buys |

### What the Zhang & Sandholm preset changes, and why

The vendored search implements the paper's algorithm — GT-CFR growth, PCFR+, the
KLUSS gadget, purification. What this repo changed is the **settings** around it,
each time a measurement disagreed with the paper's design point at ~100× smaller
trees. The preset puts those back:

| Key | Preset | Shipped | The paper claim it rests on |
|---|---|---|---|
| `chess.CHESS_DIAL.leafEval.sfDepth` | 1 | 2–4 | leaf evaluation runs at **depth 1** (App. C.5); strength comes from aggregating worlds and growing the tree |
| `chess.CHESS_DIAL.leafEval.cols` | 0 | 5–14 | price exactly the node's children (`cols` is a floor on MultiPV, not a cap) |
| `chess.MAX_SF_DEPTH` | 1 | 30 | the same design point for a per-move time limit, where the ladder replaces the dial |
| `chess.SEARCH_WIN` | 1500 | 8000 | utilities are bounded, `u: Z → [−1,+1]`, with evals clamped inside — so a certain win is worth the eval clamp, not 5.3× it |
| `chess.EXACT_BELIEF_CAP` | 10⁶ | 200,000 | `\|P\|` usually ≤ 10⁶ in the paper's C++ tracker (`TIME_GUARD_MS` rises with it, or the guard decides instead of the cap) |
| `chess.SAMPLE_ALPHA_DEFAULT` / `REACH_WEIGHTING_DEFAULT` | 0 / 0 | 0 / 0 | worlds are "sampled at random without replacement from the set of possible states", and every world in an information set is equally likely |
| `chess.MOVE_PRIOR_UNIFORM` | `true` | `false` | the paper has **no opponent model**; the fitted move prior is this repo's addition |
| `search.PURIFY_MAX_SUPPORT`, `DIAL.*.purifyMax` | 3 | 3 | MaxSupport = 3 (§3.5 / App. C.8) |
| `search.DIAL.power.worlds` / `.maxInfosets` / `.timeBudgetMs` / `.maxRounds` | 100 / 10⁶ / 5000 / unbounded | 1–48 / 400–6000 / 30–2000 / 6–100 | "hundreds of worlds", "~10⁶-node trees", "seconds/move", and a solver that runs until the clock rather than a round cap — **`zhang-sandholm` only** |

Three things worth knowing before you read a result:

- **The preset pins values that already agree with the default** (α, β,
  MaxSupport). It is a statement of a configuration, not a diff against this
  week's defaults.
- **It is a reference point, not a recommendation — and how much worse it plays is
  not measured.** Equalising `SEARCH_WIN` with the clamp gives up the asymmetric
  own-king-hang penalty, which exists because of an observed failure (the AI
  walking its king onto a square a hidden pawn covered), so that one stands on its
  own. The depth-1 argument used to cite `move-quality.mjs --grid`, but every
  move-quality number before 2026-08-07 is void — the harness advanced the belief
  with a move the agent never played, so both arms ran on the particle fallback
  ([PARAMETERS.md §2.4.1](PARAMETERS.md)). Running `--preset paper-design` against
  a default run on the repaired harness is the missing measurement.
- **Where the paper gives a mechanism but no number, the preset leaves the
  parameter alone** — `safePmaxThreshold`, `stableSnapshotEps`,
  `RESOLVE_PRIOR_UNIFORM_BLEND` and the other implementation guards. `presets.js`
  lists those explicitly, so "not set" reads as a decision rather than an
  oversight.

From code:

```js
import { PRESETS, preset, loadPreset, mergeSettings } from 'obscuro-chess';

loadPreset('paper');                                  // = loadSettings(preset('paper'))
loadSettings(mergeSettings(preset('paper'), mine));   // a preset with your own layer on top
```

---

## Where a value can come from

Lowest precedence first. Anything not set anywhere falls through to the
constant declared next to the code it tunes.

| Layer | How | Scope |
|---|---|---|
| built-in default | the constant in `src/*.js` | — |
| `./obscuro-chess.settings.json` | present in the working directory | process |
| `$OBSCURO_CHESS_SETTINGS` | a path to a JSON file | process |
| `--preset <name>` / `loadPreset(name)` | a named configuration | process |
| `--settings <file>` / `loadSettings(path \| object)` | explicit | process |
| `--set path=value` / `setOverrides(tree)` | explicit | process |
| `gameSpecific.obscuro` | `createInitialState(players, { obscuro: {…} })` | one game |
| constructor opts | `new ChessObscuroAgent({ particles: 32 })` | one agent |

The last two speak the agent's own opts vocabulary — `particles`,
`timeBudgetMs`, `maxRounds`, `maxInfosets`, `expandPerRound`, `cfrPerRound`,
`finalCfr`, `purifyMax`, `moveRings`, `moveSpokes`, `sfDepth` — rather than
settings paths, because they are per-instance rather than per-process.

**`gameSpecific.obscuro` is the one to reach for in a host.** The production
agent is a shared module-level singleton (`ObscuroAgent`), so constructor opts
are not available to a host that registers it; the session bag rides the game
state the way `difficulty` does, and a host running several games at once can
give each its own knobs.

Resolution happens at **read time**, per move — which is what lets a host
configure the singleton after importing it.

A preset and `--settings` share **one** layer (`loadSettings`): given both, the
CLI merges them, preset underneath. So `--preset paper --set chess.SEARCH_WIN=8000`
is "the paper's setup except for this one knob", which is the shape a sweep over
a preset takes.

---

## Command line

Every command understands the same six flags.

```
--preset <name>       load a named configuration (see Presets)
--list-presets        print the shipped presets, with their aliases
--settings <file>     load a JSON settings file
--set <path>=<value>  override one parameter; repeatable
--print-config        every parameter, its value, and where it came from
--print-changed       only what is not at its default
```

```sh
obscuro-chess config                                  # what am I running?
obscuro-chess config --settings sweep.json --print-changed
obscuro-chess config --preset paper --print-changed

obscuro-chess demo --difficulty 60 --set search.DIAL.power.worlds=8
obscuro-chess move-quality --arm reach --set chess.EXACT_BELIEF_CAP=500000
```

`--set` values are parsed as JSON when they are valid JSON and left as strings
otherwise, so all of these do the obvious thing:

```sh
--set chess.LEAF_CLAMP=900                          # 900
--set search.SEARCH_DEFAULTS.identityDiagnostic=false   # false
--set search.DIAL.power.worlds='{"min":1,"max":96}'     # a range
--set chess.SF_CACHE_DIR=/tmp/sf                    # "/tmp/sf"
```

Commands: `demo`, `move-quality`, `strength`, `calibrate`, `fit-prior`,
`config`. Each also takes its own flags.

---

## From code

```js
import { loadSettings, setOverrides, param, resolvedConfig } from 'obscuro-chess';

loadSettings({ chess: { LEAF_CLAMP: 900 } });   // or a path to a JSON file
setOverrides({ search: { DIAL: { power: { worlds: 32 } } } });

await resolvedConfig();   // [{ path, value, source }, …] for every parameter
```

`resetSettings()` drops every layer, here and in the generic search — tests
should call it in a `beforeEach`.

---

## Running a sweep

The reason most of this exists. `move-quality.mjs` and `strength-belief.mjs`
are the measurement harnesses (see
[PARAMETERS.md §3](PARAMETERS.md#3-adding-or-changing-a-parameter) — a change
that affects play strength has to be measured, and several plausible-looking
values in this repo measured *worse*).

```sh
for cap in 100000 200000 500000; do
  obscuro-chess move-quality --set chess.EXACT_BELIEF_CAP=$cap --games 40
done

# the paper's design points against the shipped defaults, same search budget
obscuro-chess move-quality --games 40
obscuro-chess move-quality --preset paper-design --games 40
```

Two things worth knowing before you trust the numbers:

- **Use power mode, not a time limit.** `move-quality.mjs` sets
  `timeBudgetMs: 0` deliberately: a wall-clock budget makes paired runs
  incomparable, because Stockfish cache warmth changes how much search a given
  number of milliseconds buys. Its header explains this at length.
- **Check what you actually ran.** `--print-changed` prints the effective
  overrides and their provenance, and every move's `lastAnalysis.overrides`
  carries the same information, so a result can be traced back to the
  configuration that produced it rather than to an assumption about it.

---

## How the two halves fit together

`chess.*` is resolved here. `search.*` is **forwarded verbatim** to
[obscuro-ai's own settings layer](../vendor/obscuro/docs/SETTINGS.md), which
owns those parameters and reads them through the same mechanism.

That is why `search.<NAME>` lines up 1:1 with a key upstream, why every generic
knob is settable — including `MIN_SUPPORT_PROB`, `PURIFY_MAX_SUPPORT`,
`RESOLVE_PRIOR_UNIFORM_BLEND` and `MIN_EXPANDED_ROOT_WORLDS`, which used to be
`const` scalars nothing could reach — and why there is only one copy of the
precedence rules in the stack instead of two that can drift.

A bare `obscuro.settings.json` still works underneath, for a process that also
drives the generic search directly. A chess settings file sits **above** it: in
a chess app, the chess configuration is the authority.

Adding a generic knob upstream makes it settable here the moment the submodule
is bumped — `SETTING_PATHS.search` is taken from obscuro-ai rather than restated,
and a test asserts it.
