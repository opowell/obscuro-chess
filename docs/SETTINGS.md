# Settings

Two ways to change what the AI does. They are the same mechanism.

1. **Fix an individual parameter.** Give it a value. The difficulty dial stops
   moving it.
2. **Scale everything at once.** Leave the parameters alone and turn the
   difficulty dial — or reshape the dial itself, by overriding the `{min, max,
   curve}` endpoints it ramps over, or the exponent of its convex ramp
   (`search.DIAL_CONVEX_EXPONENT`).

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

## Where a value can come from

Lowest precedence first. Anything not set anywhere falls through to the
constant declared next to the code it tunes.

| Layer | How | Scope |
|---|---|---|
| built-in default | the constant in `src/*.js` | — |
| `./obscuro-chess.settings.json` | present in the working directory | process |
| `$OBSCURO_CHESS_SETTINGS` | a path to a JSON file | process |
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

---

## Command line

Every command understands the same four flags.

```
--settings <file>     load a JSON settings file
--set <path>=<value>  override one parameter; repeatable
--print-config        every parameter, its value, and where it came from
--print-changed       only what is not at its default
```

```sh
obscuro-chess config                                  # what am I running?
obscuro-chess config --settings sweep.json --print-changed

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
