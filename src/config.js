// ---------------------------------------------------------------------------
// Settings resolution — the ONE place a parameter's effective value is decided.
//
// Two ways to change what the AI does, and they are the same mechanism:
//
//   1. FIX a parameter.  Give it a value; the difficulty dial stops moving it.
//   2. SCALE everything. Leave parameters alone and turn the difficulty dial
//      (gameSpecific.difficulty 0-100, or aiTimeMs) — or reshape the dial
//      itself by overriding its {min, max, curve} endpoints, or the exponent
//      of its convex ramp.
//
// Both are expressed by writing to the SAME key. A dial entry is either a
// {min, max, curve} range the dial ramps over, or a bare number the dial does
// not move — that convention is already how DIAL.power.purifyMax works
// upstream, so "pin this knob" is just "replace the range with a number".
//
//   { "search": { "DIAL": { "power": { "worlds": 32 } } } }        // fixed
//   { "search": { "DIAL": { "power": { "worlds": { "max": 96 } } } } }  // reshaped
//
// KEY NAMES ARE THE AGGREGATE EXPORT NAMES. There is no second vocabulary to
// learn: every key under `chess.` is an export of src/settings.js, every key
// under `search.` is an export of vendor/obscuro/src/settings.js, and both are
// the tables in docs/PARAMETERS.md. See docs/SETTINGS.md.
//
// WHERE VALUES COME FROM, lowest precedence first:
//
//   built-in default          the constant declared next to the code it tunes
//   ./obscuro-chess.settings.json      picked up from the working directory
//   $OBSCURO_CHESS_SETTINGS   a path to a settings file
//   loadSettings(x)           an explicit path or object (also: --settings)
//   setOverrides(x)           a partial tree (also: --set path=value)
//   gameSpecific.obscuro      per-session, rides the game state (see FogChess)
//   new ChessObscuroAgent({}) per-instance constructor opts — always win
//
// Resolution is LAZY and happens at READ time, not at import or construction
// time. That is deliberate: the production entry point is the module-level
// `ObscuroAgent` singleton in ObscuroAgent.js, constructed before any host has
// had a chance to configure anything. Reading through param() at use time is
// what lets settings reach it at all.
//
// THE `search.` HALF IS NOT RESOLVED HERE. It is forwarded verbatim to
// obscuro-ai's own settings layer, which owns those parameters and reads them
// through the same mechanism. That is why `search.` keys line up 1:1 with
// upstream's key space, why the generic search's own `obscuro.settings.json`
// still works underneath, and why there is only one copy of the precedence
// rules in the stack rather than two that can drift.
//
// This module imports no other chess module, so nothing here can create an
// import cycle with the files whose constants it resolves.
// ---------------------------------------------------------------------------

import {
  createSettingsStore, LEAF, flatten, deepMerge, readJsonFile,
  setPath as setTreePath, ramp as rampSpec,
  SETTING_PATHS as SEARCH_SETTING_PATHS,
  setOverrides as setSearchOverrides,
  resetSettings as resetSearchSettings,
} from '../vendor/obscuro/src/config.js';

export { setTreePath as setPath };
// Re-exported (rather than imported from vendor/obscuro by every caller) so a
// host merging two settings trees — a preset under a file, say — uses the SAME
// merge the settings layer itself applies between layers, and reads a settings
// file with the same errors. See src/presets.js and src/cli.js.
export { deepMerge, readJsonFile as readSettingsFile };

export const SETTINGS_FILENAME = 'obscuro-chess.settings.json';
export const SETTINGS_ENV_VAR = 'OBSCURO_CHESS_SETTINGS';

// ---------------------------------------------------------------------------
// The valid key space.
//
// A shape, not a value table — the values still live at their declaration sites
// (docs/PARAMETERS.md §3), and this file must never become a second copy of
// them. Its job is to reject typos: an unrecognised agent opt used to be
// silently ignored, which is the worst possible behaviour for something you are
// running a parameter sweep with.
//
// Leaf markers: 'n' number, 'b' boolean, 's' string, 'd' dial entry (a number,
// or {min, max, curve, floor}), '*' anything (validated by its consumer).
//
// test/settings.test.js asserts this tree stays in step with the two
// settings.js aggregates' export lists, so a new constant cannot be added
// upstream or here without also becoming settable.
// ---------------------------------------------------------------------------
const { number: n, boolean: b, string: s, dial: d, any } = LEAF;

export const SETTING_PATHS = {
  // vendor/obscuro/src/settings.js — the generic search. Taken from upstream
  // rather than restated, so a knob added there is settable here the moment
  // the submodule is bumped.
  search: SEARCH_SETTING_PATHS,

  // src/settings.js — fog chess.
  chess: {
    DEFAULT_DIFFICULTY: n,
    CHESS_DIAL: {
      leafEval: { sfDepth: d, cols: d },
      proportionalPick: { multipvCap: d, depth: d, betaAtHalf: n, betaMax: n },
      timeModeEvalsPerWorldReserve: n,
      timeModeMinPerCallMs: n,
      timeModeSkill: n,
    },
    ANALYSIS_DEFAULTS: {
      strategy: { particles: n, maxRounds: n, expandPerRound: n, cfrPerRound: n, purifyMax: n },
      batchMixing: { maxRounds: n, expandPerRound: n, cfrPerRound: n },
      maxTotalMs: n, batchSize: n, sweepBatches: n,
      likelyWorldsCap: n, scoredWorldsCap: n,
      captureMultipv: n, captureDepth: n,
    },
    LEAF_CLAMP: n,
    SEARCH_WIN: n,
    MAX_SF_DEPTH: n,
    REFUSED_CHILD_CAP: n,
    CHESS_AGENT_DIAL: {
      depth: d, noiseCp: n, noiseZeroAt: n, quiesceFrom: n,
      fog: { particles: d, topK: d, depthShallow: n, depthDeep: n, shallowBelow: n },
    },
    CHESS_AGENT_SCORING: { pessimism: n, tailFraction: n, infoWeight: n, fogClamp: n },
    // belief.js — the heuristic particle belief.
    MAX_POSSIBLE: n, THREAT_BIAS: n, MAX_LURKERS: n,
    RECAPTURE_TYPE_WEIGHT: { pawn: n, knight: n, bishop: n, rook: n, queen: n, king: n },
    MAX_ATTEMPTS_PER_PARTICLE: n, PHANTOM_CHECK_REJECT_WINDOW: n,
    // exactBelief.js — the exact position-set tracker P.
    EXACT_BELIEF_CAP: n, EXACT_BELIEF_TIME_GUARD_MS: n, REACQUIRE_BOUND: n,
    SAMPLE_ALPHA_DEFAULT: n, REACH_WEIGHTING_DEFAULT: n,
    // movePrior.js — π(move | position). Deep-merged onto the fitted model;
    // read movePrior.js's header before touching these.
    MOVE_PRIOR_FITTED_WEIGHTS: any,
    // …or no opponent model at all: uniform π, which is the paper's setting.
    MOVE_PRIOR_UNIFORM: b,
    // stockfish.js — the vendored engine backend.
    RECYCLE_AFTER: n, CACHE_MAX: n, STOP_POLL_MS: n, SF_CACHE_DIR: s,
    LEGACY_DIFFICULTY: { easy: n, medium: n, hard: n, expert: n },
    SF_DIFFICULTY_RAMP: { movetimeMs: d, skill: d },
  },
};

// A generic knob written without its namespace is a common enough mistake to
// name explicitly, since `chess.` and `search.` hold disjoint key sets.
function crossNamespaceHint(key, path) {
  const inSearch = key in SEARCH_SETTING_PATHS;
  const inChess = key in SETTING_PATHS.chess;
  if (path.startsWith('chess.') && inSearch) return ` — "${key}" is a generic-search knob; write it as "search.${key}".`;
  if (path.startsWith('search.') && inChess) return ` — "${key}" is a fog-chess knob; write it as "chess.${key}".`;
  return null;
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

const store = createSettingsStore({
  schema: SETTING_PATHS,
  filename: SETTINGS_FILENAME,
  envVar: SETTINGS_ENV_VAR,
  extraHint: crossNamespaceHint,
  // Hand the generic half to the package that owns it. It sits in obscuro-ai's
  // top layer, so a chess settings file outranks a bare obscuro.settings.json
  // underneath it — which is the right way round: in a chess app, the chess
  // configuration is the authority.
  onResolve: tree => setSearchOverrides(tree.search ?? null, 'obscuro-chess'),
});

export const loadSettings = store.loadSettings;
export const setOverrides = store.setOverrides;
export const rediscoverSettings = store.rediscover;
export const param = store.param;
export const isOverridden = store.isOverridden;
export const settingsTree = store.tree;
export const settingsProvenance = store.provenance;
export const settingsEpoch = store.epoch;
export const validate = store.validate;

/**
 * Drop every layer, here and in the generic search — including restoring
 * anything obscuro-ai picked up from its own environment. Tests should call
 * this in a `beforeEach`.
 */
export function resetSettings() {
  resetSearchSettings();
  store.reset();
}

// ---------------------------------------------------------------------------
// The dial
// ---------------------------------------------------------------------------

/**
 * Evaluate one dial entry at position t ∈ [0,1].
 *
 * Upstream's implementation, used for chess's own dials too (CHESS_DIAL,
 * SF_DIFFICULTY_RAMP, CHESS_AGENT_DIAL). Those each used to carry their own
 * inline lerp, so an endpoint or curve change reached some of them and not
 * others; there is one ramp in the stack now, and `search.DIAL_CONVEX_EXPONENT`
 * reshapes all of them together.
 *
 * A bare number is a constant the dial does not move — which is exactly what a
 * settings file writes to fix a parameter.
 */
export const ramp = rampSpec;

/** param() + ramp(): the effective value of a dial-scaled knob at position t. */
export function dialParam(path, declaredDefault, t) {
  return ramp(param(path, declaredDefault), t);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/**
 * The full effective configuration, leaf by leaf, with where each value came
 * from — what `obscuro-chess config` prints.
 *
 * Async and dynamically imported on purpose: the default values live in the two
 * settings.js aggregates, and importing src/settings.js eagerly from here would
 * close the cycle config.js → settings.js → ObscuroAgent.js → config.js. By the
 * time anyone asks for a report, every module is loaded anyway.
 *
 * @returns {Promise<Array<{path: string, value: any, source: string}>>}
 */
export async function resolvedConfig() {
  const [chess, search] = await Promise.all([
    import('./settings.js'),
    import('../vendor/obscuro/src/settings.js'),
  ]);
  const defaults = { chess: {}, search: {} };
  for (const key of Object.keys(SETTING_PATHS.chess)) defaults.chess[key] = chess[key];
  for (const key of Object.keys(SETTING_PATHS.search)) defaults.search[key] = search[key];

  const provenance = settingsProvenance();
  return [...flatten(deepMerge(defaults, settingsTree()))]
    .map(([path, value]) => ({ path, value, source: provenance.get(path) ?? 'default' }))
    .sort((a, b) => a.path.localeCompare(b.path));
}

/** resolvedConfig() rendered as aligned text. */
export async function formatConfig({ changedOnly = false } = {}) {
  const rows = (await resolvedConfig()).filter(r => !changedOnly || r.source !== 'default');
  if (!rows.length) return changedOnly ? '(every parameter is at its default)' : '';
  const width = Math.max(...rows.map(r => r.path.length));
  return rows
    .map(r => `${r.path.padEnd(width)}  ${JSON.stringify(r.value)}  [${r.source}]`)
    .join('\n');
}
