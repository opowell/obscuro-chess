// ---------------------------------------------------------------------------
// Shared command-line handling for the demo, the tuning scripts and the
// `obscuro-chess` binary.
//
// Two jobs:
//
//   1. SETTINGS FLAGS. --preset / --settings / --set / --print-config are
//      understood identically everywhere, so any entry point can pin a
//      parameter, load a whole named configuration, or reshape the difficulty
//      dial without growing its own vocabulary:
//
//        obscuro-chess demo --difficulty 60 --set chess.LEAF_CLAMP=900
//        node scripts/move-quality.mjs --settings sweep.json --arm reach
//        node scripts/move-quality.mjs --preset paper-design --games 40
//
//   2. ONE arg() HELPER. Four scripts each carried a verbatim copy of
//        const arg = (n, d) => { const i = argv.indexOf('--' + n); ... }
//      which is now imported instead.
//
// WHY NOT node:util parseArgs. It has to be told which flags take a value. In
// strict mode it rejects each script's own flags; in non-strict mode it reads
// `--sessions logs/` as a boolean followed by a positional, silently dropping
// the path. This scanner only consumes the flags it knows and hands the rest
// back untouched, which is the whole requirement for a shared pre-pass.
// ---------------------------------------------------------------------------

import {
  loadSettings, setOverrides, setPath, formatConfig, deepMerge, readSettingsFile,
} from './config.js';
import { preset, formatPresets } from './presets.js';

/** The four-copies-in-four-scripts helper, once. */
export function makeArgReader(argv) {
  return (name, fallback) => {
    const i = argv.indexOf('--' + name);
    return i >= 0 && argv[i + 1] != null && !argv[i + 1].startsWith('--') ? argv[i + 1] : fallback;
  };
}

/** Numeric variant, for flags like --difficulty that are always numbers. */
export function makeNumberReader(argv) {
  const read = makeArgReader(argv);
  return (name, fallback) => {
    const raw = read(name, null);
    if (raw == null) return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n)) throw new Error(`--${name} expects a number, got ${JSON.stringify(raw)}`);
    return n;
  };
}

// `--set path=value`. The value is parsed as JSON when it is valid JSON and
// left as a plain string otherwise, so all of these do the obvious thing:
//   chess.LEAF_CLAMP=900                       → 900
//   search.SEARCH_DEFAULTS.identityDiagnostic=false → false
//   search.DIAL.power.worlds={"min":1,"max":96}     → an object
//   chess.SF_CACHE_DIR=/tmp/sf                 → "/tmp/sf"
export function parseSetValue(raw) {
  try { return JSON.parse(raw); } catch { return raw; }
}

export const SETTINGS_FLAGS = ['--preset', '--list-presets', '--settings', '--set',
  '--print-config', '--print-changed'];

/**
 * Consume the settings flags from argv and apply them.
 *
 * @param {string[]} [argv] defaults to process.argv.slice(2)
 * @returns {{rest: string[], printConfig: false|'all'|'changed'|'presets'}}
 *          `rest` is argv with the settings flags removed, ready for the
 *          caller's own parsing.
 */
export function applyCliSettings(argv = process.argv.slice(2)) {
  const rest = [];
  const overrides = {};
  let sawOverride = false;
  let settingsFile = null;
  let presetName = null;
  let printConfig = false;

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--preset' || a.startsWith('--preset=')) {
      presetName = a === '--preset' ? argv[++i] : a.slice('--preset='.length);
      if (presetName == null) throw new Error('--preset needs a name, e.g. --preset paper');
    } else if (a === '--list-presets') {
      // Rides the same "print what I am running" channel as --print-config
      // rather than exiting here: an entry point decides what to do with it, and
      // a parser that called process.exit could not be tested.
      printConfig = 'presets';
    } else if (a === '--settings') {
      settingsFile = argv[++i];
      if (settingsFile == null) throw new Error('--settings needs a path to a JSON file');
    } else if (a.startsWith('--settings=')) {
      settingsFile = a.slice('--settings='.length);
    } else if (a === '--set' || a.startsWith('--set=')) {
      const pair = a === '--set' ? argv[++i] : a.slice('--set='.length);
      if (pair == null) throw new Error('--set needs path=value, e.g. --set chess.LEAF_CLAMP=900');
      const eq = pair.indexOf('=');
      if (eq < 0) throw new Error(`--set needs path=value, got ${JSON.stringify(pair)}`);
      setPath(overrides, pair.slice(0, eq), parseSetValue(pair.slice(eq + 1)));
      sawOverride = true;
    } else if (a === '--print-config') {
      printConfig = 'all';
    } else if (a === '--print-changed') {
      printConfig = 'changed';
    } else {
      rest.push(a);
    }
  }

  // Order matters: the preset is the lowest of the three, then the file, then
  // --set — so `--preset paper --set chess.SEARCH_WIN=8000` is "the paper's setup
  // except for this one knob", which is the shape a sweep over a preset takes.
  // The preset and the file share ONE settings layer (loadSettings), so the
  // merge happens here rather than by stacking two layers.
  //
  // A bad key, a bad value or an unknown preset is a user mistake at the command
  // line, not a bug worth a stack trace — the message already says which key and
  // what was expected, and (for a near miss) what was probably meant.
  try {
    const base = presetName ? preset(presetName) : null;
    if (base && settingsFile) loadSettings(deepMerge(base, readSettingsFile(settingsFile)));
    else if (base) loadSettings(base);
    else if (settingsFile) loadSettings(settingsFile);
    if (sawOverride) setOverrides(overrides);
  } catch (err) {
    if (!String(err.message).startsWith('settings:')) throw err;
    console.error(err.message);
    process.exit(2);
  }

  return { rest, printConfig };
}

/**
 * Print the resolved configuration if --print-config/--print-changed was given,
 * or the shipped presets for --list-presets. Returns whether anything was
 * printed, which is how a caller knows to separate it from its own output.
 */
export async function maybePrintConfig(printConfig) {
  if (!printConfig) return false;
  if (printConfig === 'presets') { console.log(formatPresets()); return true; }
  console.log(await formatConfig({ changedOnly: printConfig === 'changed' }));
  return true;
}

export const SETTINGS_USAGE = `
Settings flags (understood by every command):
  --preset <name>       load a named configuration (--list-presets to see them)
                        e.g. --preset paper       the Zhang & Sandholm setup
  --list-presets        print the shipped presets, with their aliases
  --settings <file>     load a JSON settings file (see docs/SETTINGS.md)
  --set <path>=<value>  override one parameter; repeatable
                        e.g. --set chess.LEAF_CLAMP=900
                             --set search.DIAL.power.worlds=32          (fix it)
                             --set search.DIAL.power.worlds.max=96      (reshape the dial)
  --print-config        print every parameter, its value and where it came from
  --print-changed       print only the parameters that are not at their default
`.trim();
