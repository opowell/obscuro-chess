// Presets: a whole configuration under one name, in the same key space as a
// settings file — and the layering that lets a sweep put one knob on top of one.
//
// The Zhang & Sandholm preset is the reason this exists (src/presets.js), so the
// assertions here are mostly about it being a REAL configuration: every key
// settable, every value reaching the code that reads it, and the paper's design
// half being exactly the affordable subset of the whole thing.

import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { param, resetSettings, validate, settingsTree } from '../src/config.js';
import { PRESETS, preset, presetNames, loadPreset, formatPresets } from '../src/presets.js';
import { applyCliSettings } from '../src/cli.js';
import { FogChess } from '../src/FogChess.js';
import { ChessObscuroAgent, SEARCH_WIN, LEAF_CLAMP, MAX_SF_DEPTH } from '../src/ObscuroAgent.js';
import { getDefaultMovePrior } from '../src/exactBelief.js';
import { UNIFORM_PRIOR, UNIFORM_ONLY } from '../src/movePrior.js';
import { CHESS_AGENT_SCORING } from '../src/ChessAgent.js';
import { quit as stockfishQuit } from '../src/stockfish.js';

test.beforeEach(() => resetSettings());
test.after(async () => {
  resetSettings();
  await stockfishQuit();
});

const configFor = (config = {}) => new ChessObscuroAgent({ rng: () => 0.42 })._config(
  FogChess.getVisibleState(FogChess.createInitialState(
    [{ id: 'white', name: 'White' }, { id: 'black', name: 'Black' }],
    { fogOfWar: true, difficulty: 40, ...config },
  ), 'white'));

// ---------------------------------------------------------------------------
// A preset is data in the documented key space
// ---------------------------------------------------------------------------

test('every shipped preset validates against the settable key space', () => {
  // The point of a preset being data: it goes through the same validation as a
  // settings file, so a renamed parameter breaks the preset HERE rather than
  // silently configuring nothing during a measurement run.
  for (const [name, { about, settings }] of Object.entries(PRESETS)) {
    assert.equal(typeof about, 'string', `${name} needs a one-line description`);
    assert.doesNotThrow(() => validate(settings), `${name} should be a valid settings tree`);
  }
});

test('the design half is exactly the affordable subset of the full preset', () => {
  // Both come from one definition (`deepMerge(DESIGN, SCALE)`), so this pins the
  // relationship rather than two hand-maintained copies agreeing by luck.
  const full = preset('zhang-sandholm');
  const design = preset('zhang-sandholm-design');
  for (const [key, value] of Object.entries(design.chess)) {
    assert.deepEqual(full.chess[key], value, `chess.${key} should carry into the full preset`);
  }
  // …and the scale half is only in the full one.
  assert.equal(design.search.DIAL.power.worlds, undefined, 'the design half leaves the dial alone');
  assert.equal(full.search.DIAL.power.worlds, 100);
});

test('aliases resolve, and an unknown name names the known ones', () => {
  assert.equal(preset('paper'), preset('zhang-sandholm'));
  assert.equal(preset('paper-design'), preset('zhang-sandholm-design'));
  assert.deepEqual(presetNames(),
    ['zhang-sandholm', 'zhang-sandholm-design', 'paper', 'paper-design']);
  assert.throws(() => preset('zhang'), /unknown preset "zhang" — known presets: zhang-sandholm/);
  // --list-presets output lists each configuration once, aliases beside it.
  assert.match(formatPresets(), /zhang-sandholm \(paper\)/);
});

// ---------------------------------------------------------------------------
// The Zhang & Sandholm setup, applied
// ---------------------------------------------------------------------------

test('the paper preset reaches the parameters it claims to set', () => {
  loadPreset('paper');

  // Bounded utilities: a win is worth the eval clamp (u: Z → [−1,+1]).
  assert.equal(param('chess.SEARCH_WIN', SEARCH_WIN), 1500);
  assert.equal(param('chess.SEARCH_WIN', SEARCH_WIN), param('chess.LEAF_CLAMP', LEAF_CLAMP));
  // Depth-1 leaf evaluation (App. C.5), in both dial modes.
  assert.equal(param('chess.CHESS_DIAL', {}).leafEval.sfDepth, 1);
  assert.equal(param('chess.MAX_SF_DEPTH', MAX_SF_DEPTH), 1);
  // |P| ≤ 10⁶, with a guard that won't trip before the cap does.
  assert.equal(param('chess.EXACT_BELIEF_CAP', 0), 1000000);

  // Hundreds of worlds and a ~10⁶-node tree, at every dial position: the paper's
  // engine has one strength, so these are fixed rather than ramped.
  for (const difficulty of [1, 40, 100]) {
    const cfg = configFor({ difficulty });
    assert.equal(cfg.worlds, 100, `worlds at difficulty ${difficulty}`);
    assert.equal(cfg.maxInfosets, 1000000);
    assert.equal(cfg.timeBudgetMs, 5000);
    assert.equal(cfg.purifyMax, 3);
  }
});

test('the paper preset serves no opponent model', () => {
  // The fitted move prior is this repo's addition; the paper samples uniformly
  // from P and models nothing about how the opponent chooses.
  assert.notEqual(getDefaultMovePrior(), UNIFORM_PRIOR, 'the default is the fitted model');
  loadPreset('paper-design');
  assert.equal(getDefaultMovePrior(), UNIFORM_PRIOR);
  resetSettings();
  assert.notEqual(getDefaultMovePrior(), UNIFORM_PRIOR, 'and it comes back');
});

test('the design preset changes no search budget, so it is measurable', () => {
  // The whole reason the two halves are separate: an A/B of the paper's design
  // points has to hold the search size fixed, or the result is a cost comparison.
  const before = configFor({ difficulty: 40 });
  loadPreset('paper-design');
  assert.deepEqual(configFor({ difficulty: 40 }), before);
});

// ---------------------------------------------------------------------------
// Layering
// ---------------------------------------------------------------------------

test('--preset is the bottom layer: a file and --set both beat it', () => {
  const dir = mkdtempSync(join(tmpdir(), 'obscuro-preset-'));
  try {
    const file = join(dir, 'sweep.json');
    writeFileSync(file, JSON.stringify({ chess: { SEARCH_WIN: 4000 } }));

    applyCliSettings(['--preset', 'paper', '--settings', file,
      '--set', 'chess.EXACT_BELIEF_CAP=500000']);

    assert.equal(param('chess.SEARCH_WIN', SEARCH_WIN), 4000, 'the file beats the preset');
    assert.equal(param('chess.EXACT_BELIEF_CAP', 0), 500000, '--set beats both');
    // Everything the two upper layers did not mention still comes from the preset.
    assert.equal(param('chess.MAX_SF_DEPTH', MAX_SF_DEPTH), 1);
    assert.equal(configFor().worlds, 100);
  } finally {
    resetSettings();
    rmSync(dir, { recursive: true, force: true });
  }
});

test('--list-presets rides the same print channel as --print-config', () => {
  // Not a process.exit inside the parser: an entry point decides what to do with
  // it, exactly as it does for --print-config, and a test can call it.
  assert.equal(applyCliSettings(['--list-presets']).printConfig, 'presets');
  assert.deepEqual(applyCliSettings(['--list-presets', 'keep-me']).rest, ['keep-me']);
});

test('a preset lands in the settings tree, not in a code path of its own', () => {
  // A preset that configured the engine some other way would be invisible to
  // --print-changed and to lastAnalysis.overrides, which is what makes a result
  // traceable back to the configuration that produced it.
  applyCliSettings(['--preset', 'paper-design']);
  assert.equal(settingsTree().chess.MOVE_PRIOR_UNIFORM, true);
});

// ---------------------------------------------------------------------------
// The parameters this preset needed, which used to be unreachable
// ---------------------------------------------------------------------------

test('MOVE_PRIOR_UNIFORM is off by default and reaches the compiled prior', () => {
  assert.equal(param('chess.MOVE_PRIOR_UNIFORM', UNIFORM_ONLY), false, 'off by default');
  applyCliSettings(['--preset', 'paper-design']);
  assert.equal(param('chess.MOVE_PRIOR_UNIFORM', UNIFORM_ONLY), true);
  assert.equal(getDefaultMovePrior(), UNIFORM_PRIOR);
});

test('the alpha-beta agent\'s scoring knobs are settable and deep-merged', () => {
  // pessimism / tailFraction / infoWeight / fogClamp were `const` scalars that no
  // aggregate listed — the same gap CHESS_AGENT_DIAL used to have.
  assert.deepEqual(param('chess.CHESS_AGENT_SCORING', CHESS_AGENT_SCORING), CHESS_AGENT_SCORING);
  applyCliSettings(['--set', 'chess.CHESS_AGENT_SCORING.pessimism=1']);
  const scoring = param('chess.CHESS_AGENT_SCORING', CHESS_AGENT_SCORING);
  assert.equal(scoring.pessimism, 1);
  assert.equal(scoring.fogClamp, CHESS_AGENT_SCORING.fogClamp, 'siblings survive');
});
