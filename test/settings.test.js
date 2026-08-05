// Settings: the two things a caller can do — fix an individual parameter, or
// reshape the difficulty dial that scales all of them — and the order in which
// the layers that can do either of those win.
//
// Assertions read off `agent.lastAnalysis`, the same surface the generic dial
// tests upstream use (vendor/obscuro/test/agent.test.js).

import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  loadSettings, setOverrides, resetSettings, rediscoverSettings,
  param, ramp, resolvedConfig, settingsProvenance,
  validate, SETTING_PATHS, setPath,
} from '../src/config.js';
import { FogChess } from '../src/FogChess.js';
import { ChessObscuroAgent, ObscuroAgent, CHESS_DIAL, LEAF_CLAMP } from '../src/ObscuroAgent.js';
import { DEFAULT_DIFFICULTY, difficultyToNumber, quit as stockfishQuit } from '../src/stockfish.js';
import { getDefaultMovePrior, setDefaultMovePrior } from '../src/exactBelief.js';
import * as chessSettings from '../src/settings.js';
import * as searchSettings from '../vendor/obscuro/src/settings.js';
import { DIAL } from '../vendor/obscuro/src/settings.js';
import { param as searchParam, settingsTree as searchTree } from '../vendor/obscuro/src/config.js';

// Every test starts from "no settings anywhere", including the DIAL object the
// generic search closed over at import time.
test.beforeEach(() => resetSettings());
test.after(async () => {
  resetSettings();
  await stockfishQuit();   // the engine worker keeps the event loop alive
});

const rng = () => 0.42;

// A fog position with a few pieces hidden, so the agent takes the belief search
// path rather than the perfect-information shortcut.
function fogState(config = {}) {
  return FogChess.createInitialState(
    [{ id: 'white', name: 'White' }, { id: 'black', name: 'Black' }],
    { fogOfWar: true, difficulty: 40, ...config },
  );
}

// The resolved per-move knobs, without running a search. This is the layer the
// settings system actually decides, and it is deterministic — `lastAnalysis
// .worlds` is not a substitute, because the number of belief worlds the search
// ends up with is min(knob, |P|), and a belief that has not been advanced
// through onActionCommitted still holds exactly one position.
function configFor(state, agentOpts = {}) {
  const agent = new ChessObscuroAgent({ rng, ...agentOpts });
  return agent._config(FogChess.getVisibleState(state, 'white'));
}

// End-to-end counterpart: run a real move and read what the search was given.
// timeBudgetMs is the knob to probe with — nothing downstream clamps it.
async function analysisFor(state, agentOpts = {}) {
  const agent = new ChessObscuroAgent({ rng, ...agentOpts });
  const observation = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(observation, 'white');
  await agent.chooseAction(observation, legal);
  return agent.lastAnalysis;
}

// ---------------------------------------------------------------------------
// The key space
// ---------------------------------------------------------------------------

test('the settable key space is exactly the two settings.js aggregates', () => {
  // Guards the promise src/settings.js's header makes: a settings key IS an
  // aggregate export name. A new constant that is exported but not settable
  // (or vice versa) fails here rather than silently doing nothing at runtime.
  const settable = Object.keys(SETTING_PATHS.chess).sort();
  const exported = Object.keys(chessSettings).sort();
  assert.deepEqual(settable, exported);

  // The generic half is taken verbatim from obscuro-ai's own key space (which
  // its suite pins to its settings.js exports), so every generic knob is
  // settable here the moment the submodule is bumped.
  assert.deepEqual(Object.keys(SETTING_PATHS.search).sort(), Object.keys(searchSettings).sort());
});

test('an unknown key is rejected, with the near miss named', () => {
  assert.throws(() => validate({ chess: { LEAF_CLAM: 900 } }),
    /unknown key "chess.LEAF_CLAM" — did you mean "LEAF_CLAMP"\?/);
  assert.throws(() => validate({ chess: { MIN_SUPPORT_PROB: 0.5 } }),
    /is a generic-search knob; write it as "search.MIN_SUPPORT_PROB"/);
  assert.throws(() => validate({ search: { LEAF_CLAMP: 900 } }),
    /is a fog-chess knob; write it as "chess.LEAF_CLAMP"/);
  assert.throws(() => validate({ chess: { LEAF_CLAMP: 'nope' } }),
    /chess.LEAF_CLAMP must be a number/);
  assert.throws(() => validate({ search: { DIAL: { power: { worlds: { mn: 1 } } } } }),
    /is not a dial field/);
  assert.throws(() => validate({ search: { DIAL: { power: { worlds: { curve: 'wiggly' } } } } }),
    /must be "convex" or "linear"/);
});

// ---------------------------------------------------------------------------
// Fixing a parameter
// ---------------------------------------------------------------------------

test('a fixed dial entry is the same at every difficulty', () => {
  setOverrides({ search: { DIAL: { power: { worlds: 32 } } } });
  for (const difficulty of [1, 10, 50, 90, 100]) {
    assert.equal(configFor(fogState({ difficulty })).worlds, 32,
      `worlds should be pinned at difficulty ${difficulty}`);
  }
  // Everything not pinned still scales with the dial.
  assert.ok(configFor(fogState({ difficulty: 90 })).timeBudgetMs
    > configFor(fogState({ difficulty: 10 })).timeBudgetMs);
});

test('a fixed parameter reaches the agent, and the dial still moves the rest', () => {
  const before = configFor(fogState({ difficulty: 40 }));
  setOverrides({ search: { DIAL: { power: { worlds: 3 } } } });
  const after = configFor(fogState({ difficulty: 40 }));

  assert.equal(after.worlds, 3);
  assert.notEqual(before.worlds, 3, 'the default at difficulty 40 should not already be 3');
  // timeBudgetMs was not pinned, so it is unchanged by pinning worlds.
  assert.equal(after.timeBudgetMs, before.timeBudgetMs);
});

test('a fixed parameter reaches a real search end to end', async () => {
  const clean = await analysisFor(fogState({ difficulty: 40 }));
  assert.notEqual(clean.timeBudgetMs, 137);

  setOverrides({ search: { DIAL: { power: { timeBudgetMs: 137 } } } });
  const pinned = await analysisFor(fogState({ difficulty: 40 }));
  assert.equal(pinned.timeBudgetMs, 137);
});

test('time mode scales off aiTimeMs, and the user limit stays the budget', () => {
  const cfg = configFor(fogState({ difficulty: null, aiTimeMs: 30000 }));
  assert.equal(cfg.timeMode, true);
  assert.equal(cfg.timeBudgetMs, 30000, 'the per-move limit IS the budget');

  // The knobs the budget cannot bound still ramp with it…
  const slower = configFor(fogState({ difficulty: null, aiTimeMs: 60000 }));
  assert.ok(slower.maxInfosets > cfg.maxInfosets);

  // …and pinning one works the same way as in power mode.
  setOverrides({ search: { DIAL: { time: { maxInfosets: 4242 } } } });
  assert.equal(configFor(fogState({ difficulty: null, aiTimeMs: 30000 })).maxInfosets, 4242);
});

test('a plain chess constant is settable and read at use time', () => {
  assert.equal(param('chess.LEAF_CLAMP', LEAF_CLAMP), 1500);
  setOverrides({ chess: { LEAF_CLAMP: 900 } });
  assert.equal(param('chess.LEAF_CLAMP', LEAF_CLAMP), 900);
  resetSettings();
  assert.equal(param('chess.LEAF_CLAMP', LEAF_CLAMP), 1500);
});

test('a nested default is deep-merged, not replaced', () => {
  setOverrides({ chess: { CHESS_DIAL: { leafEval: { cols: 20 } } } });
  const dial = param('chess.CHESS_DIAL', CHESS_DIAL);
  assert.equal(dial.leafEval.cols, 20);
  // Siblings survive: sfDepth's range and the whole proportionalPick block.
  assert.deepEqual(dial.leafEval.sfDepth, CHESS_DIAL.leafEval.sfDepth);
  assert.equal(dial.proportionalPick.betaMax, CHESS_DIAL.proportionalPick.betaMax);
});

// ---------------------------------------------------------------------------
// Reshaping the dial
// ---------------------------------------------------------------------------

test('raising a dial endpoint changes what the same difficulty resolves to', () => {
  const worldsAt = difficulty => configFor(fogState({ difficulty })).worlds;
  const base = worldsAt(50);
  setOverrides({ search: { DIAL: { power: { worlds: { min: 1, max: 96, curve: 'convex' } } } } });
  assert.ok(worldsAt(50) > base, `raising max should raise the mid-dial value (${base} → ${worldsAt(50)})`);
  // Endpoints still hold at the ends of the dial.
  assert.equal(worldsAt(100), 96);
  assert.equal(worldsAt(1), 1);
});

test('the convex exponent reshapes the curve without moving its endpoints', () => {
  // The exponent is a `const` scalar in vendor/obscuro. It is settable because
  // obscuro-ai reads it through its own param(), and this package forwards
  // `search.*` there rather than trying to reach into it.
  const worldsAt = difficulty => configFor(fogState({ difficulty })).worlds;
  const mid = () => worldsAt(50);
  const ends = () => [worldsAt(1), worldsAt(100)];

  const base = mid();
  const baseEnds = ends();

  // A smaller exponent is a flatter curve: the mid-dial buys MORE.
  setOverrides({ search: { DIAL_CONVEX_EXPONENT: 1.0 } });
  assert.ok(mid() > base, `exponent 1.0 should raise the midpoint (${base} → ${mid()})`);
  assert.deepEqual(ends(), baseEnds, 'the endpoints are fixed points of the curve');

  // A larger one is steeper: the mid-dial buys less.
  setOverrides({ search: { DIAL_CONVEX_EXPONENT: 3 } });
  assert.ok(mid() < base, `exponent 3 should lower the midpoint (${base} → ${mid()})`);
  assert.deepEqual(ends(), baseEnds);
});

test('a generic knob with no chess namespace is still settable through search.*', () => {
  // These four are `const` scalars upstream that no caller-supplied config was
  // threaded through, so before obscuro-ai grew its own settings layer nothing
  // in this package could reach them at all.
  for (const key of ['MIN_SUPPORT_PROB', 'PURIFY_MAX_SUPPORT',
    'RESOLVE_PRIOR_UNIFORM_BLEND', 'MIN_EXPANDED_ROOT_WORLDS']) {
    setOverrides({ search: { [key]: 0.25 } });
    assert.equal(searchParam(key, null), 0.25, key);
  }
});

test('ramp: a bare number is a constant, a range is scaled, and floor binds', () => {
  assert.equal(ramp(7, 0), 7);
  assert.equal(ramp(7, 1), 7);
  assert.equal(ramp({ min: 0, max: 10, curve: 'linear' }, 0.5), 5);
  assert.equal(ramp({ min: 0, max: 10, curve: 'convex' }, 0.5), 4); // 10 * 0.5^1.5
  assert.equal(ramp({ min: 0, max: 10, curve: 'linear', floor: 3 }, 0), 3);
});

test('the generic half is forwarded to obscuro-ai, not resolved here', () => {
  // obscuro-ai owns its own parameters and reads them through the same
  // mechanism; this package forwards the `search.` subtree rather than keeping
  // a second copy of the precedence rules. The shipped DIAL object is left
  // alone — an earlier version patched it in place, which worked but meant two
  // packages mutating one object.
  setOverrides({ search: { DIAL: { power: { worlds: { max: 96 } } } } });
  assert.equal(searchTree().DIAL.power.worlds.max, 96, 'upstream sees the override');
  assert.equal(DIAL.power.worlds.max, 48, 'the shipped default is untouched');
  assert.equal(searchParam('DIAL.power', DIAL.power).worlds.max, 96);

  resetSettings();
  assert.equal(searchTree().DIAL, undefined, 'reset clears it upstream too');
  assert.equal(searchParam('DIAL.power', DIAL.power).worlds.max, 48);
});

// ---------------------------------------------------------------------------
// Precedence
// ---------------------------------------------------------------------------

test('each layer beats the one below it', () => {
  const dir = mkdtempSync(join(tmpdir(), 'obscuro-settings-'));
  try {
    const file = join(dir, 'settings.json');
    writeFileSync(file, JSON.stringify({ search: { DIAL: { power: { worlds: 5 } } } }));

    // 1. the dial
    assert.ok(configFor(fogState()).worlds > 0);

    // 2. a settings file beats the dial
    loadSettings(file);
    assert.equal(configFor(fogState()).worlds, 5);

    // 3. --set beats the file
    setOverrides({ search: { DIAL: { power: { worlds: 6 } } } });
    assert.equal(configFor(fogState()).worlds, 6);

    // 4. a per-session bag beats --set
    const session = fogState({ obscuro: { particles: 7 } });
    assert.equal(configFor(session).worlds, 7);

    // 5. the agent's own opts beat everything
    assert.equal(configFor(session, { particles: 9 }).worlds, 9);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a settings object may be passed instead of a path', () => {
  loadSettings({ search: { DIAL: { power: { worlds: 4 } } } });
  assert.equal(configFor(fogState()).worlds, 4);
});

test('an auto-discovered file in the working directory is picked up', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'obscuro-cwd-'));
  const cwd = process.cwd();
  try {
    writeFileSync(join(dir, 'obscuro-chess.settings.json'),
      JSON.stringify({ chess: { LEAF_CLAMP: 777 } }));
    process.chdir(dir);
    rediscoverSettings();
    assert.equal(param('chess.LEAF_CLAMP', LEAF_CLAMP), 777);
    assert.equal(settingsProvenance().get('chess.LEAF_CLAMP')?.startsWith('file:'), true);
  } finally {
    process.chdir(cwd);
    resetSettings();
    rediscoverSettings();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Reaching production
// ---------------------------------------------------------------------------

test('the shared ObscuroAgent singleton sees settings applied after import', async () => {
  // The host registers this module-level instance, constructed at import —
  // before any host has had a chance to configure anything. If resolution were
  // baked in at construction, nothing a host sets could reach the agent it
  // actually runs, which is how per-parameter overrides were unreachable in
  // production before this existed.
  setOverrides({ search: { DIAL: { power: { timeBudgetMs: 111 } } } });
  const state = fogState();
  const observation = FogChess.getVisibleState(state, 'white');
  const legal = FogChess.getLegalActions(observation, 'white');
  await ObscuroAgent.chooseAction(observation, legal);
  assert.equal(ObscuroAgent.lastAnalysis.timeBudgetMs, 111);
});

test('lastAnalysis reports which parameters were not left at their defaults', async () => {
  const clean = await analysisFor(fogState());
  assert.equal(clean.overrides, undefined, 'a default configuration reports nothing');

  setOverrides({ chess: { LEAF_CLAMP: 900 } });
  const dirty = await analysisFor(fogState(), { particles: 3 });
  assert.equal(dirty.overrides['chess.LEAF_CLAMP'], 'cli');
  assert.equal(dirty.overrides['opts.particles'], 'agent');
});

// ---------------------------------------------------------------------------
// Reporting, and the unified default difficulty
// ---------------------------------------------------------------------------

test('resolvedConfig reports every parameter with its provenance', async () => {
  setOverrides({ chess: { LEAF_CLAMP: 900 } });
  const rows = await resolvedConfig();
  const byPath = new Map(rows.map(r => [r.path, r]));

  assert.equal(byPath.get('chess.LEAF_CLAMP').value, 900);
  assert.equal(byPath.get('chess.LEAF_CLAMP').source, 'cli');
  assert.equal(byPath.get('chess.SEARCH_WIN').value, 8000);
  assert.equal(byPath.get('chess.SEARCH_WIN').source, 'default');
  // Defaults come from the aggregates, so nested ones are reported leaf by leaf.
  assert.equal(byPath.get('search.DIAL.power.worlds.max').value, 48);
});

test('one default difficulty answers for every code path', () => {
  assert.equal(difficultyToNumber(undefined), DEFAULT_DIFFICULTY);
  assert.equal(fogState({ difficulty: undefined }).gameSpecific.difficulty, DEFAULT_DIFFICULTY);

  setOverrides({ chess: { DEFAULT_DIFFICULTY: 70 } });
  assert.equal(difficultyToNumber(undefined), 70);
  assert.equal(fogState({ difficulty: undefined }).gameSpecific.difficulty, 70);
});

test('a derived value is re-derived when settings change, not frozen', () => {
  // The move prior is COMPILED from its weight vector, so it is cached rather
  // than rebuilt per use. Caching it without watching the settings epoch would
  // freeze the first read for the process — which would quietly turn a sweep
  // over the weights into many runs of one arm.
  const before = getDefaultMovePrior();
  assert.equal(getDefaultMovePrior(), before, 'stable while nothing changes');

  setOverrides({ chess: { MOVE_PRIOR_FITTED_WEIGHTS: { captureWeight: 5 } } });
  assert.notEqual(getDefaultMovePrior(), before, 'rebuilt after a settings change');

  // An explicit setter is a deliberate choice and outlives later settings changes.
  const explicit = () => 0;
  setDefaultMovePrior(explicit);
  setOverrides({ chess: { MOVE_PRIOR_FITTED_WEIGHTS: { captureWeight: 9 } } });
  assert.equal(getDefaultMovePrior(), explicit);
  setDefaultMovePrior(null);
  assert.notEqual(getDefaultMovePrior(), explicit);
});

test('setPath writes a dotted --set path into a tree', () => {
  const tree = {};
  setPath(tree, 'search.DIAL.power.worlds.max', 96);
  setPath(tree, 'chess.LEAF_CLAMP', 900);
  assert.deepEqual(tree, {
    search: { DIAL: { power: { worlds: { max: 96 } } } },
    chess: { LEAF_CLAMP: 900 },
  });
});
