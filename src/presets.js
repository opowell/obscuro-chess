// ---------------------------------------------------------------------------
// Named settings presets — a whole configuration under one name, expressed in
// exactly the same key space as a settings file (`chess.*` / `search.*`, see
// docs/SETTINGS.md). A preset is DATA, not a code path: `--preset paper` loads
// the tree below through the ordinary settings layer, which validates it against
// SETTING_PATHS like any other source, so a preset cannot reach a parameter a
// settings file could not.
//
// WHY A JS MODULE AND NOT A .json FILE. Two reasons, both of them this repo's
// existing convention: settings files are strict JSON (no comments), and every
// number in this package lives next to the reason it is that number. A preset
// whose values could not carry their citation would be a table of magic numbers
// with the argument stripped out — which is exactly what docs/PARAMETERS.md
// exists to prevent. It also means a browser host can use one (no fs).
//
// ---------------------------------------------------------------------------
// THE ZHANG & SANDHOLM PRESET
// ---------------------------------------------------------------------------
//
// `zhang-sandholm` (alias `paper`) answers one question: what does this engine
// play like when it is configured the way the PAPER's Obscuro is, rather than
// the way this repo measured to be best at JavaScript scale?
//
// That question is worth a named preset because the two differ on purpose, and
// in both directions. The generic search (vendor/obscuro) implements the paper's
// algorithm faithfully — GT-CFR growth, PCFR+, the KLUSS gadget, purification.
// What this repo changed is the SETTINGS around it, every time a measurement
// disagreed with the paper's design point at ~100× smaller trees. This preset
// puts every one of those back.
//
// EACH ENTRY BELOW CITES THE PAPER CLAIM IT RESTS ON, and every citation is a
// comment in this stack (the paper itself is not vendored here), so a value is
// traceable to the code that describes the paper's behaviour rather than to a
// recollection of it. Where the paper gives a MECHANISM but no number (e.g. the
// safety switch of App. C.8, whose 0.05 threshold is this repo's choice), the
// parameter is deliberately LEFT ALONE — see "not from the paper" at the bottom.
//
// PINNING INCLUDES VALUES THAT ALREADY AGREE WITH THE DEFAULT (α = 0, β = 0,
// MaxSupport = 3). A preset is a statement of a configuration, not a diff
// against today's defaults: if a default later moves, the preset must keep
// saying the paper's thing, and `--print-changed` should read as "this is the
// paper's setup" rather than "here is what happens to differ this week".
//
// THIS PRESET IS A REFERENCE POINT, NOT A RECOMMENDATION. Several of these
// choices are measurably WORSE in this engine — depth-1 leaves cost 142.3 cp
// against 108.9 at the shipped depth 2 on 128 matched positions
// (ObscuroAgent.js `_leafEval`), and bounding a win at the eval clamp gives up
// the asymmetric own-king-hang penalty that fixed a real failure (the AI walking
// its king onto a square a hidden pawn covered). That is the point: it is the
// arm you measure against, with `move-quality.mjs` / `strength-belief.mjs`.
// ---------------------------------------------------------------------------

import { loadSettings, deepMerge } from './config.js';

// The paper's DESIGN points: the choices that are choices, at any budget. This
// half is affordable — it is the one to A/B against the defaults, because it
// changes what the engine believes and how it prices leaves without changing
// how much search it buys.
const ZHANG_SANDHOLM_DESIGN = {
  chess: {
    // Leaf evaluation at DEPTH 1, full breadth and nothing wider.
    //
    // "The paper runs its leaf evaluation at DEPTH 1 (App. C.5) and gets its
    // strength from the search aggregating many worlds and growing the tree"
    // (ObscuroAgent.js `_leafEval`). This repo ships 2–4 because at its tree
    // size the leaves have to see the tactics the tree cannot — measured, and
    // the measurement is quoted at the declaration site.
    //
    // cols is a FLOOR on MultiPV, not a cap (`Math.max(actions.length, cols)` in
    // scoreChildren), so 0 means "price exactly this node's children" — the
    // paper's batched node heuristic, with no extra lines requested.
    CHESS_DIAL: { leafEval: { sfDepth: 1, cols: 0 } },
    // The same design point for a TIME limit, where depth is bought by the clock
    // instead of the dial: cap the iterative-deepening ladder at one rung.
    MAX_SF_DEPTH: 1,

    // A win is worth the EVAL CLAMP, not a multiple of it.
    //
    // "The paper bounds ALL utilities (u: Z → [−1,+1], evals clamped inside it)"
    // (ObscuroAgent.js SEARCH_WIN; vendor/obscuro ObscuroAgent.js `_winValue`).
    // Under that reading a certain win and a maximal evaluation are the same
    // number, so SEARCH_WIN = LEAF_CLAMP. This repo ships 8000 ≈ 5.3× the clamp
    // instead, which also serves as the own-king-hang penalty (`kingHang`);
    // equalising them gives that up, deliberately, for the comparison.
    LEAF_CLAMP: 1500,
    SEARCH_WIN: 1500,

    // The exact belief, at the paper's size. "paper: |P| usually ≤ 10⁶ (C++);
    // avg ~17k" (exactBelief.js CAP). The time guard rises with it because
    // "raising CAP and TIME_GUARD_MS together is the standard way to trade turn
    // latency for staying exact longer" — leaving the 4 s guard in place would
    // just make the guard, not the cap, decide when exactness is abandoned.
    EXACT_BELIEF_CAP: 1000000,
    EXACT_BELIEF_TIME_GUARD_MS: 60000,

    // Worlds are drawn UNIFORMLY from P, and every world in the information set
    // is equally likely. "sampled at random without replacement from the set of
    // possible states" (FogChess.sampleWorlds); "Uniform 1/N says every world in
    // my information set is equally likely — which is what the paper assumes,
    // because it samples uniformly and has no better model" (vendor/obscuro
    // search.js, root-world reach). α = 0 and β = 0 are already the shipped
    // defaults — for their own measured reasons, quoted in exactBelief.js — so
    // this pins an agreement rather than changing anything.
    SAMPLE_ALPHA_DEFAULT: 0,
    REACH_WEIGHTING_DEFAULT: 0,

    // NO OPPONENT MODEL. The fitted move prior π(move | position) is this
    // repo's addition: it is what makes the belief a distribution instead of a
    // set, and the paper has nothing corresponding to it. Serving uniform π is
    // what "no better model" means concretely (movePrior.js UNIFORM_ONLY).
    //
    // Worth knowing before reading a result: at α = 0 and β = 0 the posterior
    // reaches PLAY through nothing at all, so this switch changes the belief
    // that `calibrate-belief.mjs` and the analysis panel report, and changes
    // which worlds are eligible under a nonzero α or β — but on its own, in this
    // preset, it does not change the move. Turning it off is the honest
    // statement of the paper's setup, not a lever on strength.
    MOVE_PRIOR_UNIFORM: true,
  },
  search: {
    // Purification: "Cap the support at MaxSupport (= 3)" (purify.js, §3.5 /
    // App. C.8, Fig. 8 lines 13–21). Already the default in both dial modes and
    // in purify.js itself; pinned in all three places because the dial's value
    // is what the search actually passes and the constant is only its fallback.
    PURIFY_MAX_SUPPORT: 3,
    DIAL: { power: { purifyMax: 3 }, time: { purifyMax: 3 } },
  },
};

// The paper's SCALE: hundreds of belief worlds, ~10⁶-node trees, seconds per
// move, and no difficulty dial at all — the paper's engine has one strength,
// full. Every knob the paper gives no number for therefore sits at the TOP of
// its shipped range rather than in the middle of a ramp.
//
// IN JAVASCRIPT THIS IS NOT THE PAPER'S MOVE TIME. The paper's numbers come from
// a C++ engine running solver and expander threads in parallel (vendor/obscuro
// search.js header); here the wall-clock budget is what actually binds, so what
// this block buys is "spend the whole budget in the paper's regime", not "match
// its throughput". Expect minutes, not seconds, if you raise the budget to
// match the tree cap.
const ZHANG_SANDHOLM_SCALE = {
  search: {
    DIAL: {
      // "it samples hundreds of worlds and grows ~10^6-node trees at
      // seconds/move" (vendor/obscuro ObscuroAgent.js `_config`, POWER mode).
      // 100 is the bottom of "hundreds" and already ~2× the top of this repo's
      // dial (48); maxInfosets caps tree size, which is the closest settable
      // stand-in for a node count.
      power: {
        worlds: 100,
        maxInfosets: 1000000,
        timeBudgetMs: 5000,
        // "The paper runs a solver thread and expander threads in parallel until
        // a time limit" (vendor/obscuro search.js header) — there is no round
        // cap to hit, so this is the TIME-mode "effectively unbounded" value and
        // the clock is left to bound the search.
        maxRounds: 100000,
        // No number in the paper; full strength means the top of each range,
        // which is what TIME mode already uses as its constant.
        expandPerRound: 24,
        cfrPerRound: 10,
        finalCfr: 200,
      },
      // A per-move time limit is the paper's own mode of operation, so the same
      // width and tree cap apply there.
      time: { worlds: 100, maxInfosets: 1000000 },
    },
  },
};

// ---------------------------------------------------------------------------
// NOT FROM THE PAPER, and therefore deliberately left at this repo's defaults.
// Named here so "the preset didn't set it" reads as a decision instead of an
// oversight:
//
//   search.SEARCH_DEFAULTS.safePmaxThreshold   the App. C.8 safety switch is the
//     paper's; 0.05 is this repo's threshold for "almost no incentive".
//   search.SEARCH_DEFAULTS.stableSnapshotEps   the "stable since T½" window is
//     the paper's; 1e-3 is a numerical-noise floor.
//   search.MIN_SUPPORT_PROB, search.MIN_EXPANDED_ROOT_WORLDS,
//   search.SEARCH_DEFAULTS.carriedRootWidthFloor / finalCfrDeadlineFactor
//     implementation guards with no counterpart in the paper.
//   search.RESOLVE_PRIOR_UNIFORM_BLEND  0.5 — the even mix of blueprint and
//     uniform that kluss.js quotes as the paper's Resolve prior α(J). Moving it
//     would be a departure FROM the paper, not toward it.
//   chess.CHESS_DIAL.proportionalPick  the perfect-information sampling
//     temperature is this engine's difficulty handicap for chess WITHOUT fog,
//     which is not the game the paper plays. Left alone so the preset changes
//     only fog play.
//   chess.MOVE_PRIOR_FITTED_WEIGHTS  the model's shape and its fitted numbers
//     are switched off wholesale by MOVE_PRIOR_UNIFORM rather than zeroed field
//     by field, so the fit stays intact for the run you compare against.
//   chess.ANALYSIS_DEFAULTS  the analysis panel is this engine's UI, not play.
// ---------------------------------------------------------------------------

/**
 * The shipped presets: name → { about, settings }.
 *
 * `about` is one line, for `--list-presets`. `settings` is a settings tree in
 * the documented key space, so `loadSettings(PRESETS[name].settings)` is all a
 * host needs and every other layer (a settings file, `--set`,
 * `gameSpecific.obscuro`, constructor opts) still outranks it.
 */
export const PRESETS = {
  'zhang-sandholm': {
    about: "the paper's setup: its design points AND its scale (slow in JS)",
    settings: deepMerge(ZHANG_SANDHOLM_DESIGN, ZHANG_SANDHOLM_SCALE),
  },
  'zhang-sandholm-design': {
    about: "the paper's design points at this engine's budget — the arm to measure",
    settings: ZHANG_SANDHOLM_DESIGN,
  },
};

// Short aliases, because these two get typed a lot on a command line. Kept out
// of PRESETS itself so `--list-presets` prints each configuration once.
const ALIASES = { paper: 'zhang-sandholm', 'paper-design': 'zhang-sandholm-design' };

/** Every preset name, aliases included, for messages and completion. */
export function presetNames() {
  return [...Object.keys(PRESETS), ...Object.keys(ALIASES)];
}

/**
 * One preset's settings tree, by name or alias.
 *
 * An unknown name throws rather than falling back to the defaults, for the same
 * reason an unknown settings key does (see src/config.js): the run would
 * otherwise complete, look plausible, and have measured nothing.
 */
export function preset(name) {
  const key = ALIASES[name] ?? name;
  const found = PRESETS[key];
  if (!found) {
    throw new Error(`settings: unknown preset "${name}" — known presets: ${presetNames().join(', ')}`);
  }
  return found.settings;
}

/**
 * Install a preset as the explicit settings layer — the programmatic form of
 * `--preset <name>`, equivalent to `loadSettings(preset(name))`.
 *
 * It occupies the SAME layer as `loadSettings`, so a caller that wants both a
 * preset and a file of its own should merge them itself (which is what the CLI
 * does: the preset underneath, the file on top).
 */
export function loadPreset(name) {
  return loadSettings(preset(name));
}

/** `--list-presets` output: one aligned line per configuration. */
export function formatPresets() {
  const aliasesOf = key => Object.keys(ALIASES).filter(a => ALIASES[a] === key);
  const rows = Object.entries(PRESETS).map(([key, { about }]) => {
    const alias = aliasesOf(key);
    return [alias.length ? `${key} (${alias.join(', ')})` : key, about];
  });
  const width = Math.max(...rows.map(([label]) => label.length));
  return rows.map(([label, about]) => `  ${label.padEnd(width)}  ${about}`).join('\n');
}
