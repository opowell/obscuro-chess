// ---------------------------------------------------------------------------
// Fog-chess Obscuro parameter defaults — the fog-chess counterpart of
// vendor/obscuro/src/settings.js. Every number here TUNES fog-of-war chess
// specifically (belief sampling, the exact position-set tracker, the move
// prior, Stockfish leaf-eval scaling, the analysis panel's search sizes) —
// nothing here is read by the generic Obscuro search itself.
//
// Like vendor/obscuro/src/settings.js, this module does not hold its own
// literals — it re-exports the named constants each file already declares (and
// documents WHY next to the code that uses them), so there is exactly one
// place to come and read "what does chess's Obscuro do by default", while the
// numbers and their reasoning stay next to the logic they tune.
//
// See docs/PARAMETERS.md for the full prose write-up of every entry below, and
// docs/SETTINGS.md for how to override one. The generic search's own knobs are
// documented upstream, in vendor/obscuro/docs/PARAMETERS.md.
//
// EVERY NAME BELOW IS ALSO A SETTINGS KEY, as `chess.<NAME>` — src/config.js
// resolves overrides against exactly this export list, so there is no second
// vocabulary to keep in step. (test/settings.test.js asserts the two match.)
// ---------------------------------------------------------------------------

// src/ObscuroAgent.js — the chess-specific difficulty dial (leaf-eval
// depth/breadth, proportional-pick sampling), the terminal/material clamps,
// and the analysis panel's own search-size defaults.
export {
  CHESS_DIAL,
  ANALYSIS_DEFAULTS,
  LEAF_CLAMP,
  SEARCH_WIN,
  MAX_SF_DEPTH,
  REFUSED_CHILD_CAP,
} from './ObscuroAgent.js';

// src/ChessAgent.js — the plain alpha-beta agent's own difficulty ramp
// (search depth, score noise, fog particle count) and how it prices a cloud of
// particles (tail risk, the scouting bonus, the fog score clamp).
export { CHESS_AGENT_DIAL, CHESS_AGENT_SCORING } from './ChessAgent.js';

// src/belief.js — the heuristic particle belief (used once exact
// tracking is lost; see exactBelief.js below for the primary tracker).
export {
  MAX_POSSIBLE,
  THREAT_BIAS,
  MAX_LURKERS,
  RECAPTURE_TYPE_WEIGHT,
  MAX_ATTEMPTS_PER_PARTICLE,
  PHANTOM_CHECK_REJECT_WINDOW,
} from './belief.js';

// src/exactBelief.js — the exact position-set tracker P (the paper's
// belief). SAMPLE_ALPHA_DEFAULT is the posterior-sampling sharpness (ships at
// 0 = uniform-over-P; see the long comment on setBeliefSampleAlpha for why).
export {
  CAP as EXACT_BELIEF_CAP,
  TIME_GUARD_MS as EXACT_BELIEF_TIME_GUARD_MS,
  REACQUIRE_BOUND,
  SAMPLE_ALPHA_DEFAULT,
  REACH_WEIGHTING_DEFAULT,
} from './exactBelief.js';

// src/movePrior.js — π(move | position), the opponent model that
// turns the exact belief from a set into a distribution. FITTED_WEIGHTS is
// the production model (fitted by MLE, not hand-tuned — see the file's own
// header before changing any of its numbers).
// RATING_SLOPE tilts those weights by the opponent's rating, continuously
// (weight_k(r) = base_k + slope_k·z, z = (r − PIVOT)/SCALE, |z| ≤ Z_CLAMP). It
// ships as ZEROS, and stays that way until a corpus with ratings shows the tilt
// beats the flat model out of sample; the pivot/scale/clamp describe the corpus
// the slopes were fitted on, so a host serving its own slopes sets them too.
// MOVE_PRIOR_UNIFORM turns the model off entirely (the paper's setting).
export {
  FITTED_WEIGHTS as MOVE_PRIOR_FITTED_WEIGHTS,
  RATING_SLOPE as MOVE_PRIOR_RATING_SLOPE,
  RATING_PIVOT as MOVE_PRIOR_RATING_PIVOT,
  RATING_SCALE as MOVE_PRIOR_RATING_SCALE,
  RATING_Z_CLAMP as MOVE_PRIOR_RATING_Z_CLAMP,
  UNIFORM_ONLY as MOVE_PRIOR_UNIFORM,
} from './movePrior.js';

// src/stockfish.js — the vendored-engine backend: cache/recycle
// bookkeeping and the difficulty→(movetime, Skill Level) ramp used by the
// perfect-information paths (chooseAction's time-mode branch, sfOptsForDifficulty).
export {
  RECYCLE_AFTER,
  CACHE_MAX,
  STOP_POLL_MS,
  SF_CACHE_DIR,
  LEGACY_DIFFICULTY,
  DEFAULT_DIFFICULTY,
  SF_DIFFICULTY_RAMP,
} from './stockfish.js';
