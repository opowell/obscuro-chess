// ---------------------------------------------------------------------------
// obscuro-chess — the fog-of-war chess specialisation of the Obscuro search.
//
// The generic search lives upstream in vendor/obscuro (github.com/opowell/
// obscuro-ai) and contains no game knowledge. This package supplies the chess
// half of the paper's division of labour: the rules and observation model, a
// Stockfish leaf evaluator, and the belief trackers that say which positions are
// still possible given everything a player has seen.
//
// The public surface, grouped:
//   AGENTS        ChessObscuroAgent / ObscuroAgent (the equilibrium agent),
//                 ChessAgent (a plain alpha-beta + Stockfish agent),
//                 obscuroStrategy / analyzeObscuro* (inspection helpers)
//   BELIEF        ExactBelief (the paper's P) and the heuristic particle Belief,
//                 plus the move prior π(move | position) that weights P
//   ENGINE        the vendored Stockfish backend (Node + browser)
//   RULES         board / move generation / FEN, shared by all of the above
//   SETTINGS      every fog-chess default in one place (see docs/PARAMETERS.md)
//   GAME          FogChess, the GameDefinition tying all of the above together
// ---------------------------------------------------------------------------

// --- Game -----------------------------------------------------------------
export { FogChess, initialBoard, boardToUnits } from './FogChess.js';
export { playMatch } from './playMatch.js';

// --- Agents ---------------------------------------------------------------
export {
  ChessObscuroAgent, ObscuroAgent, setGame, getGame,
  obscuroStrategy, analyzeObscuro, analyzeObscuroProgressive, cpSumsOverWorlds,
  makeChessLeafEval, makeIterativeChessLeafEval,
  getLeafEvalStats, resetLeafEvalStats,
  CHESS_DIAL, ANALYSIS_DEFAULTS, LEAF_CLAMP, SEARCH_WIN, MAX_SF_DEPTH,
} from './ObscuroAgent.js';

export {
  ChessAgent, evaluate, alphaBeta, scoreMoveInParticle, advanceGs, clearTT, FULL_INFO_CFG,
  CHESS_AGENT_DIAL,
} from './ChessAgent.js';

// --- Belief ---------------------------------------------------------------
export {
  ExactBelief, getExactBelief, fromBoardObject, toBoardObject, genFogMoves,
  setDefaultMovePrior, getDefaultMovePrior, setMovePriorForSeat,
  setBeliefSampleAlpha, getBeliefSampleAlpha, setBeliefSampleAlphaForSeat,
  setBeliefReachWeighting, setBeliefReachWeightingForSeat, getBeliefReachWeighting,
} from './exactBelief.js';

export { Belief, getBelief, possibleSquaresFor, impossiblePlacement } from './belief.js';
export { makeMovePrior, UNIFORM_PRIOR, FITTED_WEIGHTS } from './movePrior.js';
export { replayBelief, placementSig } from './beliefCalibration.js';

// --- Engine ---------------------------------------------------------------
export {
  available, bestMove, multiPV, quit, setCacheDir,
  stockfishBestAction, sfOptsForDifficulty, difficultyToNumber,
} from './stockfish.js';

// --- Rules ----------------------------------------------------------------
export {
  FILES, fileOf, rankOf, fileIndex, squareAt, squareToXY, squareToGrid,
  isAttackedBy, isKingInCheck, applyMoveToBoard, getVisibleSquares, renderBoard,
} from './board.js';
export { getAllLegalMoves, getAllFogMoves, pseudoLegalForUnit } from './moves.js';
export { toFEN, uciToAction } from './fen.js';
export { PIECE_VALUE, PST } from './pieceTables.js';

// --- Settings -------------------------------------------------------------
// `settings` is the DEFAULTS (every constant, in one namespace). The rest is
// the resolution layer: fix a parameter, or reshape the difficulty dial that
// scales them all. See docs/SETTINGS.md.
export * as settings from './settings.js';
export {
  loadSettings, setOverrides, resetSettings, rediscoverSettings,
  settingsTree, settingsProvenance, settingsEpoch, isOverridden,
  param, ramp, dialParam, setPath,
  resolvedConfig, formatConfig,
  validate as validateSettings,
  SETTING_PATHS, SETTINGS_FILENAME, SETTINGS_ENV_VAR,
} from './config.js';
