// ---------------------------------------------------------------------------
// Belief calibration: replay recorded games and ask how much probability the
// belief put on REALITY.
//
// The belief's whole claim is that some boards in P are likelier than others.
// That claim is falsifiable, and this is what falsifies it. Walk a recorded fog
// game one seat at a time, and at every one of that seat's turns locate the TRUE
// position inside P and record:
//
//   • logLoss = −log w(true) — the honest scalar. How much probability did we put
//     on what actually happened? Lower is better, and the baseline to beat is
//     log|P|, which is what a flat posterior scores by definition. A prior that
//     cannot beat log|P| is not a model, it is a decoration.
//   • rank — where the true position sits in the weight ordering (1 = the
//     belief's top pick). This is the number the analysis panel's world stepper
//     literally shows, so it is the user-facing version of the same question.
//   • ms — beginTurn wall time, against exactBelief's TIME_GUARD_MS.
//
// Shared by exact-belief.test.js (which asserts the invariant that the true
// position is always IN P) and scripts/calibrate-belief.mjs (which compares
// priors). Kept out of the test file so the script isn't importing tests.
//
// MATCHING THE TRUE POSITION. We match on PLACEMENT only, ignoring castling
// rights and en passant, for the same reason the invariant test does: placement
// is what the fog hides and what the panel draws, and demanding cr/ep agreement
// would let a disagreement in bookkeeping masquerade as a belief failure. Several
// members of P can therefore share the true placement (differing only in rights);
// their weights are SUMMED for logLoss, since "the probability the belief assigns
// to the real board" is the probability of that placement. Rank is computed
// against the heaviest of them, i.e. how far the viewer would have to step.
// ---------------------------------------------------------------------------

import { FogChess } from './FogChess.js';
import { ExactBelief, fromBoardObject } from './exactBelief.js';

/** Placement-level signature (ignores piece ids, castling rights, en passant). */
export function placementSig(board) {
  let s = '';
  for (const f of 'abcdefgh') {
    for (let r = 1; r <= 8; r++) {
      const p = board[f + r];
      if (!p) { s += '.'; continue; }
      const c = p.type === 'knight' ? 'n' : p.type[0];
      s += p.ownerId === 'white' ? c.toUpperCase() : c;
    }
  }
  return s;
}

/**
 * Replay one recorded session from one seat's point of view.
 *
 * `sess` is a parsed session JSON. Returns
 *   { turns: [...], gaveUpAtPly: number|null, tracker }
 * where each turn record is
 *   { ply, size, found, mass, rank, logLoss, logSize, ties, ms }
 * `found: false` means the true position was NOT in P — a violation of the
 * subsystem's central invariant, and the thing the replay tests assert against.
 *
 * `game` is the GameDefinition the session was RECORDED with. It defaults to this
 * package's FogChess, which is right for anything this repo produced; an embedder
 * replaying its own sessions passes its own definition, so a rules difference in
 * its engine cannot show up here as a belief failure.
 */
export function replayBelief(sess, aiColor, { game = FogChess, movePrior = null, maxPlies = Infinity, sampleN = 0, sampleTrials = 24, rng = Math.random } = {}) {
  let state = game.createInitialState(sess.params.players, sess.params.config);
  const tracker = new ExactBelief(aiColor, movePrior);
  const turns = [];
  let gaveUpAtPly = null;
  const log = sess.log ?? [];
  for (let i = 0; i < Math.min(log.length, maxPlies); i++) {
    const pa = log[i].playerActions?.[0];
    if (!pa?.action) break;
    if (pa.playerId === aiColor) {
      const view = game.getVisibleState(state, aiColor);
      const t0 = Date.now();
      tracker.beginTurn(view, view.turnNumber ?? null);
      const ms = Date.now() - t0;
      if (!tracker.exact) { gaveUpAtPly = i; break; }

      // Compare on the typed representation, not via toBoardObject + string keys:
      // |P| runs to 200k and this loop runs at every turn of every game of every
      // prior being compared, so an object allocation per member would dominate
      // the whole harness.
      const truth = fromBoardObject(state.board, null, null);
      const P = tracker.positions;
      const W = tracker.weights;
      let mass = 0, ties = 0, best = 0;
      const trueIdx = new Set();
      for (let j = 0; j < P.length; j++) {
        const p = P[j];
        let same = true;
        for (let s = 0; s < 64; s++) if (p[s] !== truth[s]) { same = false; break; }
        if (!same) continue;
        trueIdx.add(j);
        const w = W ? W[j] : 1 / P.length;
        mass += w; ties++;
        if (w > best) best = w;
      }
      // Rank among MEMBERS by weight: how many worlds the belief thinks are
      // likelier than the real one. 1 = the belief's top pick is the truth.
      let rank = null;
      if (ties > 0) {
        rank = 1;
        for (let j = 0; j < P.length; j++) if ((W ? W[j] : 1 / P.length) > best) rank++;
      }
      // SAMPLE COVERAGE — the number that actually explains the strength result.
      // The search does not consume the posterior; it consumes an n-world DRAW from
      // it. So the question that matters for play is not "how much probability did
      // we put on reality" but "does the handful of worlds the search will actually
      // look at contain reality at all". Measured at both extremes of the sampling
      // exponent over the same belief: α=1 draws ∝ the posterior, α=0 uniformly.
      let hit1 = null, hit0 = null;
      if (sampleN > 0 && ties > 0) {
        let h1 = 0, h0 = 0;
        for (let t = 0; t < sampleTrials; t++) {
          if (tracker.sampleIndices(sampleN, rng, 1).some(j => trueIdx.has(j))) h1++;
          if (tracker.sampleIndices(sampleN, rng, 0).some(j => trueIdx.has(j))) h0++;
        }
        hit1 = h1 / sampleTrials;
        hit0 = h0 / sampleTrials;
      }

      turns.push({
        ply: i,
        size: P.length,
        found: ties > 0,
        mass, ties, rank, ms,
        logLoss: mass > 0 ? -Math.log(mass) : Infinity,
        logSize: Math.log(P.length), // the flat-posterior baseline
        hit1, hit0,
      });
      tracker.commitOurMove(pa.action);
    }
    state = game.applyActions(state, [pa]);
  }
  return { turns, gaveUpAtPly, tracker };
}

/** Mean of a numeric array, or null when empty. */
export function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}
