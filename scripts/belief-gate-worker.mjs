// ---------------------------------------------------------------------------
// One worker of fit-move-prior.mjs's belief gate. Replays a held-out game from
// one seat through the exact belief under a given prior, and reports how much
// mass the belief put on the true position at each turn.
//
// Replays are independent and their cost varies by two orders of magnitude (a
// short game is milliseconds, one whose P reaches 10⁶ is a minute), so the main
// thread hands them out one at a time to whichever worker is free.
// ---------------------------------------------------------------------------

import { parentPort, workerData } from 'node:worker_threads';
import { loadCorpus } from '../src/corpus.js';
import { replayBelief } from '../src/beliefCalibration.js';
import { makeMovePrior, UNIFORM_PRIOR, FITTED_WEIGHTS, tablesFromVector } from '../src/movePrior.js';

const { sessions, maxGames, unit, floor, thetas } = workerData;
// The same call the main thread made, so game indices agree.
const { games } = loadCorpus(sessions, { maxGames });

const priors = new Map();
function priorFor(arm, fold) {
  const key = arm === 'learned+floor' ? `${arm}|${fold}` : arm;
  let prior = priors.get(key);
  if (prior) return prior;
  if (arm === 'uniform-π') prior = UNIFORM_PRIOR;
  else if (arm === 'shipped') prior = makeMovePrior(FITTED_WEIGHTS);
  else {
    const { tables, castleBonus } = tablesFromVector(Float64Array.from(thetas[fold], x => x * unit));
    prior = makeMovePrior({
      temperature: unit, floor, captureWeight: 1, promoWeight: 1,
      pstWeight: [0, 1, 1, 1, 1, 1, 1], castleBonus, tables,
    });
  }
  priors.set(key, prior);
  return prior;
}

parentPort.on('message', ({ id, arm, fold, gi, seat }) => {
  const r = replayBelief(games[gi].sess, seat, { movePrior: priorFor(arm, fold) });
  parentPort.postMessage({
    id, arm, giveup: r.gaveUpAtPly != null,
    turns: r.turns.map(t => [t.found ? 1 : 0, t.logLoss, t.logSize, t.rank ?? 0, t.ply]),
  });
});
