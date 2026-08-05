// ---------------------------------------------------------------------------
// Compare move priors by how much probability the belief puts on reality.
//
//   node scripts/calibrate-belief.mjs [--max-games N] [--seat white|black|both]
//
// Walks every recorded chess fog game in sessions/ from both seats, under each
// prior below, and reports mean log-loss of the TRUE position against the
// flat-posterior baseline log|P|. See beliefCalibration.js for what is measured
// and why, and OBSCURO-MOVE-PRIOR-PLAN.md step 5 for the gate this decides:
//
//   • uniform π must beat log|P| — otherwise the weight MECHANISM (colliding
//     histories summing, branching factor, the observation filter) is worthless.
//   • a real prior must beat uniform π — otherwise the MODEL is worthless, no
//     matter how reasonable its terms look, and it should not ship.
//
// The corpus is passed with --sessions <dir> (see below); without it the script
// replays the three fixtures the tests use, which is a smoke test, not a fit.
// ---------------------------------------------------------------------------

import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { replayBelief, mean } from '../src/beliefCalibration.js';
import { makeMovePrior, UNIFORM_PRIOR, FITTED_WEIGHTS } from '../src/movePrior.js';
import { applyCliSettings, maybePrintConfig, makeArgReader } from '../src/cli.js';

const HERE = dirname(fileURLToPath(import.meta.url));
// The recorded games to replay. Defaults to the three fixtures the test suite
// uses, so the script runs from a bare clone; that is enough to smoke-test the
// pipeline and nowhere near enough to draw a conclusion from. Point --sessions
// at a real corpus for that (battle-simulator keeps its recorded games in its
// own untracked sessions/ directory).
const { rest: argv, printConfig } = applyCliSettings();
await maybePrintConfig(printConfig);
const arg = makeArgReader(argv);
const SESSIONS = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
const maxGames = Number(arg('max-games', '999'));
// `--sample-n 16` additionally reports SAMPLE COVERAGE: how often an n-world draw
// from the belief contains the true position, at α=1 (∝ posterior) and α=0
// (uniform). 16 is the analysis walk's batch size and the right order of magnitude
// for what the search actually looks at. See beliefCalibration.js.
const sampleN = Number(arg('sample-n', '0'));
const seatArg = arg('seat', 'both');
const seats = seatArg === 'both' ? ['white', 'black'] : [seatArg];

// `--taus 800,400,200` sweeps temperatures only; the default list adds the two
// single-term ablations, which answer "where is the signal actually coming from".
const taus = arg('taus', null);
const PRIORS = taus
  ? [['uniform-π', UNIFORM_PRIOR],
     ...taus.split(',').map(t => [`τ=${t}`, makeMovePrior({ temperature: Number(t) })])]
  : [
    ['uniform-π', UNIFORM_PRIOR],
    // The shipped model. NOTE it is fitted on these same sessions, so its number
    // here is IN-SAMPLE and flattering; the honest held-out comparison is
    // `fit-move-prior.mjs --e2e`, which refits per fold. This arm is for
    // "did anything regress", not for quoting.
    ['FITTED (shipped, in-sample)', makeMovePrior(FITTED_WEIGHTS)],
    ['τ=800', makeMovePrior({ temperature: 800 })],
    ['τ=400', makeMovePrior({ temperature: 400 })],
    ['τ=300', makeMovePrior({ temperature: 300 })],
    ['τ=200', makeMovePrior({ temperature: 200 })],
    ['τ=100', makeMovePrior({ temperature: 100 })],
    // Capture term only: is the PST delta pulling any weight, or is all the signal
    // in "they take material"?
    ['τ=300 cap-only', makeMovePrior({ temperature: 300, pstWeight: 0 })],
    // PST term only, the complement of the above.
    ['τ=300 pst-only', makeMovePrior({ temperature: 300, captureWeight: 0 })],
  ];

if (!existsSync(SESSIONS)) {
  console.error(`No sessions at ${SESSIONS}.\n` +
    'Pass --sessions <dir> pointing at a directory of recorded session JSONs.');
  process.exit(1);
}

const games = [];
for (const f of readdirSync(SESSIONS).sort()) {
  if (!f.endsWith('.json')) continue;
  let sess;
  try { sess = JSON.parse(readFileSync(join(SESSIONS, f), 'utf8')); } catch { continue; }
  const p = sess.params;
  if (p?.game !== 'chess') continue;
  const c = p.config ?? {};
  if (!(c.fogOfWar || c.fog)) continue;
  if ((sess.log?.length ?? 0) < 10) continue;
  games.push({ file: f, sess });
  if (games.length >= maxGames) break;
}
console.log(`${games.length} chess fog games, seats: ${seats.join('+')}\n`);

const rows = [];
for (const [label, prior] of PRIORS) {
  const logLoss = [], logSize = [], ranks = [], top1 = [], times = [], hit1 = [], hit0 = [];
  let turns = 0, notFound = 0, gaveUp = 0, infinite = 0;
  const t0 = Date.now();
  for (const { sess } of games) {
    for (const seat of seats) {
      const r = replayBelief(sess, seat, { movePrior: prior, sampleN });
      if (r.gaveUpAtPly != null) gaveUp++;
      for (const t of r.turns) {
        turns++;
        if (!t.found) { notFound++; continue; }
        if (!Number.isFinite(t.logLoss)) { infinite++; continue; }
        logLoss.push(t.logLoss);
        logSize.push(t.logSize);
        ranks.push(t.rank);
        top1.push(t.rank === 1 ? 1 : 0);
        times.push(t.ms);
        if (t.hit1 != null) { hit1.push(t.hit1); hit0.push(t.hit0); }
      }
    }
  }
  rows.push({
    label,
    turns,
    notFound,
    gaveUp,
    infinite,
    logLoss: mean(logLoss),
    baseline: mean(logSize),
    medRank: median(ranks),
    top1: mean(top1),
    maxMs: Math.max(0, ...times),
    hit1: mean(hit1),
    hit0: mean(hit0),
    wallS: (Date.now() - t0) / 1000,
  });
  const r = rows[rows.length - 1];
  console.log(
    `${r.label.padEnd(16)} logloss ${fmt(r.logLoss)}  baseline log|P| ${fmt(r.baseline)}` +
    `  Δ ${fmt(r.baseline - r.logLoss)}  medRank ${String(r.medRank).padStart(6)}` +
    `  top1 ${(r.top1 * 100).toFixed(1)}%  turns ${r.turns}` +
    `  notInP ${r.notFound}  gaveUp ${r.gaveUp}  w=0 ${r.infinite}` +
    `  maxTurnMs ${r.maxMs}  wall ${r.wallS.toFixed(1)}s` +
    (r.hit1 == null ? ''
      : `\n${' '.repeat(16)}sample coverage @${sampleN}:  α=1 ${(r.hit1 * 100).toFixed(1)}%` +
        `   α=0 ${(r.hit0 * 100).toFixed(1)}%   (chance the search's draw contains reality)`));
}

console.log('\nΔ is how many nats better than a flat posterior over the same set;' +
  ' higher is better. medRank/top1 are the true board\'s place in the weight ordering.');

function fmt(x) { return x == null ? '  n/a ' : x.toFixed(3).padStart(7); }
function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}
