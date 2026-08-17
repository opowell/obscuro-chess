// ---------------------------------------------------------------------------
// Take a new corpus of recorded fog games from "someone handed me this file" to
// "the shipped prior is updated, or it isn't and here is why", in one command.
//
//   node scripts/adopt-corpus.mjs <corpus>            # measure, change nothing
//   node scripts/adopt-corpus.mjs <corpus> --write    # …and ship it if it wins
//
// <corpus> is anything src/corpus.js reads: a directory, a .zip, a .pgn, a
// session .json, a crawl .json, any of them .gz.
//
// WHY THIS EXISTS. Adopting a corpus is not one step, it is five, and the four
// after the first are the ones people skip. Doing it by hand for the 2026-08-06
// Chess.com crawl took a dozen invocations and turned up two scraper artifacts
// that silently ate half the games before anyone noticed. This script runs the
// same sequence in the same order every time and refuses to ship on a number it
// did not check:
//
//   1. INGEST HEALTH. How much of the corpus actually parsed, and does our
//      replay agree with the source's own ply counts? A corpus that half-loads
//      is the failure mode that produces a confident number over a third of the
//      data. Below --min-health this stops, because everything after it would be
//      measuring the parser rather than the model.
//   2. THE MOVE-LEVEL FIT. Cross-validated by game, refit vs the weights already
//      shipped — which have not seen this corpus, so the gap between them is the
//      case for refitting.
//   3. THE GATE. Belief log-loss of the true position on held-out games. This is
//      the number that decides, and it is NOT the move-level one: the
//      observation filter has already priced much of what π would say, and the
//      two have disagreed before.
//   4. RATING. Whether conditioning π on the opponent's rating buys anything,
//      as a continuous interaction.
//   5. THE HONEST CAVEAT. π reaches MOVE SELECTION only when the search draws
//      worlds by the posterior (`SAMPLE_ALPHA_DEFAULT`) or weights their reach
//      by it (`REACH_WEIGHTING_DEFAULT`). Both ship at 0. So a better prior is a
//      better BELIEF and, at the shipped defaults, not yet better PLAY — this
//      script says so every time rather than letting a log-loss win be read as a
//      strength win.
//
// --write only fires when steps 1–3 pass, and it delegates to fit-move-prior.mjs
// rather than reimplementing the write, so there is one code path that edits
// movePrior.js.
// ---------------------------------------------------------------------------

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadCorpus, describeCorpus, ratingSpread } from '../src/corpus.js';
import { SAMPLE_ALPHA_DEFAULT, REACH_WEIGHTING_DEFAULT } from '../src/exactBelief.js';
import { makeArgReader } from '../src/cli.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FITTER = join(HERE, 'fit-move-prior.mjs');

const argv = process.argv.slice(2);
const arg = makeArgReader(argv);
const has = name => argv.includes('--' + name);

// The corpus is the one positional argument. Walk argv skipping flags and the
// values of the flags that take one, so `--folds 5 games.zip` and
// `games.zip --folds 5` both find it.
const VALUE_FLAGS = new Set(['--folds', '--e2e-games', '--min-health']);
const corpus = (() => {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) { if (VALUE_FLAGS.has(argv[i])) i++; continue; }
    return argv[i];
  }
  return null;
})();

if (!corpus || has('help')) {
  console.log(`
adopt-corpus <corpus> [--write] [--folds N] [--e2e-games N] [--min-health P]

  <corpus>        a directory, .zip, .pgn, session .json or crawl .json (.gz ok)
  --write         update FITTED_WEIGHTS if the corpus passes every gate
  --folds N       cross-validation folds, by game (default 5)
  --e2e-games N   games to run the belief gate over (default 60; it is the slow
                  step, ~4s per game per arm)
  --min-health P  refuse to proceed below this replay-health fraction (default 0.9)
`.trim());
  process.exit(corpus ? 0 : 1);
}
if (!existsSync(corpus)) { console.error(`adopt-corpus: no such corpus: ${corpus}`); process.exit(1); }

const FOLDS = arg('folds', '5');
const E2E_GAMES = arg('e2e-games', '60');
const MIN_HEALTH = Number(arg('min-health', '0.9'));

const rule = (t) => console.log(`\n${'═'.repeat(72)}\n${t}\n${'═'.repeat(72)}`);

/** Run the fitter, echo its output, and hand back the text for scraping. */
function fitter(args) {
  const r = spawnSync(process.execPath, [FITTER, '--sessions', corpus, ...args], {
    encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
  });
  const out = (r.stdout ?? '') + (r.stderr ?? '');
  // Node's SQLite notice is noise here; the fitter itself never emits warnings.
  console.log(out.split('\n').filter(l => !/ExperimentalWarning|trace-warnings/.test(l)).join('\n').trim());
  if (r.status !== 0) { console.error(`\nadopt-corpus: the fitter exited ${r.status}; stopping.`); process.exit(1); }
  return out;
}

const num = (text, re) => { const m = re.exec(text); return m ? Number(m[1]) : null; };

// --- 1. ingest health --------------------------------------------------------

rule('1/5  INGEST — did the corpus actually parse?');
const { games, stats } = loadCorpus(corpus);
console.log(describeCorpus(games, stats));
if (!games.length) { console.error('\nadopt-corpus: nothing loaded; stopping.'); process.exit(1); }

const plies = games.reduce((a, g) => a + g.plies, 0);
const checkable = games.filter(g => g.sourcePlies != null);
const agreeing = checkable.filter(g => g.plies === g.sourcePlies).length;
// Health is measured against the SOURCE's own ply counts where it states them,
// because that is the only external check on the replay; when it states none
// (PGN, session JSON), fall back to "no game truncated at an unparseable move".
const health = checkable.length ? agreeing / checkable.length
  : (games.length - stats.truncated) / games.length;
console.log(`${games.length} games, ${plies} plies replayed`);
if (checkable.length) {
  console.log(`replay agrees with the source's own ply count on ${agreeing}/${checkable.length} games ` +
    `(${(100 * health).toFixed(1)}%)`);
} else {
  console.log(`no source ply counts to check against; ${stats.truncated} games truncated at an unparseable move`);
}
const spread = ratingSpread(games);
console.log(spread
  ? `ratings: ${spread.n} rated seats, ${spread.min}–${spread.max} (p10 ${spread.p10}, median ${spread.median}, p90 ${spread.p90})`
  : 'ratings: none in this corpus');

if (health < MIN_HEALTH) {
  console.error(`\nadopt-corpus: replay health ${(100 * health).toFixed(1)}% is below ` +
    `--min-health ${(100 * MIN_HEALTH).toFixed(0)}%. STOPPING.\n\n` +
    'That is a parser/format problem, not a model result, and fitting through it\n' +
    'would train on boards that never occurred. Inspect the tokens that failed —\n' +
    'src/pgn.js normalizeMoveList documents the two artifacts already handled\n' +
    '(termination glyphs, and scraped SAN split by whitespace).');
  process.exit(1);
}
console.log(`\n✓ health ${(100 * health).toFixed(1)}% ≥ ${(100 * MIN_HEALTH).toFixed(0)}%`);

// --- 2 & 3. the fit, and the gate -------------------------------------------

rule('2/5  MOVE-LEVEL FIT — refit vs the weights already shipped, held out');
const cv = fitter(['--folds', FOLDS]);
const moveGain = num(cv, /refitting on this corpus buys (-?[\d.]+) nats/);

rule(`3/5  THE GATE — belief log-loss of the true position (${E2E_GAMES} games)`);
console.log('This is the number that decides. Slow: it replays the belief per arm.');
// --max-games caps the whole corpus load, so the `fitted` arm is refit on a
// fraction of THIS subset while `shipped` carries whatever it was fitted on.
// Below about half the corpus that handicap is big enough to flip the verdict,
// and a user trimming --e2e-games to save time would never guess why.
const e2eFrac = Math.min(1, Number(E2E_GAMES) / games.length);
if (e2eFrac < 0.5) {
  console.log(`\n! --e2e-games ${E2E_GAMES} is ${(100 * e2eFrac).toFixed(0)}% of the ${games.length} loaded games.`);
  console.log('! The `fitted` arm is refit on a fraction of that subset, while `shipped`');
  console.log('! keeps whatever it was fitted on, so this comparison UNDERSTATES the refit.');
  console.log('! A negative gate here is weak evidence; raise --e2e-games before believing it.');
}
console.log('');
const e2e = fitter(['--folds', FOLDS, '--max-games', E2E_GAMES, '--e2e']);
const shippedLL = num(e2e, /^\s*shipped\s+ll=([\d.]+)/m);
const fittedLL = num(e2e, /^\s*fitted\s+ll=([\d.]+)/m);
const notIn = num(e2e, /notInP=(\d+)/);
const beliefGain = shippedLL != null && fittedLL != null ? shippedLL - fittedLL : null;

// --- 4. rating ---------------------------------------------------------------

rule('4/5  RATING — does conditioning π on opponent strength buy anything?');
const rating = spread ? fitter(['--folds', FOLDS, '--rating']) : (console.log('no ratings in this corpus; skipped.'), '');
const ratingGain = num(rating, /Δ ([+-][\d.]+)\s+(?:ships|no better|LOSES)/);

// --- 5. verdict --------------------------------------------------------------

rule('5/5  VERDICT');
const fmt = x => x == null ? 'n/a' : `${x >= 0 ? '+' : ''}${x.toFixed(4)} nats`;
console.log(`  ingest health       ${(100 * health).toFixed(1)}%`);
console.log(`  move-level gain     ${fmt(moveGain)}   (refit vs shipped, held out)`);
console.log(`  BELIEF gain         ${fmt(beliefGain)}   (the gate: ${shippedLL ?? '?'} → ${fittedLL ?? '?'})`);
console.log(`  rating conditioning ${fmt(ratingGain)}   (sloped vs flat, held out)`);
console.log(`  notInP              ${notIn ?? '?'}   (must be 0 — subsystem invariant, not a metric)`);

const gatePassed = beliefGain != null && beliefGain > 0 && notIn === 0;
// A corpus the shipped weights were ALREADY fitted on scores ~0 on both gains,
// which is a different situation from a corpus that has nothing to offer, and
// deserves different advice.
const alreadyAdopted = moveGain != null && Math.abs(moveGain) < 0.005;
const verdict = gatePassed
  ? 'SHIP — the refit beats the shipped weights on held-out belief log-loss.'
  : alreadyAdopted
    ? 'NOTHING TO DO — the shipped weights already fit this corpus as well as a\n  refit does, which is what you see when it has been adopted already.'
    : 'DO NOT SHIP — the refit does not beat the shipped weights on the belief gate.';
console.log(`\n  ${verdict}`);
if (!gatePassed && e2eFrac < 0.5) {
  console.log(`  (…and the gate ran on ${(100 * e2eFrac).toFixed(0)}% of the corpus, which handicaps the refit —`);
  console.log('   re-run with a larger --e2e-games before treating this as settled.)');
}

// The caveat that keeps a calibration win from being reported as a strength win.
console.log(`
  ── and the thing this does NOT measure ────────────────────────────────────
  π reaches MOVE SELECTION only through the world draw, and only when the
  search is told to use the posterior. Both switches ship at zero:

      SAMPLE_ALPHA_DEFAULT     = ${SAMPLE_ALPHA_DEFAULT}   (draw ∝ w^α; 0 = uniform over P)
      REACH_WEIGHTING_DEFAULT  = ${REACH_WEIGHTING_DEFAULT}   (β; 0 = flat 1/N reach)

  At those defaults a better prior is a better BELIEF — better analysis, better
  calibration — and changes nothing about which move the AI plays. To convert
  belief accuracy into strength, re-test the α/β decision against this corpus:

      node scripts/move-quality.mjs --sessions ${corpus} --arm alpha
      node scripts/move-quality.mjs --sessions ${corpus} --arm null    # control
      node scripts/strength-belief.mjs --arm alpha

  Whether the shipped α is right is a decision worth revisiting against fresh
  measurements (see docs/STRENGTH-PLAN.md), not a settled one.`);

// --- write -------------------------------------------------------------------

if (has('write')) {
  if (!gatePassed) {
    console.error('\nadopt-corpus: --write refused; the gate did not pass.');
    process.exit(1);
  }
  rule('WRITING');
  fitter(['--folds', FOLDS, '--rating', '--write']);
  console.log('\nNow re-run `npm test`, and update the corpus provenance in');
  console.log('docs/STRENGTH-PLAN.md and docs/PARAMETERS.md — the numbers above are');
  console.log('the ones to record.');
} else {
  console.log('\n(Nothing was changed. Re-run with --write to ship the refit.)');
}
