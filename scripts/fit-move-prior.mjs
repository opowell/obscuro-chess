// ---------------------------------------------------------------------------
// Fit π's weights to how people (and agents) ACTUALLY move under fog.
//
//   node scripts/fit-move-prior.mjs                 # fit + cross-validate
//   node scripts/fit-move-prior.mjs --e2e           # + belief log-loss, held out
//   node scripts/fit-move-prior.mjs --e2e --ablate  # + per-term ablations
//   node scripts/fit-move-prior.mjs --write         # update FITTED_WEIGHTS
//   node scripts/fit-move-prior.mjs --actor human   # fit one opponent type
//   node scripts/fit-move-prior.mjs --sessions corpus.zip        # zip / dir / PGN
//   node scripts/fit-move-prior.mjs --rating --write             # rating-tilted π
//
// WHY THIS EXISTS. The first version of the prior gave every term weight 1 and
// divided by one temperature, and τ was chosen by sweeping it against the
// belief's own log-loss. That is a one-parameter fit of a nine-parameter model,
// and it left most of the signal on the floor: the terms want temperatures a
// factor of ~6 apart, so no single τ can serve them (see OBSCURO-MOVE-PRIOR-PLAN
// .md). This script fits all nine by maximum likelihood instead.
//
// THE MODEL IS A CONDITIONAL LOGIT, which is exactly what π already is:
//
//     π(m | p) = softmax_m ( Σ_k V_k · features_k(p, m) )
//
// so fitting is textbook — the gradient of the log-likelihood is
// (features of the move actually played) − (their expectation under π), and the
// objective is concave. No local optima, no learning-rate archaeology.
//
// WHAT IT TRAINS ON. Every ply of every recorded fog game in sessions/: the true
// position, the full fog-legal move list from `genFogMoves` (production's own
// choice set — imported, not reimplemented), and which move was played. Both
// seats, all actor types, because π models "the opponent" generically.
//
// TWO THINGS TO BE CAREFUL ABOUT, both of which have bitten this subsystem:
//
//  • MOVE log-loss is not POSITION log-loss. The belief's gate is how much mass
//    it puts on the TRUE BOARD, and the observation filter already prices much of
//    what π would say. A model that predicts moves 48% better bought ~3× on the
//    board posterior — related, but not the same number, and only `--e2e`
//    measures the one that matters.
//  • Fitting on all the data and then reporting the fit's own score is how you
//    ship a number you can't reproduce. Everything here is cross-validated by
//    GAME (never by ply — plies inside one game are anything but independent),
//    and --write refits on everything only AFTER the CV number has been printed.
//
// OPPONENT RATING (--rating). A 1500-rated opponent and a 2400-rated one are not
// the same distribution, and π is supposed to be a model of the opponent. When
// the corpus carries ratings (PGN `WhiteElo`/`BlackElo`, a `rating` on a
// session's player, or a crawl's per-player Elo), rating enters CONTINUOUSLY, as
// an interaction rather than a bucket:
//
//     weight_k(r) = a_k + b_k · z(r),    z(r) = (r − PIVOT) / SCALE
//
// which is still a conditional logit — each interaction is just another column,
// f_k · z — so the same concave MLE fits all 18 parameters at once. Every rated
// decision informs every slope, which is the whole reason not to bucket: three
// bands would estimate each of nine weights from a third of the data, and a real
// effect drowns in that. It also has no edges to pick and no discontinuity where
// two adjacent ratings get different models.
//
// Doubling the parameters fits the training data better by construction, and
// that fact is worth nothing. `--rating` therefore reports the only number that
// can justify it: on HELD-OUT games, the sloped model against the flat one, over
// the same rated decisions. Slopes that do not earn it are not written.
// ---------------------------------------------------------------------------

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { genFogMoves, fromBoardObject } from '../src/exactBelief.js';
import { FogChess } from '../src/FogChess.js';
import { replayBelief, mean } from '../src/beliefCalibration.js';
import { loadCorpus, describeCorpus, ratingSpread } from '../src/corpus.js';
import {
  moveFeatures, NUM_FEATURES, FEATURE_NAMES, makeMovePrior, weightsFromVector,
  weightVector, UNIFORM_PRIOR, FITTED_WEIGHTS,
  RATING_PIVOT, RATING_SCALE, RATING_SLOPE, ratingZ,
} from '../src/movePrior.js';
import { applyCliSettings, maybePrintConfig, makeArgReader } from '../src/cli.js';

const HERE = dirname(fileURLToPath(import.meta.url));
// The recorded games to replay. Defaults to the three fixtures the test suite
// uses, so the script runs from a bare clone; that is enough to smoke-test the
// pipeline and nowhere near enough to draw a conclusion from. Point --sessions
// at a real corpus for that — a directory, a .zip, a .pgn or a .json, in any
// nesting (see src/corpus.js). battle-simulator keeps its recorded games in its
// own untracked sessions/ directory.
const { rest: argv, printConfig } = applyCliSettings();
await maybePrintConfig(printConfig);
const arg = makeArgReader(argv);
const SESSIONS = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
const has = name => argv.includes('--' + name);
const FOLDS = Number(arg('folds', '3'));
const L2 = Number(arg('l2', '1e-3'));
const ITERS = Number(arg('iters', '1500'));
const ACTOR = arg('actor', null);
const MAX_GAMES = Number(arg('max-games', '999'));
const RATING = has('rating');
// A floor on the gain. Slopes that beat the flat model by 0.0004 nats have found
// nothing; shipping them costs nine numbers and a rating lookup at serve time to
// buy noise. 0.01 is the same order as what `floor` costs (0.008), i.e. the
// smallest difference this subsystem has ever treated as real.
const MIN_RATING_GAIN = Number(arg('min-rating-gain', '0.01'));

// Serving divides the weighted feature sum by a temperature, so the weight
// vector is only defined up to that scale; TEMPERATURE fixes the unit and
// nothing else. Weight_k = τ · (logits per raw feature unit).
//
// The fit itself standardizes each feature by its RMS first. That is not
// cosmetic: a queen capture is 900 raw units and the castle indicator is 1, and
// on raw features the optimizer converges on the big ones while the castle
// weight is still crawling toward its true value of ~200. The first version of
// this script skipped that step, castle came out at ~0, and the KING weight
// silently absorbed it — castling is a king move with a large positive PST
// delta, so a starved castle term flipped the king's sign. If you change the
// feature set, keep the standardization.
const TEMPERATURE = 100;

const PIECE_CODE = { pawn: 1, knight: 2, bishop: 3, rook: 4, queen: 5, king: 6 };
const sqToIdx = sq => (sq.charCodeAt(1) - 49) * 8 + (sq.charCodeAt(0) - 97);

// --- data --------------------------------------------------------------------

if (!existsSync(SESSIONS)) {
  console.error(`No corpus at ${SESSIONS}.\n` +
    'Pass --sessions <path> pointing at a directory, .zip, .pgn or .json of recorded fog games.');
  process.exit(1);
}

const { games, stats } = loadCorpus(SESSIONS, { maxGames: MAX_GAMES });
if (!games.length) {
  console.error(`No chess fog games in ${SESSIONS}.\n  ${describeCorpus(games, stats)}`);
  process.exit(1);
}
console.log(describeCorpus(games, stats));

/** Locate a recorded action inside genFogMoves' output. */
function indexOfAction(moves, action) {
  const f = sqToIdx(action.from), t = sqToIdx(action.to);
  const castle = action.type === 'castle' ? (action.side === 'kingside' ? 1 : 2) : 0;
  const promo = action.payload?.promote ? PIECE_CODE[action.payload.promote] : 0;
  for (let j = 0; j < moves.length; j++) {
    const m = moves[j];
    if (m.f === f && m.t === t && m.castle === castle && m.promo === promo) return j;
  }
  return -1;
}

let unmatched = 0;
const examples = [];   // { F: Float64Array[], chosen, actor, game, movedType, band }
games.forEach(({ sess, actors, ratings }, gi) => {
  let state = FogChess.createInitialState(sess.params.players, sess.params.config);
  for (const entry of sess.log ?? []) {
    const pa = entry.playerActions?.[0];
    if (!pa?.action) break;
    const gs = state.gameSpecific;
    const sign = pa.playerId === 'white' ? 1 : -1;
    const pos = fromBoardObject(state.board, gs.castlingRights, gs.enPassantTarget);
    const moves = genFogMoves(pos, sign);
    const chosen = indexOfAction(moves, pa.action);
    if (chosen < 0) unmatched++;
    // A one-move position carries no information about preferences.
    if (chosen >= 0 && moves.length > 1) {
      const F = moves.map(m => moveFeatures(pos, m, sign, new Float64Array(NUM_FEATURES)));
      examples.push({
        F, chosen, game: gi, actor: actors[pa.playerId] ?? '?',
        movedType: Math.abs(pos[sqToIdx(pa.action.from)]),
        // The rating of the SEAT THAT MOVED. π models whoever is choosing, and
        // in a 1500-vs-2400 game the two seats are two different opponents.
        rating: ratings[pa.playerId] ?? null,
      });
    }
    state = FogChess.applyActions(state, [pa]);
  }
});

// Standardize: s_k = 1/RMS_k over every candidate move, so every coordinate
// reaches the optimizer at the same scale. Everything from here on is in
// standardized units; `weightsFrom` converts back.
//
// The scale is computed over ALL examples, including held-out folds. That is a
// deliberate (and harmless) exception to the split: it changes only the
// optimizer's conditioning and the metric L2 is measured in, not the model
// family — and the CV number is unmoved by L2 across two orders of magnitude.
const SCALE = new Float64Array(NUM_FEATURES).fill(1);
{
  const sq = new Float64Array(NUM_FEATURES);
  let n = 0;
  for (const ex of examples) for (const f of ex.F) { n++; for (let k = 0; k < NUM_FEATURES; k++) sq[k] += f[k] * f[k]; }
  for (let k = 0; k < NUM_FEATURES; k++) {
    const rms = Math.sqrt(sq[k] / Math.max(1, n));
    SCALE[k] = rms > 1e-9 ? 1 / rms : 1;
  }
  for (const ex of examples) for (const f of ex.F) for (let k = 0; k < NUM_FEATURES; k++) f[k] *= SCALE[k];
}
/** standardized vector → the weights `makeMovePrior` takes, at TEMPERATURE. */
const weightsFrom = U => Array.from(U, (u, k) => TEMPERATURE * u * SCALE[k]);
/** …and back, so an existing model can be scored in the same space. */
const stdFrom = w => Float64Array.from(w, (x, k) => x / (TEMPERATURE * SCALE[k]));

console.log(`${games.length} fog games → ${examples.length} decisions` +
  (unmatched ? `  (WARNING: ${unmatched} recorded actions not found in genFogMoves output)` : ''));
if (unmatched) {
  // Not fatal, but it means the fitter and production disagree about the choice
  // set, which is the one assumption everything here rests on.
  console.log('  ^ genFogMoves and the recorded game disagree — investigate before trusting these weights.');
}

// --- conditional logit -------------------------------------------------------

/** Mean log-likelihood per decision under weight vector V. */
function logLik(V, idxs) {
  let ll = 0;
  for (const i of idxs) {
    const { F, chosen } = examples[i];
    let max = -Infinity;
    const s = new Float64Array(F.length);
    for (let j = 0; j < F.length; j++) {
      let v = 0;
      for (let k = 0; k < NUM_FEATURES; k++) v += V[k] * F[j][k];
      s[j] = v;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let j = 0; j < F.length; j++) sum += Math.exp(s[j] - max);
    ll += (s[chosen] - max) - Math.log(sum);
  }
  return ll / idxs.length;
}

/** log|M| — what a uniform π scores on the same decisions. The baseline. */
function uniformLogLoss(idxs) {
  let s = 0;
  for (const i of idxs) s += Math.log(examples[i].F.length);
  return s / idxs.length;
}

/**
 * Full-batch Adam. The objective is concave, so this is only about getting there
 * quickly across features whose scales differ by ~100× (a queen capture is 9
 * units, a castle indicator is 0.01); plain gradient descent needs a different
 * step size for each and Adam picks them itself.
 */
function fit(idxs, { iters = ITERS, lr = 0.05, l2 = L2 } = {}) {
  const V = new Float64Array(NUM_FEATURES);
  const g = new Float64Array(NUM_FEATURES);
  const m = new Float64Array(NUM_FEATURES);
  const v = new Float64Array(NUM_FEATURES);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  for (let it = 1; it <= iters; it++) {
    g.fill(0);
    for (const i of idxs) {
      const { F, chosen } = examples[i];
      const n = F.length;
      const s = new Float64Array(n);
      let max = -Infinity;
      for (let j = 0; j < n; j++) {
        let x = 0;
        for (let k = 0; k < NUM_FEATURES; k++) x += V[k] * F[j][k];
        s[j] = x;
        if (x > max) max = x;
      }
      let sum = 0;
      const e = new Float64Array(n);
      for (let j = 0; j < n; j++) { e[j] = Math.exp(s[j] - max); sum += e[j]; }
      // ∇ = f(chosen) − E_π[f]
      for (let k = 0; k < NUM_FEATURES; k++) {
        let exp = 0;
        for (let j = 0; j < n; j++) exp += (e[j] / sum) * F[j][k];
        g[k] += F[chosen][k] - exp;
      }
    }
    for (let k = 0; k < NUM_FEATURES; k++) {
      const grad = g[k] / idxs.length - l2 * V[k];
      m[k] = b1 * m[k] + (1 - b1) * grad;
      v[k] = b2 * v[k] + (1 - b2) * grad * grad;
      const mh = m[k] / (1 - Math.pow(b1, it));
      const vh = v[k] / (1 - Math.pow(b2, it));
      V[k] += lr * mh / (Math.sqrt(vh) + eps);
    }
  }
  return V;
}

// --- the same conditional logit, widened by the rating interaction -----------
//
// Column 2k is f_k and column 2k+1 is f_k · z(r), so the 18-vector W means
// weight_k(r) = W[2k] + W[2k+1]·z. Everything else — concavity, the
// f(played) − E_π[f] gradient, Adam — is unchanged; this is one design matrix,
// not a second model. The features are already standardized, and z is O(1) by
// construction, so the interaction columns arrive at the optimizer at a sane
// scale without a second standardization pass.
const WIDE = 2 * NUM_FEATURES;
const zOf = ex => Math.max(-1.5, Math.min(1.5, ratingZ(ex.rating, RATING_PIVOT, RATING_SCALE)));

/** Row j of example `ex`, widened in place into `out` (length WIDE). */
function wideRow(ex, j, out) {
  const z = zOf(ex), F = ex.F[j];
  for (let k = 0; k < NUM_FEATURES; k++) { out[2 * k] = F[k]; out[2 * k + 1] = F[k] * z; }
  return out;
}

function logLikWide(W, idxs) {
  const row = new Float64Array(WIDE);
  let ll = 0;
  for (const i of idxs) {
    const ex = examples[i], n = ex.F.length;
    const s = new Float64Array(n);
    let max = -Infinity;
    for (let j = 0; j < n; j++) {
      wideRow(ex, j, row);
      let v = 0;
      for (let k = 0; k < WIDE; k++) v += W[k] * row[k];
      s[j] = v;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Math.exp(s[j] - max);
    ll += (s[ex.chosen] - max) - Math.log(sum);
  }
  return ll / idxs.length;
}

function fitWide(idxs, { iters = ITERS, lr = 0.05, l2 = L2 } = {}) {
  const W = new Float64Array(WIDE), g = new Float64Array(WIDE);
  const m = new Float64Array(WIDE), v = new Float64Array(WIDE);
  const row = new Float64Array(WIDE);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  for (let it = 1; it <= iters; it++) {
    g.fill(0);
    for (const i of idxs) {
      const ex = examples[i], n = ex.F.length;
      const s = new Float64Array(n);
      let max = -Infinity;
      for (let j = 0; j < n; j++) {
        wideRow(ex, j, row);
        let x = 0;
        for (let k = 0; k < WIDE; k++) x += W[k] * row[k];
        s[j] = x;
        if (x > max) max = x;
      }
      let sum = 0;
      const e = new Float64Array(n);
      for (let j = 0; j < n; j++) { e[j] = Math.exp(s[j] - max); sum += e[j]; }
      for (let j = 0; j < n; j++) {
        wideRow(ex, j, row);
        const p = e[j] / sum;
        for (let k = 0; k < WIDE; k++) g[k] -= p * row[k];
      }
      wideRow(ex, ex.chosen, row);
      for (let k = 0; k < WIDE; k++) g[k] += row[k];
    }
    for (let k = 0; k < WIDE; k++) {
      const grad = g[k] / idxs.length - l2 * W[k];
      m[k] = b1 * m[k] + (1 - b1) * grad;
      v[k] = b2 * v[k] + (1 - b2) * grad * grad;
      W[k] += lr * (m[k] / (1 - Math.pow(b1, it))) / (Math.sqrt(v[k] / (1 - Math.pow(b2, it))) + eps);
    }
  }
  // STANDARDIZED units, like `fit` — `logLikWide` scores against the same
  // standardized features, so converting here would silently compare a serving-
  // unit weight vector to standardized data. `wideToServing` does the conversion
  // once, at the point of reporting and writing.
  return W;
}

/** Standardized 18-vector → serving units. Both halves share feature k's scale. */
const wideToServing = W => Array.from(W, (x, k) => TEMPERATURE * x * SCALE[k >> 1]);

const all = examples.map((_, i) => i);
const trainSet = ACTOR ? all.filter(i => examples[i].actor === ACTOR) : all;
if (ACTOR) console.log(`--actor ${ACTOR}: ${trainSet.length} of ${all.length} decisions`);

// Fold by GAME. Plies within a game share a position stream; splitting by ply
// would let the fit see the same board from both sides of the split and report a
// held-out number that is nothing of the sort.
const foldOf = gi => gi % FOLDS;
const inFold = (i, f) => examples[i].game % FOLDS === f;

console.log(`\n=== ${FOLDS}-fold CV by game, per-move log-loss (nats; lower better) ===`);
let cvFit = 0, cvUnif = 0, cvOld = 0, cvShip = 0, cvN = 0;
// The model as it shipped before fitting: every term weight 1, no castle bonus,
// τ = 200 — i.e. weight 100/200 per term in TEMPERATURE=100 units.
const oldV = stdFrom([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0]);
// The weights CURRENTLY SHIPPED, on the same held-out folds. `fitted` is refit
// per fold and has therefore seen this corpus; this arm is a fixed model that
// has not seen any of it. When you point the script at a corpus from a
// different population than the shipped numbers came off, the gap between these
// two arms IS the case for refitting — and if there is no gap, there is no case.
const shippedV = stdFrom(weightVector(FITTED_WEIGHTS)
  .map(x => x * (TEMPERATURE / FITTED_WEIGHTS.temperature)));
for (let f = 0; f < FOLDS; f++) {
  const tr = trainSet.filter(i => !inFold(i, f));
  const te = all.filter(i => inFold(i, f));
  if (!tr.length || !te.length) continue;
  const V = fit(tr);
  const fitLL = -logLik(V, te), unifLL = uniformLogLoss(te);
  const oldLL = -logLik(oldV, te), shipLL = -logLik(shippedV, te);
  cvFit += fitLL * te.length; cvUnif += unifLL * te.length;
  cvOld += oldLL * te.length; cvShip += shipLL * te.length; cvN += te.length;
  console.log(`  fold ${f}: n=${String(te.length).padStart(5)}  uniform ${unifLL.toFixed(3)}  ` +
    `τ=200 ${oldLL.toFixed(3)}  shipped ${shipLL.toFixed(3)}  fitted ${fitLL.toFixed(3)}`);
}
console.log(`  POOLED : uniform ${(cvUnif / cvN).toFixed(3)}  ` +
  `τ=200 ${(cvOld / cvN).toFixed(3)} (Δ ${((cvUnif - cvOld) / cvN).toFixed(3)})  ` +
  `shipped ${(cvShip / cvN).toFixed(3)} (Δ ${((cvUnif - cvShip) / cvN).toFixed(3)})  ` +
  `fitted ${(cvFit / cvN).toFixed(3)} (Δ ${((cvUnif - cvFit) / cvN).toFixed(3)})`);
console.log(`  refitting on this corpus buys ${((cvShip - cvFit) / cvN).toFixed(3)} nats over the shipped model.`);

// --- opponent rating, as a continuous interaction ----------------------------
//
// The model doubles to 18 parameters: for each feature f_k, a base weight a_k
// and a slope b_k multiplying f_k · z(r). Fitting is unchanged — an interaction
// is just another column, and the objective stays concave — so `fitWide` is the
// same optimizer over a wider design matrix.
//
// Everything here runs on RATED decisions only, and the flat model it is
// compared against is refitted on those same rated decisions. Giving unrated
// decisions z = 0 would silently assert they are average, and comparing against
// a flat model fitted on a different set would confound the slope with the
// change of population.
let ratingFit = null;   // { n, flatLL, slopedLL, delta, slope, spread }
if (RATING) {
  const rated = all.filter(i => examples[i].rating != null);
  console.log(`\n=== opponent rating as a continuous term — held-out, sloped vs flat ===`);
  const spread = ratingSpread(games);
  if (!rated.length) {
    console.log('  no rated decisions in this corpus; slopes cannot be fitted.');
    console.log('  (battle-simulator session JSON carries no ratings — use PGN or a crawl.)');
  } else if (rated.length < 500) {
    console.log(`  only ${rated.length} rated decisions — too few to fit 18 parameters; skipping.`);
  } else {
    console.log(`  ${rated.length} of ${all.length} decisions carry a rating; ` +
      `ratings ${spread.min}–${spread.max} (p10 ${spread.p10}, median ${spread.median}, p90 ${spread.p90})`);
    console.log(`  z = (rating − ${RATING_PIVOT}) / ${RATING_SCALE}`);

    let accFlat = 0, accSloped = 0, accUnif = 0, accN = 0;
    const inTrainSet = new Set(trainSet);
    for (let f = 0; f < FOLDS; f++) {
      const tr = rated.filter(i => !inFold(i, f) && inTrainSet.has(i));
      const te = rated.filter(i => inFold(i, f));
      if (!tr.length || !te.length) continue;
      const flatV = fit(tr);
      const wideV = fitWide(tr);
      accFlat += -logLik(flatV, te) * te.length;
      accSloped += -logLikWide(wideV, te) * te.length;
      accUnif += uniformLogLoss(te) * te.length;
      accN += te.length;
    }
    const flatLL = accFlat / accN, slopedLL = accSloped / accN;
    const delta = flatLL - slopedLL;
    console.log(`  n=${accN}  uniform ${(accUnif / accN).toFixed(3)}  ` +
      `flat ${flatLL.toFixed(3)}  sloped ${slopedLL.toFixed(3)}  ` +
      `Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}  ` +
      (delta > MIN_RATING_GAIN ? 'ships'
        : delta > 0 ? `no better than flat (Δ < ${MIN_RATING_GAIN})` : 'LOSES to flat'));

    // The slopes themselves, fitted on everything, reported per term in the
    // units a reader can act on: how much the weight moves per 400 Elo.
    const wide = wideToServing(fitWide(rated));
    const slope = [];
    for (let k = 0; k < NUM_FEATURES; k++) slope.push(wide[2 * k + 1]);
    console.log(`\n  per-term slope (change in weight per ${RATING_SCALE} Elo), base at ${RATING_PIVOT}:`);
    for (let k = 0; k < NUM_FEATURES; k++) {
      const a = wide[2 * k], b = wide[2 * k + 1];
      const rel = a !== 0 ? ` (${(100 * b / Math.abs(a)).toFixed(0)}% of base)` : '';
      console.log(`    ${FEATURE_NAMES[k].padEnd(11)} base ${a.toFixed(3).padStart(9)}   ` +
        `slope ${b.toFixed(3).padStart(9)}${rel}`);
    }
    ratingFit = { n: accN, flatLL, slopedLL, delta, slope, spread };
  }
}

// --- the real gate: position log-loss on held-out games ----------------------

if (has('e2e')) {
  console.log(`\n=== ${FOLDS}-fold CV, BELIEF log-loss of the true position (the gate) ===`);
  const arms = new Map([
    ['uniform-π', () => UNIFORM_PRIOR],
    ['τ=200 (old default)', () => makeMovePrior({ temperature: 200 })],
    ['τ=150 (old optimum)', () => makeMovePrior({ temperature: 150 })],
    // The old model sharpened past its optimum. Kept as a permanent arm because
    // "sharper is dangerous" was the lesson drawn from it, and the fitted model
    // needs to be compared against that lesson in the same measurement, not
    // against a number remembered from a different session's harness.
    ['τ=60 (old cliff)', () => makeMovePrior({ temperature: 60 })],
    // The weights currently in movePrior.js, which have not seen this corpus.
    // `fitted` below is refit on each training fold, so this pair is the whole
    // decision: ship the refit only if it beats what is already there ON THE
    // BELIEF, which is the gate move-level log-loss is not.
    ['shipped', () => makeMovePrior(FITTED_WEIGHTS)],
    ['fitted', V => makeMovePrior(weightsFromVector(weightsFrom(V), { temperature: TEMPERATURE }))],
    ['fitted+floor', V => makeMovePrior(weightsFromVector(weightsFrom(V), { temperature: TEMPERATURE, floor: FITTED_WEIGHTS.floor }))],
    // MLE lands on the right sharpness by construction, so this should LOSE —
    // the question is by how much, i.e. how steep the slope is now.
    ['fitted ×1.5 (cliff test)', V => makeMovePrior(
      weightsFromVector(weightsFrom(V).map(x => x * 1.5), { temperature: TEMPERATURE }))],
  ]);
  // --ablate: zero one term at a time, refitting nothing, to see which terms the
  // POSITION posterior actually cares about. (It is not the same ranking as the
  // move-level fit — the observation filter has already priced some of them.)
  if (has('ablate')) {
    const mut = (k, f) => V => {
      const w = weightsFrom(V);
      w[k] = f(w[k]);
      return makeMovePrior(weightsFromVector(w, { temperature: TEMPERATURE }));
    };
    arms.set('  ablate capture→0', mut(0, () => 0));
    arms.set('  ablate castle→0', mut(8, () => 0));
    arms.set('  ablate king→0', mut(7, () => 0));
    arms.set('  ablate king→+|w|', mut(7, x => Math.abs(x)));
  }
  const acc = new Map([...arms.keys()].map(k => [k, { ll: [], base: [], ranks: [], notIn: 0, giveups: 0, maxMs: 0 }]));
  for (let f = 0; f < FOLDS; f++) {
    const tr = trainSet.filter(i => !inFold(i, f));
    if (!tr.length) continue;
    const V = fit(tr);
    for (const [name, mk] of arms) {
      const prior = mk(V);
      const a = acc.get(name);
      games.forEach(({ sess }, gi) => {
        if (foldOf(gi) !== f) return;
        for (const seat of ['white', 'black']) {
          const r = replayBelief(sess, seat, { movePrior: prior });
          if (r.gaveUpAtPly != null) a.giveups++;
          for (const t of r.turns) {
            if (!t.found) { a.notIn++; continue; }
            a.ll.push(t.logLoss); a.base.push(t.logSize); a.ranks.push(t.rank);
            if (t.ms > a.maxMs) a.maxMs = t.ms;
          }
        }
      });
    }
  }
  for (const [name, a] of acc) {
    a.ranks.sort((x, y) => x - y);
    const med = a.ranks[Math.floor(a.ranks.length / 2)];
    console.log(`  ${name.padEnd(20)} ll=${mean(a.ll).toFixed(3)}  flat=${mean(a.base).toFixed(3)}  ` +
      `Δ=${(mean(a.base) - mean(a.ll)).toFixed(3)}  medRank=${med}  ` +
      `notInP=${a.notIn}  giveups=${a.giveups}  turns=${a.ll.length}  maxMs=${a.maxMs}`);
  }
  console.log('  (notInP must be 0 in every arm — it is the subsystem invariant, not a metric.)');
}

// --- final fit on everything, and optionally write it back -------------------

const V = weightsFrom(fit(trainSet));
console.log('\n=== fitted on all data (this is what --write ships) ===');
V.forEach((x, k) => {
  // τ_eff answers "what single temperature would this term want?", which is the
  // whole argument for per-term weights: the spread is what one τ could not do.
  const tau = k === 8 ? null : (x === 0 ? Infinity : TEMPERATURE / x);
  console.log(`  ${FEATURE_NAMES[k].padEnd(11)} ${x.toFixed(3).padStart(8)}` +
    (tau === null ? `   (flat bonus in cp; ${(x / TEMPERATURE).toFixed(2)} logits)`
      : `   τ_eff ≈ ${Math.abs(tau).toFixed(0)}${x < 0 ? '  (SIGN FLIPPED vs the hand model)' : ''}`));
});

/** A fitted weight vector → the object literal movePrior.js holds. */
function weightsLiteral(v, indent = '') {
  return `{\n` +
    `${indent}  temperature: ${TEMPERATURE},\n` +
    `${indent}  floor: ${FITTED_WEIGHTS.floor},\n` +
    `${indent}  captureWeight: ${v[0].toFixed(3)},\n` +
    `${indent}  promoWeight: ${v[1].toFixed(3)},\n` +
    `${indent}  //          -  pawn  knight bishop  rook  queen   king\n` +
    `${indent}  pstWeight: [0, ${v[2].toFixed(3)}, ${v[3].toFixed(3)}, ${v[4].toFixed(3)}, ` +
    `${v[5].toFixed(3)}, ${v[6].toFixed(3)}, ${v[7].toFixed(3)}],\n` +
    `${indent}  castleBonus: ${v[8].toFixed(1)},\n` +
    `${indent}}`;
}

const literal = `export const FITTED_WEIGHTS = ${weightsLiteral(V)};`;

// RATING_SLOPE ships only if the sloped model EARNED it on held-out games. If it
// did not, the zeros stay and serving reduces to FITTED_WEIGHTS at every rating,
// which is the honest outcome: "we have no evidence rating moves these weights".
let ratingLiteral = null;
if (ratingFit && ratingFit.delta > MIN_RATING_GAIN) {
  const s = ratingFit.slope;
  ratingLiteral =
    `export const RATING_SLOPE = [\n` +
    `  // ${ratingFit.n} held-out decisions, ${ratingFit.delta.toFixed(4)} nats better than flat.\n` +
    `  // Ratings ${ratingFit.spread.min}–${ratingFit.spread.max}; the fit is only evidence over that range.\n` +
    `  //  capture  promo   pawn  knight bishop  rook  queen   king  castle\n` +
    `  ${s.map(x => x.toFixed(3)).join(', ')},\n` +
    `];`;
}

if (has('write')) {
  // NOTE the path: movePrior.js lives in src/, not next to this script. It was
  // join(HERE, 'movePrior.js') until 2026-08-05, which meant --write threw
  // ENOENT for anyone who followed the documented regeneration instructions.
  const file = join(HERE, '..', 'src', 'movePrior.js');
  const src = readFileSync(file, 'utf8');
  const re = /export const FITTED_WEIGHTS = \{[\s\S]*?\n\};/;
  if (!re.test(src)) {
    console.error(`\nCould not find the FITTED_WEIGHTS literal in ${file} — not writing.`);
    process.exit(1);
  }
  let out = src.replace(re, literal);
  if (ratingLiteral) {
    const rre = /export const RATING_SLOPE = \[[\s\S]*?\n\];/;
    if (!rre.test(out)) {
      console.error('\nCould not find the RATING_SLOPE literal — not writing.');
      process.exit(1);
    }
    out = out.replace(rre, ratingLiteral);
  }
  writeFileSync(file, out);
  console.log(`\nWrote FITTED_WEIGHTS${ratingLiteral ? ' and RATING_SLOPE' : ''} to ${file}. Re-run the chess tests.`);
} else {
  console.log(`\n--write would replace src/movePrior.js's FITTED_WEIGHTS with:\n\n${literal}`);
  if (ratingLiteral) console.log(`\n…and RATING_SLOPE with:\n\n${ratingLiteral}`);
  else if (RATING) console.log('\n--write would leave RATING_SLOPE at zero (the slopes did not earn it).');
}
