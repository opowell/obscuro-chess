// ---------------------------------------------------------------------------
// Fit the search's leaf value to how FOG games actually end.
//
//   node scripts/fit-fog-value.mjs --sessions <corpus> [--depth 4] [--folds 3] [--workers N]
//                                  [--rows <file>] [--model <name>] [--write]
//
// WHY. The search values a leaf by an expected result. Stockfish's win/draw/loss
// is the expected result of FULL-INFORMATION engine play: it calls the opening
// 94% drawn, while 19 of the Chess.com crawl's 3,156 fog games were drawn, and it
// calls a position "lost" where, under fog, the opponent cannot see everything and
// a defended king still has chances — whereas a capturable king loses for
// certain (the capturer always sees it). On 300 hidden king threats that made the
// agent leave its king capturable 52–54% of the time (scripts/hidden-threats.mjs).
//
// WHAT IS FITTED. For a position with side X to move:
//
//     P(X wins the game) = σ(β₀ + β₁·L + β₂·L·m + β₃·m + β₄·w)
//
//   L  logit of Stockfish's expected score for X, (W + D/2)/1000 at --depth,
//      clipped to (0.0005, 0.9995) so a mate is a finite, very strong input
//   m  material on the board / 78 (the phase of the game)
//   w  1 if X is white
//
// and, in the models that add them, the INFORMATION STATE (src/fogFeatures.js):
// how much of the board each side sees and what share of the other's pieces.
// Stockfish values the true board as if both sides saw it all, so without these
// a move is never credited for what it reveals or hides beyond the horizon.
// Every model in MODELS is fitted and compared held out, paired by game against
// `base` and against its matched control without the information state
// (CONTROL); --write writes --model (default: the best), and only if it beats
// Stockfish and beats both by more than two standard errors.
//
// --rows <file> caches the per-position features and results, so a refit with
// other models needs no engine.
//
// by maximum likelihood (Newton's method; the objective is concave) on every
// position of every game in the corpus whose outcome is decisive, with 3-fold
// cross-validation by GAME. A drawn game is 0.6% of the corpus and is left out:
// the model is a win probability, and the search values a leaf at
// P(win) − P(loss) = 2·P(win) − 1.
//
// Positions the engine refuses (a king already capturable — the search values
// those itself, as a loss or a win) are skipped.
// ---------------------------------------------------------------------------

import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CLIP = 0.0005;
const BASE = ['1', 'L', 'Lm', 'm', 'w'];
const MODELS = {
  base: BASE,
  vis: [...BASE, 'visX', 'visY'],
  seen: [...BASE, 'seenX', 'seenY'],
  info: [...BASE, 'visX', 'visY', 'seenX', 'seenY'],
  // Controls: does the information state still help once the curve in the
  // engine's score is flexible enough not to be the thing it corrects?
  flex: [...BASE, 'e', 'em'],
  flexinfo: [...BASE, 'e', 'em', 'visX', 'visY', 'seenX', 'seenY'],
  bal: [...BASE, 'e', 'em', 'bal'],
  balinfo: [...BASE, 'e', 'em', 'bal', 'visX', 'visY', 'seenX', 'seenY'],
  balseen: [...BASE, 'e', 'em', 'bal', 'seenX', 'seenY'],
};
// Each information model's matched control: the same model without the
// information state. Its gain is measured against this, not against `base`.
const CONTROL = { vis: 'base', seen: 'base', info: 'base', flexinfo: 'flex', balinfo: 'bal', balseen: 'bal' };

if (isMainThread) await main(); else await worker();

async function main() {
  const { makeArgReader } = await import('../src/cli.js');
  const { loadCorpus, describeCorpus } = await import('../src/corpus.js');
  const argv = process.argv.slice(2);
  const arg = makeArgReader(argv);
  const sessions = arg('sessions', join(HERE, '..', 'test', 'fixtures'));
  const depth = Number(arg('depth', '4'));
  const FOLDS = Number(arg('folds', '3'));
  const workers = Number(arg('workers', String(Math.max(1, availableParallelism() - 1))));
  const { games, stats } = loadCorpus(sessions, {});
  console.log(describeCorpus(games, stats));

  const rowsFile = arg('rows', null);
  let rows;   // [game, features, y]
  if (rowsFile && existsSync(rowsFile)) {
    rows = JSON.parse(readFileSync(rowsFile, 'utf8'));
    console.log(`Read ${rows.length} positions from ${rowsFile}.`);
  } else {
    rows = await evaluateCorpus(games, sessions, depth, workers);
    if (rowsFile) writeFileSync(rowsFile, JSON.stringify(rows));
  }
  const Y = rows.map(r => r[2]);
  const G = rows.map(r => r[0]);
  const N = rows.length;
  const all = Array.from({ length: N }, (_, i) => i);
  const sig = z => 1 / (1 + Math.exp(-z));
  // Rows cached before a feature existed derive it here from L and m.
  for (const [, f] of rows) if (f.e === undefined) { f.e = sig(f.L); f.em = f.e * f.m; }
  const design = names => rows.map(([, f]) => names.map(n => f[n]));
  // Per-position log-loss under beta.
  const losses = (X, beta, idx) => idx.map(i => {
    let z = 0; for (let k = 0; k < beta.length; k++) z += beta[k] * X[i][k];
    const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(z)));
    return Y[i] ? -Math.log(p) : -Math.log(1 - p);
  });
  // Newton–Raphson on the logistic log-likelihood: concave, a handful of steps.
  const fit = (X, idx) => {
    const K = X[0].length, beta = new Array(K).fill(0);
    for (let it = 0; it < 50; it++) {
      const g = new Array(K).fill(0);
      const H = Array.from({ length: K }, () => new Array(K).fill(0));
      for (const i of idx) {
        let z = 0; for (let k = 0; k < K; k++) z += beta[k] * X[i][k];
        const p = sig(z), r = Y[i] - p, wgt = p * (1 - p);
        for (let a = 0; a < K; a++) {
          g[a] += r * X[i][a];
          for (let b = 0; b < K; b++) H[a][b] += wgt * X[i][a] * X[i][b];
        }
      }
      const step = solve(H, g);
      let move = 0;
      for (let k = 0; k < K; k++) { beta[k] += step[k]; move += Math.abs(step[k]); }
      if (move < 1e-10) break;
    }
    return beta;
  };
  // Held-out loss of every position, each predicted by the fold that left its game out.
  const heldOut = X => {
    const out = new Array(N);
    for (let f = 0; f < FOLDS; f++) {
      const tr = [], te = [];
      for (let i = 0; i < N; i++) (G[i] % FOLDS === f ? te : tr).push(i);
      const beta = fit(X, tr), l = losses(X, beta, te);
      te.forEach((i, j) => { out[i] = l[j]; });
    }
    return out;
  };
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  // Stockfish's own full-information estimate, as a win probability: its
  // expected score, which spends the draw mass half on each side.
  const sfLoss = all.map(i => { const e = sig(rows[i][1].L); return Y[i] ? -Math.log(e) : -Math.log(1 - e); });
  const p0 = mean(Y);
  const baseRate = -(p0 * Math.log(p0) + (1 - p0) * Math.log(1 - p0));

  console.log(`\n${N} positions from ${new Set(G).size} decisive games, depth ${depth}`);
  console.log(`=== ${FOLDS}-fold CV by game, log-loss of the game's result (nats; lower better) ===`);
  console.log(`  base rate ${baseRate.toFixed(4)}   Stockfish ${mean(sfLoss).toFixed(4)}`);
  // A game's positions share one result, so the paired difference is summed per
  // game and its standard error taken over games.
  const games_ = [...new Set(G)];
  const pairedSE = (a, b) => {
    const per = new Map();
    for (let i = 0; i < N; i++) per.set(G[i], (per.get(G[i]) ?? 0) + a[i] - b[i]);
    const d = games_.map(g => per.get(g));
    const md = mean(d), v = d.reduce((s, x) => s + (x - md) ** 2, 0) / (d.length - 1);
    return { diff: md * d.length / N, se: Math.sqrt(v / d.length) * d.length / N };
  };
  const results = {};
  for (const [name, names] of Object.entries(MODELS)) {
    const X = design(names);
    const l = heldOut(X);
    results[name] = { names, X, l, loss: mean(l) };
  }
  for (const [name, r] of Object.entries(results)) {
    const { diff, se } = pairedSE(r.l, results.base.l);
    r.diff = diff; r.se = se;
    const fmt = (d, e) => `${d >= 0 ? '+' : ''}${d.toFixed(5)} ± ${e.toFixed(5)} (z = ${(d / e).toFixed(1)})`;
    let vs = name === 'base' ? '' : `   vs base ${fmt(diff, se)}`;
    if (CONTROL[name] && CONTROL[name] !== 'base') {
      const c = pairedSE(r.l, results[CONTROL[name]].l);
      vs += `   vs ${CONTROL[name]} ${fmt(c.diff, c.se)}`;
    }
    console.log(`  ${name.padEnd(8)} ${r.loss.toFixed(4)}${vs}   [${r.names.join(', ')}]`);
  }
  const best = Object.keys(results).reduce((a, b) => (results[b].loss < results[a].loss ? b : a));
  const chosen = arg('model', best);
  if (!results[chosen]) { console.error(`No model ${chosen}; have ${Object.keys(MODELS).join(', ')}.`); process.exit(1); }
  const { names, X } = results[chosen];
  const beta = fit(X, all);
  console.log(`\n  ${chosen}, fitted on all: ${names.map((n, k) => `${n} ${beta[k].toFixed(4)}`).join(', ')}`);

  // Calibration: Stockfish's expected score vs how often the side to move won.
  console.log('\n  Stockfish expected score  →  fog win rate (observed)  fitted (at the bin\'s mean)');
  const bins = [0, 0.02, 0.1, 0.3, 0.45, 0.55, 0.7, 0.9, 0.98, 1.0001];
  for (let b = 0; b + 1 < bins.length; b++) {
    const ix = all.filter(i => { const e = sig(rows[i][1].L); return e >= bins[b] && e < bins[b + 1]; });
    if (!ix.length) continue;
    const obs = ix.reduce((a, i) => a + Y[i], 0) / ix.length;
    let pf = 0; for (const i of ix) { let z = 0; for (let k = 0; k < beta.length; k++) z += beta[k] * X[i][k]; pf += sig(z); }
    console.log(`    ${bins[b].toFixed(2)}–${Math.min(1, bins[b + 1]).toFixed(2)}   n=${String(ix.length).padStart(6)}   ${(100 * obs).toFixed(1).padStart(5)}%   ${(100 * pf / ix.length).toFixed(1).padStart(5)}%`);
  }

  if (argv.includes('--write')) {
    const r = results[chosen];
    if (!(r.loss < mean(sfLoss))) { console.error('\nNot writing: the model does not beat Stockfish held out.'); process.exit(1); }
    for (const c of new Set(['base', CONTROL[chosen] ?? 'base'])) {
      if (chosen === c) continue;
      const { diff, se } = pairedSE(r.l, results[c].l);
      if (!(diff < -2 * se)) { console.error(`\nNot writing: ${chosen} does not beat ${c} by two standard errors.`); process.exit(1); }
    }
    const file = join(HERE, '..', 'src', 'fogValueModel.js');
    writeFileSync(file, `// GENERATED by scripts/fit-fog-value.mjs --write. Do not edit by hand.
//
// P(side to move wins) = σ(Σ beta[k] · feature[k]), the features as computed by
// src/fogFeatures.js (L = logit of Stockfish's expected score at depth ${depth},
// clipped to ${CLIP}). Model '${chosen}', fitted ${new Date().toISOString().slice(0, 10)} on ${N} positions
// from ${new Set(G).size} decisive games (${sessions.split('/').pop()}). Held-out log-loss of
// the result ${r.loss.toFixed(4)} nats (base model ${results.base.loss.toFixed(4)}), against ${mean(sfLoss).toFixed(4)} for
// Stockfish's own expected score and ${baseRate.toFixed(4)} for the base rate (${FOLDS}-fold CV by game).

export const FOG_VALUE_MODEL = {
  depth: ${depth},
  clip: ${CLIP},
  features: [${names.map(n => `'${n}'`).join(', ')}],
  beta: [${beta.map(x => x.toFixed(6)).join(', ')}],
};
`);
    console.log(`\nWrote ${file}.`);
  }
  process.exit(0);
}

async function evaluateCorpus(games, sessions, depth, workers) {
  // Evaluate every position, in chunks of games handed to whichever worker is free.
  const CHUNK = 20;
  const chunks = [];
  for (let g = 0; g < games.length; g += CHUNK) chunks.push([g, Math.min(games.length, g + CHUNK)]);
  const rows = [];
  const t0 = Date.now();
  let next = 0, done = 0, lastReport = 0;
  await new Promise((resolve, reject) => {
    const pool = Array.from({ length: Math.min(workers, chunks.length) }, () =>
      new Worker(fileURLToPath(import.meta.url), { workerData: { sessions, depth } }));
    const feed = w => { if (next < chunks.length) w.postMessage(chunks[next++]); else w.terminate(); };
    for (const w of pool) {
      w.on('error', reject);
      w.on('message', r => {
        for (const x of r) rows.push(x);
        done++;
        const now = Date.now();
        if (done === chunks.length || now - lastReport > 30000) {
          lastReport = now;
          const min = (now - t0) / 60000;
          console.log(`  ${done}/${chunks.length} chunks, ${rows.length} positions, ${min.toFixed(1)} min`);
        }
        if (done === chunks.length) resolve();
        feed(w);
      });
      feed(w);
    }
  });

  return rows;
}

// Solve H·x = g (H symmetric positive definite) by Gaussian elimination.
function solve(H, g) {
  const n = g.length, A = H.map((r, i) => [...r, g[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}

async function worker() {
  const { loadCorpus } = await import('../src/corpus.js');
  const { FogChess } = await import('../src/FogChess.js');
  const { toFEN } = await import('../src/fen.js');
  const { isAttackedBy } = await import('../src/board.js');
  const { fogFeatures } = await import('../src/fogFeatures.js');
  const { multiPV, setCacheDir } = await import('../src/stockfish.js');
  // A cache of its own: these evaluations are not search leaves.
  const dir = join(HERE, '..', 'vendor', 'stockfish', 'fog-value');
  mkdirSync(dir, { recursive: true });
  setCacheDir(dir);
  const { games } = loadCorpus(workerData.sessions, {});
  const kingSq = (b, c) => Object.keys(b).find(s => b[s]?.ownerId === c && b[s].type === 'king');

  parentPort.on('message', async ([g0, g1]) => {
    const out = [];
    for (let gi = g0; gi < g1; gi++) {
      const { sess, result } = games[gi];
      let st = FogChess.createInitialState(sess.params.players, sess.params.config);
      const winner = result === '1-0' ? 'white' : result === '0-1' ? 'black' : null;
      if (!winner) continue;   // a draw, or no recorded result
      for (const e of sess.log ?? []) {
        const pa = e.playerActions?.[0];
        if (!pa?.action || FogChess.getResult?.(st)) break;
        const me = pa.playerId, opp = me === 'white' ? 'black' : 'white';
        const myK = kingSq(st.board, me), oppK = kingSq(st.board, opp);
        // Skip positions the engine refuses: either king capturable.
        if (myK && oppK && !isAttackedBy(st.board, oppK, me) && !isAttackedBy(st.board, myK, opp)) {
          const pv = await multiPV(toFEN(st.board, st.gameSpecific, me === 'white' ? 'w' : 'b', st.turnNumber ?? 1), { multipv: 1, depth: workerData.depth });
          if (pv?.length && pv[0].wdl) {
            const [W, D] = pv[0].wdl;
            out.push([gi, fogFeatures((W + D / 2) / 1000, st.board, me, CLIP), winner === me ? 1 : 0]);
          }
        }
        try { st = FogChess.applyActions(st, [pa]); } catch { break; }
      }
    }
    parentPort.postMessage(out);
  });
}
