// ---------------------------------------------------------------------------
// Fit the search's leaf value to how FOG games actually end.
//
//   node scripts/fit-fog-value.mjs --sessions <corpus> [--depth 4] [--folds 3] [--workers N] [--write]
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
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const MATERIAL = { pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9 };
const CLIP = 0.0005;
export const NUM_FEATURES = 5; // 1, L, L·m, m, w

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

  // Evaluate every position, in chunks of games handed to whichever worker is free.
  const CHUNK = 20;
  const chunks = [];
  for (let g = 0; g < games.length; g += CHUNK) chunks.push([g, Math.min(games.length, g + CHUNK)]);
  const rows = [];   // [game, L, m, w, y]
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

  const X = rows.map(([, L, m, w]) => [1, L, L * m, m, w]);
  const Y = rows.map(r => r[4]);
  const G = rows.map(r => r[0]);
  const N = rows.length;
  const sig = z => 1 / (1 + Math.exp(-z));
  const ll = (beta, idx) => {
    let s = 0;
    for (const i of idx) {
      let z = 0; for (let k = 0; k < NUM_FEATURES; k++) z += beta[k] * X[i][k];
      const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(z)));
      s += Y[i] ? Math.log(p) : Math.log(1 - p);
    }
    return -s / idx.length;
  };
  // Newton–Raphson on the logistic log-likelihood: concave, a handful of steps.
  const fit = (idx) => {
    const beta = new Array(NUM_FEATURES).fill(0);
    for (let it = 0; it < 50; it++) {
      const g = new Array(NUM_FEATURES).fill(0);
      const H = Array.from({ length: NUM_FEATURES }, () => new Array(NUM_FEATURES).fill(0));
      for (const i of idx) {
        let z = 0; for (let k = 0; k < NUM_FEATURES; k++) z += beta[k] * X[i][k];
        const p = sig(z), r = Y[i] - p, wgt = p * (1 - p);
        for (let a = 0; a < NUM_FEATURES; a++) {
          g[a] += r * X[i][a];
          for (let b = 0; b < NUM_FEATURES; b++) H[a][b] += wgt * X[i][a] * X[i][b];
        }
      }
      const step = solve(H, g);
      let move = 0;
      for (let k = 0; k < NUM_FEATURES; k++) { beta[k] += step[k]; move += Math.abs(step[k]); }
      if (move < 1e-10) break;
    }
    return beta;
  };
  // Stockfish's own full-information estimate, as a win probability: its
  // expected score, which spends the draw mass half on each side.
  const sfLoss = (idx) => {
    let s = 0;
    for (const i of idx) { const e = sig(X[i][1]); s += Y[i] ? Math.log(e) : Math.log(1 - e); }
    return -s / idx.length;
  };
  const base = (idx) => { const p = idx.reduce((a, i) => a + Y[i], 0) / idx.length; return -(p * Math.log(p) + (1 - p) * Math.log(1 - p)); };

  console.log(`\n${N} positions from ${new Set(G).size} decisive games, depth ${depth}`);
  console.log(`=== ${FOLDS}-fold CV by game, log-loss of the game's result (nats; lower better) ===`);
  let accBase = 0, accSf = 0, accFit = 0, n = 0;
  for (let f = 0; f < FOLDS; f++) {
    const tr = [], te = [];
    for (let i = 0; i < N; i++) (G[i] % FOLDS === f ? te : tr).push(i);
    const beta = fit(tr);
    const a = base(te), b = sfLoss(te), c = ll(beta, te);
    console.log(`  fold ${f}: n=${te.length}  base rate ${a.toFixed(4)}  Stockfish ${b.toFixed(4)}  fitted ${c.toFixed(4)}`);
    accBase += a * te.length; accSf += b * te.length; accFit += c * te.length; n += te.length;
  }
  console.log(`  POOLED: base rate ${(accBase / n).toFixed(4)}  Stockfish ${(accSf / n).toFixed(4)}  fitted ${(accFit / n).toFixed(4)}`);

  const all = Array.from({ length: N }, (_, i) => i);
  const beta = fit(all);
  console.log(`\n  fitted on all: β = [${beta.map(x => x.toFixed(4)).join(', ')}]  (1, L, L·m, m, white)`);
  // Calibration: Stockfish's expected score vs how often the side to move won.
  console.log('\n  Stockfish expected score  →  fog win rate (observed)  fitted (at the bin\'s mean)');
  const bins = [0, 0.02, 0.1, 0.3, 0.45, 0.55, 0.7, 0.9, 0.98, 1.0001];
  for (let b = 0; b + 1 < bins.length; b++) {
    const ix = all.filter(i => { const e = sig(X[i][1]); return e >= bins[b] && e < bins[b + 1]; });
    if (!ix.length) continue;
    const obs = ix.reduce((a, i) => a + Y[i], 0) / ix.length;
    let pf = 0; for (const i of ix) { let z = 0; for (let k = 0; k < NUM_FEATURES; k++) z += beta[k] * X[i][k]; pf += sig(z); }
    console.log(`    ${bins[b].toFixed(2)}–${Math.min(1, bins[b + 1]).toFixed(2)}   n=${String(ix.length).padStart(6)}   ${(100 * obs).toFixed(1).padStart(5)}%   ${(100 * pf / ix.length).toFixed(1).padStart(5)}%`);
  }

  if (argv.includes('--write')) {
    if (!(accFit < accSf)) { console.error('\nNot writing: the fitted model does not beat Stockfish held out.'); process.exit(1); }
    const file = join(HERE, '..', 'src', 'fogValueModel.js');
    writeFileSync(file, `// GENERATED by scripts/fit-fog-value.mjs --write. Do not edit by hand.
//
// P(side to move wins | Stockfish's expected score e at depth ${depth}, material m/78,
// white to move) = σ(β₀ + β₁·L + β₂·L·m + β₃·m + β₄·w), L = logit(clip(e, ${CLIP})).
// Fitted ${new Date().toISOString().slice(0, 10)} on ${N} positions from ${new Set(G).size} decisive games
// (${sessions.split('/').pop()}). Held-out log-loss of the result
// ${(accFit / n).toFixed(4)} nats, against ${(accSf / n).toFixed(4)} for Stockfish's own expected score
// and ${(accBase / n).toFixed(4)} for the base rate (${FOLDS}-fold CV by game).

export const FOG_VALUE_MODEL = {
  depth: ${depth},
  clip: ${CLIP},
  beta: [${beta.map(x => x.toFixed(6)).join(', ')}],
};
`);
    console.log(`\nWrote ${file}.`);
  }
  process.exit(0);
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
  const { multiPV, setCacheDir } = await import('../src/stockfish.js');
  // A cache of its own: these evaluations are not search leaves.
  const dir = join(HERE, '..', 'vendor', 'stockfish', 'fog-value');
  mkdirSync(dir, { recursive: true });
  setCacheDir(dir);
  const { games } = loadCorpus(workerData.sessions, {});
  const kingSq = (b, c) => Object.keys(b).find(s => b[s]?.ownerId === c && b[s].type === 'king');
  const logit = e => { const x = Math.min(1 - CLIP, Math.max(CLIP, e)); return Math.log(x / (1 - x)); };

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
            let m = 0; for (const s of Object.keys(st.board)) { const p = st.board[s]; if (p) m += MATERIAL[p.type] ?? 0; }
            out.push([gi, logit((W + D / 2) / 1000), m / 78, me === 'white' ? 1 : 0, winner === me ? 1 : 0]);
          }
        }
        try { st = FogChess.applyActions(st, [pa]); } catch { break; }
      }
    }
    parentPort.postMessage(out);
  });
}
