// ---------------------------------------------------------------------------
// Train the distilled leaf evaluator (src/valueNet.js) on collect-evals output.
//
//   node scripts/train-valuenet.mjs --data train-d7.jsonl [--hidden 32]
//        [--epochs 8] [--lr 0.001] [--clip 300] [--out valuenet.json]
//
// ONE COLLECTED POSITION BECOMES ~30 SAMPLES, not one. A MultiPV sweep scores
// every legal move of a parent, and each of those scores IS the value of the
// child it leads to — so applying the move gives a labelled child position, and
// a 70k-position dump becomes a ~2M-sample training set. Training on parents
// alone (label = best move's score) would throw away 97% of what the engine
// already told us.
//
// SIGNS, which are the easy thing to get backwards and the hard thing to notice:
// multiPV returns cp from the PARENT's mover. After the move it is the
// opponent's turn, so the child's value from ITS OWN side to move is −cp. The
// net is trained in that convention throughout, matching what evalBoard returns,
// so the leaf evaluator negates once when reading it back.
//
// Targets are clipped to ±`clip` (default 300, the same LEAF_CLAMP the search
// uses) so a handful of mate scores do not own the gradient.
// ---------------------------------------------------------------------------

import { createReadStream, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { fromFEN } from '../src/fen.js';
import { ValueNet, features, INPUTS } from '../src/valueNet.js';
import { makeArgReader } from '../src/cli.js';

const arg = makeArgReader(process.argv.slice(2));
const DATA = arg('data', 'train-d7.jsonl');
const HIDDEN = Number(arg('hidden', '32'));
const EPOCHS = Number(arg('epochs', '8'));
const LR0 = Number(arg('lr', '0.001'));
const CLIP = Number(arg('clip', '300'));
const OUT = arg('out', 'valuenet.json');
const VAL_FRAC = Number(arg('val', '0.05'));

if (!existsSync(DATA)) { console.error(`no such data file: ${DATA}`); process.exit(1); }

// Apply a UCI move to a board object. Deliberately REFUSES castling and en
// passant rather than guessing: they need state this row does not carry, a
// mis-applied one silently mislabels a sample, and they are a small enough share
// that dropping them costs less than a subtle corruption of the target.
function applyUci(board, uci) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4), promo = uci[4];
  const p = board[from];
  if (!p) return null;
  if (p.type === 'king' && Math.abs(from.charCodeAt(0) - to.charCodeAt(0)) > 1) return null; // castle
  if (p.type === 'pawn' && from[0] !== to[0] && !board[to]) return null;                     // en passant
  const next = {};
  for (const sq of Object.keys(board)) if (sq !== from && sq !== to) next[sq] = board[sq];
  const type = promo ? ({ q: 'queen', r: 'rook', b: 'bishop', n: 'knight' }[promo] ?? p.type) : p.type;
  next[to] = { ...p, type, position: to };
  return next;
}

const clip = x => Math.max(-CLIP, Math.min(CLIP, x));
const flipSide = s => (s === 'w' ? 'b' : 'w');

console.log(`reading ${DATA} …`);
const X = [];            // feature index arrays
const Y = [];            // targets, centipawns from the child's side to move
let positions = 0, skipped = 0;

const rl = createInterface({ input: createReadStream(DATA), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line) continue;
  let row; try { row = JSON.parse(line); } catch { continue; }
  let board; try { ({ board } = fromFEN(row.fen)); } catch { continue; }
  positions++;
  const childSide = flipSide(row.side);
  for (const [uci, cp] of row.moves) {
    const nb = applyUci(board, uci);
    if (!nb) { skipped++; continue; }
    const idx = features(nb, childSide, []);
    if (!idx.length) { skipped++; continue; }
    X.push(Int32Array.from(idx));
    Y.push(clip(-cp) / CLIP);          // NORMALISED; see the sign note and ValueNet.scale
  }
}
console.log(`${positions} positions -> ${X.length} samples (${skipped} moves skipped: castling/en-passant/unapplicable)`);
if (!X.length) { console.error('nothing to train on'); process.exit(1); }

// A held-out slice, taken by POSITION ORDER rather than at random: samples from
// one parent are near-duplicates of each other, so a random split would put a
// position's siblings on both sides and report a validation error that is really
// a training error.
const cut = Math.floor(X.length * (1 - VAL_FRAC));
const nTrain = cut, nVal = X.length - cut;
console.log(`train ${nTrain}, validation ${nVal} (tail slice, so no sibling leakage)`);

const net = new ValueNet(HIDDEN, CLIP);   // net multiplies by CLIP on the way out
let seed = 12345;
const rng = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
net.randomize(rng);

const { h, w1, b1, w2 } = net;
const acc = new Float32Array(h);
const order = new Int32Array(nTrain);
for (let i = 0; i < nTrain; i++) order[i] = i;

const evalSet = (lo, hi) => {
  let se = 0, sa = 0;
  for (let i = lo; i < hi; i++) {
    const e = (net.rawIndices(X[i]) - Y[i]) * CLIP;   // report in centipawns
    se += e * e; sa += Math.abs(e);
  }
  const n = hi - lo;
  return { rmse: Math.sqrt(se / n), mae: sa / n };
};

console.log(`\ntraining ${INPUTS}->${HIDDEN}->1, ${EPOCHS} epochs, lr ${LR0}`);
for (let ep = 0; ep < EPOCHS; ep++) {
  for (let i = nTrain - 1; i > 0; i--) {         // shuffle
    const j = Math.floor(rng() * (i + 1));
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  const lr = LR0 / (1 + ep * 0.5);               // simple decay
  const t0 = Date.now();
  for (let s = 0; s < nTrain; s++) {
    const i = order[s], idx = X[i];
    // forward
    acc.set(b1);
    for (let k = 0; k < idx.length; k++) {
      const base = idx[k] * h;
      for (let j = 0; j < h; j++) acc[j] += w1[base + j];
    }
    let v = net.b2;
    for (let j = 0; j < h; j++) if (acc[j] > 0) v += w2[j] * acc[j];   // normalised units
    // backward: d(½e²)/dv = e
    const e = v - Y[i];
    const g = e * lr;
    net.b2 -= g;
    for (let j = 0; j < h; j++) {
      if (acc[j] <= 0) continue;                 // ReLU gate
      const gh = g * w2[j];
      w2[j] -= g * acc[j];
      b1[j] -= gh;
      for (let k = 0; k < idx.length; k++) w1[idx[k] * h + j] -= gh;
    }
  }
  const tr = evalSet(0, Math.min(nTrain, 50000)), va = evalSet(cut, X.length);
  console.log(`  epoch ${ep + 1}/${EPOCHS}  lr ${lr.toFixed(5)}  ` +
    `train rmse ${tr.rmse.toFixed(1)} mae ${tr.mae.toFixed(1)}  |  ` +
    `VAL rmse ${va.rmse.toFixed(1)} mae ${va.mae.toFixed(1)}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}

// The baseline worth beating is not zero error, it is what predicting a constant
// achieves — with clipped cp targets that is already a low RMSE, and a net that
// merely matches it has learned nothing.
let mean = 0; for (let i = 0; i < nTrain; i++) mean += Y[i];
mean /= nTrain;
let cse = 0; for (let i = cut; i < X.length; i++) cse += ((mean - Y[i]) * CLIP) ** 2;
console.log(`\nconstant-prediction baseline on the same validation slice: rmse ${Math.sqrt(cse / nVal).toFixed(1)} cp (predicting ${(mean * CLIP).toFixed(1)} cp)`);

writeFileSync(OUT, JSON.stringify(net.toJSON()));
console.log(`net -> ${OUT}`);
