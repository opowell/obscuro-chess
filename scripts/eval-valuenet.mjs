// ---------------------------------------------------------------------------
// How good is the distilled evaluator, in the terms that decide play?
//
//   node scripts/eval-valuenet.mjs --net valuenet.json --data holdout.jsonl
//
// RMSE IS THE WRONG HEADLINE. The search never uses a leaf value on its own — it
// compares siblings. A net that is 150 cp pessimistic about every child of a node
// ranks them exactly as its teacher does and plays identically; a net with half
// the RMSE that shuffles the top two plays worse. So this reports what the tree
// actually consumes:
//
//   • TOP-1 AGREEMENT — does the net's best child match the teacher's best child.
//     This is the number that most nearly predicts the move.
//   • TOP-1 REGRET — when it disagrees, how much does the teacher think the net's
//     pick costs. Disagreeing about two moves the teacher scores equally is free.
//   • SPEARMAN over each node's children, the whole ordering rather than its head.
//
// Compared against the trivial baselines, because "56% top-1" means nothing until
// you know what material-only and a random pick score.
//
// THE MOVE ORDER IS A LEAK, and it fooled the first version of this script.
// MultiPV returns its lines SORTED BEST-FIRST, so the teacher's answer is the
// row order. Any evaluator with ties — material scores 0 for every non-capture —
// then has `indexOf(max)` hand it index 0, i.e. the teacher's own best move, and
// material duly "agreed" 38.8% of the time while its Spearman correlation with
// the teacher was NEGATIVE (-0.69). A baseline that is anti-correlated cannot
// pick good moves; the agreement was tie-breaking into a sorted list. So the
// children are shuffled per node before anything looks at them.
// ---------------------------------------------------------------------------

import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { fromFEN } from '../src/fen.js';
import { ValueNet, features } from '../src/valueNet.js';
import { makeArgReader } from '../src/cli.js';

const arg = makeArgReader(process.argv.slice(2));
const NET = arg('net', 'valuenet.json');
const DATA = arg('data', 'holdout-d7.jsonl');
const LIMIT = Number(arg('limit', '100000'));
for (const f of [NET, DATA]) if (!existsSync(f)) { console.error(`missing: ${f}`); process.exit(1); }

const net = ValueNet.fromJSON(JSON.parse(readFileSync(NET, 'utf8')));
const PIECE_CP = { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900, king: 0 };

function applyUci(board, uci) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4), promo = uci[4];
  const p = board[from];
  if (!p) return null;
  if (p.type === 'king' && Math.abs(from.charCodeAt(0) - to.charCodeAt(0)) > 1) return null;
  if (p.type === 'pawn' && from[0] !== to[0] && !board[to]) return null;
  const next = {};
  for (const sq of Object.keys(board)) if (sq !== from && sq !== to) next[sq] = board[sq];
  const type = promo ? ({ q: 'queen', r: 'rook', b: 'bishop', n: 'knight' }[promo] ?? p.type) : p.type;
  next[to] = { ...p, type, position: to };
  return next;
}
// Material only, from `side`'s view — the baseline any evaluator must beat to
// have justified its existence.
function material(board, side) {
  let v = 0;
  for (const sq of Object.keys(board)) {
    const p = board[sq];
    if (!p || p.alive === false) continue;
    const own = (p.ownerId ?? p.color) === (side === 'w' ? 'white' : 'black');
    v += (own ? 1 : -1) * (PIECE_CP[p.type] ?? 0);
  }
  return v;
}
const spearman = (a, b) => {
  const rank = xs => { const o = xs.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(xs.length); o.forEach(([, i], k) => { r[i] = k; }); return r; };
  const ra = rank(a), rb = rank(b), n = a.length;
  let d = 0; for (let i = 0; i < n; i++) d += (ra[i] - rb[i]) ** 2;
  return 1 - (6 * d) / (n * (n * n - 1));
};

let seed = 20260816;
let nodes = 0, netTop = 0, matTop = 0, randTop = 0;
let netRegret = 0, matRegret = 0, sRho = 0, mRho = 0, rhoN = 0;
const rl = createInterface({ input: createReadStream(DATA), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line || nodes >= LIMIT) continue;
  let row; try { row = JSON.parse(line); } catch { continue; }
  let board; try { ({ board } = fromFEN(row.fen)); } catch { continue; }
  const childSide = row.side === 'w' ? 'b' : 'w';
  const teacher = [], mine = [], mat = [];
  // Deterministic shuffle: destroy the best-first ordering that leaks the label,
  // reproducibly, so a rerun of this script scores the same net the same way.
  const moves = row.moves.slice();
  for (let i = moves.length - 1; i > 0; i--) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const j = seed % (i + 1);
    const t = moves[i]; moves[i] = moves[j]; moves[j] = t;
  }
  for (const [uci, cp] of moves) {
    const nb = applyUci(board, uci);
    if (!nb) continue;
    teacher.push(cp);                                   // value to the mover
    mine.push(-net.evalBoard(nb, childSide));           // net, same convention
    mat.push(-material(nb, childSide));
  }
  if (teacher.length < 2) continue;
  nodes++;
  const best = teacher.indexOf(Math.max(...teacher));
  const pick = a => a.indexOf(Math.max(...a));
  const pn = pick(mine), pm = pick(mat);
  if (pn === best) netTop++;
  if (pm === best) matTop++;
  randTop += 1 / teacher.length;
  // Regret clipped to the search's own LEAF_CLAMP: raw teacher values encode
  // mate as ±100000, and a handful of those would otherwise BE the mean.
  const cl = x => Math.max(-300, Math.min(300, x));
  netRegret += cl(teacher[best]) - cl(teacher[pn]);
  matRegret += cl(teacher[best]) - cl(teacher[pm]);
  if (teacher.length > 2) { sRho += spearman(teacher, mine); mRho += spearman(teacher, mat); rhoN++; }
}

const pct = x => (100 * x / nodes).toFixed(1) + '%';
console.log(`${nodes} nodes from ${DATA}\n`);
console.log('                       top-1 agreement   mean regret (cp)   spearman');
console.log(`  distilled net        ${pct(netTop).padStart(10)}      ${(netRegret / nodes).toFixed(1).padStart(10)}       ${(sRho / rhoN).toFixed(3)}`);
console.log(`  material only        ${pct(matTop).padStart(10)}      ${(matRegret / nodes).toFixed(1).padStart(10)}       ${(mRho / rhoN).toFixed(3)}`);
console.log(`  random child         ${pct(randTop).padStart(10)}`);
console.log(`\nregret is what the TEACHER thinks the pick costs; 0 = picked a move it scores equally.`);
