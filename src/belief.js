// ---------------------------------------------------------------------------
// Fog-of-war belief tracking ("particle" / information-set model)
//
// Under fog of war the agent is handed a board with every unseen enemy piece
// deleted, so a naive search treats hidden pieces as if they did not exist and
// happily walks into their attacks. This module instead maintains the
// *information set*: the set of board states consistent with everything the
// agent has legitimately observed (the common-knowledge starting position, its
// own moves, and what its pieces have seen each turn).
//
// We represent the belief compactly as, per still-living enemy piece, the set
// of squares it could currently occupy. Each turn we:
//   1. propagate  — every unseen enemy piece may have moved one ply through the
//                   fog, so we expand its possible squares by one move of
//                   reachability (this is what lets us realise a pawn last
//                   "known" on e2 could now be on e4 attacking f5);
//   2. collapse   — squares we can now see pin or rule out possibilities;
//   3. sample     — draw a handful of concrete full boards ("particles") from
//                   the belief for the search to evaluate against.
//
// This is a deliberately lightweight, game-specific analogue of the particle /
// knowledge-limited subgame-solving approach from Zhang & Sandholm 2021
// ("Subgame solving without common knowledge"): we keep no equilibrium
// machinery, only a calibrated cloud of plausible worlds plus pessimistic
// evaluation in the agent.
// ---------------------------------------------------------------------------

import { FILES, fileIndex, rankOf, squareAt, getVisibleSquares, isAttackedBy } from './board.js';
import { pseudoLegalForUnit } from './moves.js';

const ALL_SQUARES = [];
for (const f of FILES) for (let r = 1; r <= 8; r++) ALL_SQUARES.push(f + r);

/**
 * Squares a piece of this type/colour could legally OCCUPY, ignoring history.
 *
 * Only pawns are constrained, and both constraints are hard: a pawn can never
 * stand on its own first rank (it starts on the second and only moves forward),
 * and one that reached the far rank promoted and is no longer a pawn. Every
 * other type can be anywhere.
 *
 * Exported because the same rule is the cheapest possible sanity check on any
 * imagined board — see `impossiblePlacement`.
 */
export function possibleSquaresFor(type, color) {
  if (type !== 'pawn') return ALL_SQUARES;
  const own = color === 'white' ? '1' : '8';
  const far = color === 'white' ? '8' : '1';
  return ALL_SQUARES.filter(sq => sq[1] !== own && sq[1] !== far);
}

/**
 * Is this board one that could never occur? Returns a reason, or null.
 *
 * Cheap enough to run behind a debug flag at every world producer, which is how
 * the "white pawn on d1" worlds were tracked back to their source. Illegal
 * positions are not a cosmetic problem: Stockfish answers them with zero MultiPV
 * lines, and the leaf evaluator then quietly substitutes its static fallback.
 */
export function impossiblePlacement(board) {
  const kings = { white: 0, black: 0 };
  const pawns = { white: 0, black: 0 };
  for (const [sq, p] of Object.entries(board)) {
    if (!p) continue;
    if (p.type === 'king') kings[p.ownerId]++;
    if (p.type === 'pawn') {
      pawns[p.ownerId]++;
      const own = p.ownerId === 'white' ? '1' : '8';
      const far = p.ownerId === 'white' ? '8' : '1';
      if (sq[1] === own) return `${p.ownerId} pawn on its own first rank (${sq})`;
      if (sq[1] === far) return `${p.ownerId} pawn on the promotion rank (${sq})`;
    }
  }
  for (const c of ['white', 'black']) {
    if (kings[c] !== 1) return `${c} has ${kings[c]} kings`;
    if (pawns[c] > 8) return `${c} has ${pawns[c]} pawns`;
  }
  return null;
}

// Empty castling rights — belief reachability never castles, but the king move
// generator dereferences castlingRights[ownerId], so supply a safe stub.
const NO_CASTLING = { white: { kingSide: false, queenSide: false }, black: { kingSide: false, queenSide: false } };

// Cap a piece's possible-square set. Larger = closer to the exact information
// set P (fewer valid placements truncated away), at a modest sampling cost. Kept
// below the 64-square max so a wildly-uncertain piece late in the game doesn't
// blow up the candidate lists.
//
// Exported (with the rest of this section) so src/settings.js
// can list them alongside every other fog-chess default; the values and their
// reasoning stay here, next to the sampling code they tune.
export const MAX_POSSIBLE = 48;
export const THREAT_BIAS = 3;   // how strongly to over-sample placements that attack our pieces
// At most this many invisible pieces per particle may be placed on a square that
// attacks one of our pieces. Real positions rarely have the whole hidden army
// bearing down at once; without this cap, threat-biased sampling hallucinates
// coordinated mating attacks and the AI huddles instead of saving real material.
// (THREAT_BIAS is kept modest for the same reason — over-weighting phantom
// attackers made the AI play passively, shuffling its king and pawns rather than
// developing; we still surface real threats, just without imagining a swarm.)
export const MAX_LURKERS = 2;

// Relative likelihood that a piece of each type is the one that captured on a
// forced (known-capture) square — chess recaptures strongly favour the least
// valuable capturer. Used by the forced-square inference in sample().
export const RECAPTURE_TYPE_WEIGHT = { pawn: 9, knight: 3, bishop: 3, rook: 1.5, queen: 1, king: 0.5 };
// How many resample attempts sample() gets per requested particle, and how
// long (as a multiple of n) it keeps rejecting phantom self-checks before
// giving up and accepting anything (see the two uses below).
export const MAX_ATTEMPTS_PER_PARTICLE = 6;
export const PHANTOM_CHECK_REJECT_WINDOW = 4;

function opp(color) { return color === 'white' ? 'black' : 'white'; }

// The opponent's starting line-up, with ids matching FogChess's initialBoard so
// that sightings (which carry the real piece id) collapse the right entry.
function startingPieces(color) {
  const br = color === 'white' ? 1 : 8;
  const pr = color === 'white' ? 2 : 7;
  const p  = color === 'white' ? 'w' : 'b';
  const back = [
    ['R', 'rook',   'a', ''], ['N', 'knight', 'b', ''], ['B', 'bishop', 'c', ''],
    ['Q', 'queen',  'd', ''], ['K', 'king',   'e', ''], ['B', 'bishop', 'f', '2'],
    ['N', 'knight', 'g', '2'], ['R', 'rook',  'h', '2'],
  ];
  const list = [];
  for (const [sym, type, file, suf] of back) list.push({ id: p + sym + suf, type, position: file + br });
  for (const f of FILES) list.push({ id: p + 'P' + f, type: 'pawn', position: f + pr });
  return list;
}

// Squares this piece type could move *to and rest on* in one ply, given the
// currently-known board. `board` already has hidden squares empty (so sliders
// pass freely through the fog) and known pieces present (so they block). A
// resting square must be hidden — were the piece on a visible square we would
// simply see it. Pawns additionally treat both forward diagonals as possible
// captures into the fog.
function reachableSquares(board, type, color, sq, hidden) {
  const out = [];
  if (type === 'pawn') {
    const fi = fileIndex(sq), r = rankOf(sq);
    const dir = color === 'white' ? 1 : -1;
    const startRank = color === 'white' ? 2 : 7;
    const one = squareAt(fi, r + dir);
    if (one && !board[one]) {
      if (hidden.has(one)) out.push(one);
      if (r === startRank) {
        const two = squareAt(fi, r + 2 * dir);
        if (two && !board[two] && hidden.has(two)) out.push(two);
      }
    }
    for (const dfi of [-1, 1]) {
      const cap = squareAt(fi + dfi, r + dir);
      if (cap && hidden.has(cap)) out.push(cap); // could have captured into fog
    }
    return out;
  }
  const unit = { id: '__belief__', ownerId: color, type, position: sq };
  for (const a of pseudoLegalForUnit(board, unit, { castlingRights: NO_CASTLING, enPassantTarget: null }, true)) {
    if (a.to && hidden.has(a.to)) out.push(a.to);
  }
  // Castling is not generated above (rights are stubbed off), but a hidden king
  // or rook may really have castled. Add the castle destination squares so a
  // piece's possible-set stays a SUPERSET of where it can truly be — the exact
  // belief's re-acquisition relies on that superset property.
  const home = color === 'white' ? 1 : 8;
  const castleTo = { king: { ['e' + home]: ['g' + home, 'c' + home] },
                     rook: { ['h' + home]: ['f' + home], ['a' + home]: ['d' + home] } }[type]?.[sq];
  if (castleTo) for (const c of castleTo) if (hidden.has(c) && !board[c]) out.push(c);
  return out;
}

function chebyshev(a, b) {
  return Math.max(Math.abs(fileIndex(a) - fileIndex(b)), Math.abs(rankOf(a) - rankOf(b)));
}

export class Belief {
  constructor(aiColor) {
    this.aiColor = aiColor;
    this.oppColor = opp(aiColor);
    // id -> { id, type, possible:Set<square>, anchor:square, alive:bool }
    this.pieces = new Map();
    for (const sp of startingPieces(this.oppColor)) {
      this.pieces.set(sp.id, { id: sp.id, type: sp.type, possible: new Set([sp.position]), anchor: sp.position, alive: true });
    }
    this.firstTurnDone = false;
    this.oppPlies = 0;             // number of moves the opponent has made so far
    this.ownSnapshot = null;       // id -> square of our pieces after our last move
    this.forcedEnemy = new Set();  // squares we know hold an enemy right now (just captured a piece of ours)
  }

  hiddenSquares(board) {
    const visible = getVisibleSquares(board, this.aiColor);
    return new Set(ALL_SQUARES.filter(sq => !visible.has(sq)));
  }

  // 1. Propagation: every unseen enemy piece may have advanced one ply.
  expandOnePly(board) {
    const hidden = this.hiddenSquares(board);
    for (const pc of this.pieces.values()) {
      if (!pc.alive) continue;
      const next = new Set(pc.possible); // staying put is always possible
      for (const sq of pc.possible) {
        for (const dest of reachableSquares(board, pc.type, this.oppColor, sq, hidden)) next.add(dest);
      }
      if (next.size > MAX_POSSIBLE) {
        pc.possible = new Set([...next].sort((a, b) => chebyshev(a, pc.anchor) - chebyshev(b, pc.anchor)).slice(0, MAX_POSSIBLE));
        // The set is no longer a guaranteed superset of the truth. The exact
        // belief's re-acquisition must not trust a truncated piece.
        pc.truncated = true;
      } else {
        pc.possible = next;
      }
      // A hidden pawn whose set reaches the last rank may have PROMOTED, which
      // per-piece type tracking cannot represent — flag it for the same reason.
      if (pc.type === 'pawn') {
        const last = this.oppColor === 'white' ? '8' : '1';
        for (const sq of pc.possible) if (sq[1] === last) { pc.truncated = true; break; }
      }
    }
  }

  // Detect pieces of ours captured since our last move; the capturer must sit on
  // the victim's square at the start of this turn (opponent has moved once).
  computeForcedSquares(board) {
    this.forcedEnemy = new Set();
    if (!this.ownSnapshot) return;
    const hidden = this.hiddenSquares(board);
    const present = new Set();
    for (const sq of Object.keys(board)) {
      const pc = board[sq];
      if (pc && pc.ownerId === this.aiColor) present.add(pc.id);
    }
    for (const [id, sq] of this.ownSnapshot) {
      if (!present.has(id) && hidden.has(sq)) this.forcedEnemy.add(sq);
    }
  }

  // 2. Collapse: reconcile the belief with what we can actually see this turn.
  collapse(board) {
    const visible = getVisibleSquares(board, this.aiColor);
    const seen = new Set();
    for (const sq of Object.keys(board)) {
      const pc = board[sq];
      if (!pc || pc.ownerId !== this.oppColor) continue;
      seen.add(pc.id);
      let entry = this.pieces.get(pc.id);
      if (!entry) { entry = { id: pc.id, type: pc.type, possible: new Set(), anchor: sq, alive: true }; this.pieces.set(pc.id, entry); }
      entry.type = pc.type;          // track promotions
      entry.alive = true;
      entry.anchor = sq;
      entry.possible = new Set([sq]); // pinned: we see exactly where it is
    }
    // An unseen piece is on no visible square (else we'd see it).
    for (const pc of this.pieces.values()) {
      if (!pc.alive || seen.has(pc.id)) continue;
      for (const sq of [...pc.possible]) if (visible.has(sq)) pc.possible.delete(sq);
      if (pc.possible.size === 0) {
        // Belief contradiction (over-aggressive pruning); fall back to "somewhere
        // hidden" — but only where this piece TYPE could actually stand.
        //
        // "Anywhere hidden" used to mean literally every square, which put enemy
        // pawns on their own first rank. Nothing downstream noticed: the exact
        // belief's tryReacquire trusts these sets, built worlds from them, and
        // handed Stockfish positions like
        //   3rkb1r/pppqpp2/6p1/8/1P1P3P/B4P2/PP1R1P2/3PKB2   (white pawn on d1)
        // which are ILLEGAL — and an illegal FEN makes the engine return zero
        // MultiPV lines, so every child of that node silently fell through to the
        // static evaluator (see FOG-AI-FIX-PLAN.md, 2026-08-03).
        //
        // Excluding those squares keeps the set a valid SUPERSET of the truth,
        // because the truth could never be there: a pawn cannot occupy its own
        // first rank, and one that reached the far rank would have promoted and
        // stopped being a pawn (which `truncated` above already flags).
        pc.possible = new Set(possibleSquaresFor(pc.type, this.oppColor)
          .filter(sq => !visible.has(sq)));
      }
    }
  }

  // Run the full per-turn update before the agent searches. `turnKey` makes the
  // call idempotent within one decision: the agent may sample worlds more than
  // once per move (the main search plus e.g. a king-safety re-sample), and each
  // extra call used to advance the belief another phantom opponent ply, leaving
  // it artificially diffuse. Same key → the belief is already up to date.
  beginTurn(board, turnKey = null) {
    if (turnKey != null) {
      if (this._lastTurnKey === turnKey) return;
      this._lastTurnKey = turnKey;
    }
    if (!this.firstTurnDone) {
      // Our very first move. If we are black, white has already moved once.
      if (this.aiColor === 'black') { this.expandOnePly(board); this.oppPlies = 1; }
      this.firstTurnDone = true;
    } else {
      this.expandOnePly(board); // opponent has moved exactly once since our last turn
      this.oppPlies++;
    }
    this.computeForcedSquares(board);
    this.collapse(board);
  }

  // Record the move we are about to play so next turn we can (a) drop captured
  // enemies and (b) detect our own losses.
  commitOurMove(action, board) {
    if (action) {
      if (action.isCapture && action.targetId) {
        const victim = this.pieces.get(action.targetId);
        if (victim) victim.alive = false;
      }
    }
    // Snapshot our pieces as they will stand after the move.
    const snap = new Map();
    for (const sq of Object.keys(board)) {
      const pc = board[sq];
      if (pc && pc.ownerId === this.aiColor) snap.set(pc.id, pc.position);
    }
    if (action && action.from && action.to) {
      const moved = board[action.from];
      if (moved && moved.ownerId === this.aiColor) { snap.delete(moved.id); snap.set(moved.id, action.to); }
    }
    this.ownSnapshot = snap;
  }

  /**
   * Sample up to `n` concrete full boards consistent with the current belief.
   * Each particle is the observed board plus a plausible placement of every
   * unseen, still-living enemy piece. Most-constrained pieces are placed first;
   * squares nearer a piece's anchor (last sighting / start) are favoured, but
   * far squares are still drawn so genuinely dangerous worlds appear.
   */
  sample(board, n, rng = Math.random) {
    const visible = getVisibleSquares(board, this.aiColor);
    const seen = new Set();
    for (const sq of Object.keys(board)) {
      const pc = board[sq];
      if (pc && pc.ownerId === this.oppColor) seen.add(pc.id);
    }

    // Whether a piece of a given type, placed on a given square, would attack one
    // of our pieces. Memoised since it only depends on the fixed observed board.
    const threatCache = new Map();
    const threatens = (type, sq) => {
      const key = type + sq;
      let v = threatCache.get(key);
      if (v === undefined) { v = attacksFriendly(board, type, this.oppColor, sq, this.aiColor); threatCache.set(key, v); }
      return v;
    };
    const canThreaten = pc => [...pc.possible].some(sq => !visible.has(sq) && threatens(pc.type, sq));

    const unseen = [...this.pieces.values()].filter(pc => pc.alive && !seen.has(pc.id));

    // Pieces that could move to a square attacking one of our pieces. The scarce
    // move budget is spent preferentially (but not exclusively) on these, so a
    // real lurking threat is surfaced without imagining the whole army developed.
    const threatCapable = new Set(unseen.filter(canThreaten).map(p => p.id));

    // Reserve every hidden piece's home square so a deviating piece never squats
    // on another's anchor and forces it out too (which used to cascade a single
    // move into several pieces leaving home).
    const anchors = new Set(unseen.map(pc => pc.anchor));

    // Our own king square on the observed board (always visible to us) — used to
    // reject phantom self-checks below.
    let myKingSq = null;
    for (const sq of Object.keys(board)) {
      const pc = board[sq];
      if (pc && pc.ownerId === this.aiColor && pc.type === 'king') { myKingSq = sq; break; }
    }

    const particles = [];
    const keys = new Set();
    // More attempts per requested world so a larger belief (now that we sample
    // more worlds) still yields the distinct particles it asks for rather than
    // giving up early — a fuller, more representative draw from P.
    const maxAttempts = n * MAX_ATTEMPTS_PER_PARTICLE;
    for (let attempt = 0; attempt < maxAttempts && particles.length < n; attempt++) {
      const pb = { ...board };
      const used = new Set();
      const placedHidden = [];     // [{ type, sq }] every hidden placement, for the check test
      let lurkers = 0;             // invisible attackers placed so far in this particle
      // The opponent can only have moved as many pieces as it has made moves, so
      // most of its army is still at home. Pieces placed beyond this budget are
      // forced back to their anchor (start / last-seen) square.
      let moveBudget = Math.min(this.oppPlies, unseen.length);

      // Squares we KNOW hold an enemy right now (it just captured a piece of ours
      // there) are placed FIRST, preferring a real unseen piece that could have
      // reached the square — the recapture inference. Scattering these pieces
      // elsewhere and back-filling a phantom (the old behaviour) both duplicated
      // the capturer and randomised its type, inflating belief variance exactly
      // where the position is sharpest.
      const forcedTaken = new Set();
      for (const fsq of this.forcedEnemy) {
        if (pb[fsq]) continue;
        const plausible = unseen.filter(pc => !forcedTaken.has(pc.id) && pc.possible.has(fsq));
        if (plausible.length > 0) {
          // Weight the candidate capturers by proximity AND (inverse) piece
          // value: real recaptures overwhelmingly use the least valuable piece
          // available. Without the value term the sampler put queens/rooks/kings
          // on the capture square far too often; those give phantom CHECK from a
          // square the mover can't see, and a belief where a quarter of worlds
          // start in check makes every quiet move price like death — which is
          // how the search talked itself into actual king-walk blunders.
          const weights = plausible.map(pc =>
            (RECAPTURE_TYPE_WEIGHT[pc.type] ?? 1) / (1 + chebyshev(fsq, pc.anchor)));
          const pc = weightedPick(plausible, weights, rng);
          pb[fsq] = { id: pc.id, ownerId: this.oppColor, type: pc.type, position: fsq, alive: true };
          forcedTaken.add(pc.id);
          if (fsq !== pc.anchor) moveBudget--; // the capture spent one opponent move
        } else {
          pb[fsq] = { id: '__capt__' + fsq, ownerId: this.oppColor, type: 'queen', position: fsq, alive: true };
        }
        used.add(fsq);
        placedHidden.push({ type: pb[fsq].type, sq: fsq });
      }

      // Order the remaining pieces for this particle: threat-capable ones tend to
      // come first (so they win the budget) but with jitter, so threats appear in
      // many particles rather than all — the AI stays cautious, not paralysed.
      const order = [...unseen].filter(pc => !forcedTaken.has(pc.id)).sort((a, b) =>
        ((threatCapable.has(a.id) ? -0.7 : 0) + rng()) - ((threatCapable.has(b.id) ? -0.7 : 0) + rng()));
      for (const pc of order) {
        // `possible` legitimately keeps a pawn's promotion-rank squares — the
        // piece really could have gone there — but whatever stands there is a
        // QUEEN, not a pawn, so placing a pawn on it builds a board that could
        // never occur. (`truncated` already flags the piece for this reason, which
        // is what stops the exact belief re-acquiring from it; the particle
        // sampler had no such guard.) An illegal board makes Stockfish return
        // zero MultiPV lines and the leaf evaluator then guesses silently, so
        // drop those squares here rather than emit one.
        const placeable = pc.type === 'pawn'
          ? new Set(possibleSquaresFor('pawn', this.oppColor))
          : null;
        const cands = [...pc.possible].filter(sq =>
          !pb[sq] && !used.has(sq) && !visible.has(sq) && !(anchors.has(sq) && sq !== pc.anchor)
          && (!placeable || placeable.has(sq)));
        if (cands.length === 0) continue; // leave this piece off this particle
        const anchorFree = cands.includes(pc.anchor);
        let sq;
        if (anchorFree && moveBudget <= 0) {
          sq = pc.anchor; // out of moves: this piece must still be home
        } else {
          const biasOk = lurkers < MAX_LURKERS;
          const weights = cands.map(s => {
            const w = 1 / (1 + chebyshev(s, pc.anchor));
            return biasOk && threatens(pc.type, s) ? w * THREAT_BIAS : w; // surface dangerous worlds
          });
          sq = weightedPick(cands, weights, rng);
          if (sq !== pc.anchor) moveBudget--; // spent one of the opponent's moves
        }
        if (threatens(pc.type, sq)) lurkers++;
        pb[sq] = { id: pc.id, ownerId: this.oppColor, type: pc.type, position: sq, alive: true };
        used.add(sq);
        placedHidden.push({ type: pc.type, sq });
      }

      // Reject phantom self-checks: a particle where OUR king is already in check
      // purely because of where we imagined hidden pieces (no capture evidence).
      // Under FoW rules invisible checks are rare — we see every square our king
      // could step to — yet scattering used to put ~a quarter of worlds in check,
      // swinging evals by ±WIN for no reason. Checks delivered from a forced
      // (evidenced) square, or by a visible piece, are kept. If non-check worlds
      // are genuinely scarce, the last attempts accept anything so a cornered
      // belief still yields particles.
      if (myKingSq && attempt < n * PHANTOM_CHECK_REJECT_WINDOW && isAttackedBy(pb, myKingSq, this.oppColor)) {
        const hiddenCheckers = placedHidden.filter(ph =>
          !this.forcedEnemy.has(ph.sq) && attacksSquare(pb, ph.type, this.oppColor, ph.sq, myKingSq));
        const evidencedCheck = isAttackedBy(
          stripSquares(pb, hiddenCheckers.map(ph => ph.sq)), myKingSq, this.oppColor);
        if (hiddenCheckers.length > 0 && !evidencedCheck) continue; // phantom — resample
      }

      const key = boardSignature(pb, this.oppColor);
      if (keys.has(key)) continue;
      keys.add(key);
      particles.push(pb);
    }
    return particles;
  }
}

// True if a `color` piece of `type` on `from` attacks the square `target`, given
// `board` occupancy (blockers respected).
function attacksSquare(board, type, color, from, target) {
  const unit = { id: '__chk__', ownerId: color, type, position: from };
  for (const a of pseudoLegalForUnit(board, unit, { castlingRights: NO_CASTLING, enPassantTarget: null }, true)) {
    if (a.to === target) return true;
  }
  return false;
}

// Copy of `board` with the given squares emptied.
function stripSquares(board, squares) {
  const b = { ...board };
  for (const sq of squares) b[sq] = undefined;
  return b;
}

// True if a `color` piece of `type` placed on `sq` would attack any `targetColor`
// piece on `board` (hidden squares are empty, so attack lines run through the fog).
function attacksFriendly(board, type, color, sq, targetColor) {
  const unit = { id: '__threat__', ownerId: color, type, position: sq };
  for (const a of pseudoLegalForUnit(board, unit, { castlingRights: NO_CASTLING, enPassantTarget: null }, true)) {
    const occ = a.to && board[a.to];
    if (occ && occ.ownerId === targetColor) return true;
  }
  return false;
}

function weightedPick(cands, weights, rng) {
  let total = 0;
  for (const w of weights) total += w;
  let pick = rng() * total;
  for (let i = 0; i < cands.length; i++) { pick -= weights[i]; if (pick <= 0) return cands[i]; }
  return cands[cands.length - 1];
}

function boardSignature(board, oppColor) {
  let key = '';
  for (const sq of ALL_SQUARES) {
    const p = board[sq];
    key += !p ? '.' : (p.ownerId === oppColor ? p.type[0].toUpperCase() : p.type[0]);
  }
  return key;
}

// ---------------------------------------------------------------------------
// Per-game belief store. Keyed by the (stable) players array so each game — and
// each AI colour within an AI-vs-AI game — keeps its own belief, and a new game
// (new players array) starts fresh automatically. Shared by every belief-using
// agent (ChessAgent, ObscuroAgent) so a colour keeps one belief regardless of
// which agent drives it.
// ---------------------------------------------------------------------------

const beliefStore = new WeakMap();

export function getBelief(state, aiColor) {
  let byColor = beliefStore.get(state.players);
  if (!byColor) { byColor = new Map(); beliefStore.set(state.players, byColor); }
  let belief = byColor.get(aiColor);
  if (!belief) { belief = new Belief(aiColor); byColor.set(aiColor, belief); }
  return belief;
}
