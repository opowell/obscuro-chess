// ---------------------------------------------------------------------------
// Does a better-calibrated belief make the AI PLAY better?
//
//   node scripts/strength-belief.mjs [--pairs N] [--max-turns N] [--tau 200]
//                                        [--arm prior|alpha|null]
//
// The `prior` arm tests the SHIPPED π (movePrior.js FITTED_WEIGHTS) against
// uniform π; `--tau N` swaps in the old single-temperature model instead.
//
// Calibration (calibrate-belief.mjs) answers "is the belief more accurate". This
// answers the separate and harder question of whether that accuracy converts into
// wins, by playing ChessObscuroAgent against itself under fog with the two seats
// differing in exactly ONE thing and nothing else — same search, same evaluator,
// same rng. Three arms, because the prior reaches the AI's play through exactly one
// channel (the search's world draw) and confounding the two halves hides which one
// any result belongs to:
//
//   --arm prior   one seat's belief uses the move prior, the other's uniform π.
//                 Both draw their search worlds the same way.
//   --arm alpha   BOTH seats hold the identical prior-weighted belief and differ
//                 only in the sampling exponent (exactBelief.js's `sampleAlpha`):
//                 α=1 draws ∝ the posterior, α=0 draws uniformly over P. This is
//                 the arm that answers "should the SEARCH weight its world draw",
//                 which is not the same question as "is the belief calibrated".
//   --arm null    the CONTROL: both seats identical. Establishes what a "no
//                 difference" result actually looks like in this harness. Run it
//                 before believing any lopsided result from the other two.
//
// SEAT-SWAPPED PAIRS ARE MANDATORY, not a nicety, and in THIS game the reason is
// overwhelming rather than theoretical: measured here, WHITE WINS ROUGHLY 10-11 OF
// EVERY 12 GAMES whatever the arms are doing — the same in the `null` arm, where the
// seats are identical. First-move advantage under fog dwarfs anything the belief
// contributes. So the seat swap is not guarding against a subtle bias, it is the
// only reason the numbers mean anything at all: each pairing is played twice, once
// with the arm-under-test on white and once on black, and the white advantage
// cancels between the two. Aggregating anything other than whole pairs measures
// which seat the arm sat in. (Cf. memory: civ1 seat-1 bias.)
//
// The corollary for reading the output: with white winning most games, a balanced
// pair contributes 1-1 and the residual signal lives entirely in the rare games the
// black seat wins. That makes this harness far less sensitive than the raw game
// count suggests — a 16-game run resolves almost nothing.
//
// This is SLOW — every ply is a full Obscuro search with a Stockfish leaf eval —
// so the default is a handful of short games. Treat the output as a smoke test for
// "the prior did not make the AI worse" rather than as a rating measurement; the
// sample needed to resolve a small true difference is far larger than is practical
// here, and the script says so in its own summary.
// ---------------------------------------------------------------------------

import { playMatch } from '../src/playMatch.js';
import { ChessObscuroAgent } from '../src/ObscuroAgent.js';
import { setMovePriorForSeat, setBeliefSampleAlphaForSeat } from '../src/exactBelief.js';
import { makeMovePrior, UNIFORM_PRIOR, FITTED_WEIGHTS } from '../src/movePrior.js';
import { quit as stockfishQuit } from '../src/stockfish.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };
const pairs = Number(arg('pairs', '3'));
const maxTurns = Number(arg('max-turns', '30'));
// Default: whatever production actually serves (the FITTED weights). `--tau N`
// overrides it with the old single-temperature model, which is how you reproduce
// the 2026-07-30 numbers — but note that is no longer the shipped π.
const tau = Number(arg('tau', '0'));
const armName = arg('arm', 'prior');
if (!['prior', 'alpha', 'null'].includes(armName)) throw new Error(`--arm must be prior|alpha|null, got ${armName}`);

const prior = tau > 0 ? makeMovePrior({ temperature: tau }) : makeMovePrior(FITTED_WEIGHTS);
// `A` is the arm under test, `B` the control it must beat.
const labels = armName === 'prior' ? { a: 'prior', b: 'uniform-π' }
  : armName === 'alpha' ? { a: 'α=1', b: 'α=0' }
  : { a: 'seat-A', b: 'seat-B' };
const tally = { priorWins: 0, uniformWins: 0, draws: 0, unfinished: 0 };
const games = [];

for (let p = 0; p < pairs; p++) {
  for (const priorSeat of ['white', 'black']) {
    const otherSeat = priorSeat === 'white' ? 'black' : 'white';
    // Fresh per game: the belief store is keyed by the players array identity, so
    // a new engine gets new trackers, which pick up these overrides on construction.
    if (armName === 'null') {
      // THE CONTROL. Both seats identical in every respect. Any systematic winner
      // here is a property of the harness or the engine, not of the belief, and
      // invalidates the other arms until explained — run this FIRST when an arm
      // comes back with an implausible sweep.
      setMovePriorForSeat(priorSeat, prior);
      setMovePriorForSeat(otherSeat, prior);
      setBeliefSampleAlphaForSeat(priorSeat, 0);
      setBeliefSampleAlphaForSeat(otherSeat, 0);
    } else if (armName === 'prior') {
      setMovePriorForSeat(priorSeat, prior);
      setMovePriorForSeat(otherSeat, UNIFORM_PRIOR);
      setBeliefSampleAlphaForSeat(priorSeat, null);
      setBeliefSampleAlphaForSeat(otherSeat, null);
    } else {
      // Identical beliefs; the ONLY difference is how the search draws from them.
      setMovePriorForSeat(priorSeat, prior);
      setMovePriorForSeat(otherSeat, prior);
      setBeliefSampleAlphaForSeat(priorSeat, 1);
      setBeliefSampleAlphaForSeat(otherSeat, 0);
    }

    // A FRESH AGENT PER SEAT PER GAME. The exported `ObscuroAgent` is a singleton
    // whose `_carry` / `_prevValues` maps hold the KLUSS carryover search tree per
    // colour and are never reset between games, so reusing it makes game N+1 start
    // from game N's tree — cross-game contamination that shows up as an
    // order-dependent winner and silently invalidates every arm. The `null` arm
    // caught this too.
    const agents = { white: new ChessObscuroAgent(), black: new ChessObscuroAgent() };
    const t0 = Date.now();
    const { result } = await playMatch(agents, {
      maxTurns, config: { fogOfWar: true, difficulty: 25 },
    });
    const s = Math.round((Date.now() - t0) / 1000);

    // NB: the field is `winnerId`, not `winner` (FogChess.getResult). Reading the
    // wrong one silently credits EVERY decisive game to the same arm and produces a
    // clean, entirely fictional sweep — which is exactly what the `null` arm exists
    // to catch, and did.
    const winner = result?.winnerId;
    if (result && result.outcome === 'win' && winner !== 'white' && winner !== 'black') {
      throw new Error(`unexpected winner id ${JSON.stringify(winner)} — verdict logic would be silently wrong`);
    }
    let verdict;
    if (!result) { tally.unfinished++; verdict = 'unfinished (turn cap)'; }
    else if (result.outcome === 'draw') { tally.draws++; verdict = 'draw'; }
    else if (winner === priorSeat) { tally.priorWins++; verdict = `${labels.a} wins (as ${priorSeat})`; }
    else { tally.uniformWins++; verdict = `${labels.b} wins (as ${otherSeat})`; }
    games.push({ pair: p + 1, priorSeat, verdict, s });
    console.log(`pair ${p + 1} · ${labels.a} as ${priorSeat.padEnd(5)} → ${verdict.padEnd(28)} ${s}s`);
  }
}

setMovePriorForSeat('white', null);
setMovePriorForSeat('black', null);
setBeliefSampleAlphaForSeat('white', null);
setBeliefSampleAlphaForSeat('black', null);

const decisive = tally.priorWins + tally.uniformWins;
console.log(`\narm ${armName}: ${labels.a} vs ${labels.b}, π=${tau > 0 ? `τ=${tau}` : 'FITTED'}, ${pairs} seat-swapped pairs` +
  ` (${games.length} games), maxTurns ${maxTurns}`);
console.log(`  ${labels.a.padEnd(10)} ${tally.priorWins}`);
console.log(`  ${labels.b.padEnd(10)} ${tally.uniformWins}`);
console.log(`  draws      ${tally.draws}   unfinished ${tally.unfinished}`);
if (decisive === 0) {
  console.log('\nNo decisive games — this says nothing either way about strength.');
} else {
  console.log(`\n${tally.priorWins}/${decisive} decisive games to ${labels.a}. With a sample this small` +
    ' that is a smoke test, not a measurement: it can rule out a large regression, not detect a small gain.' +
    ' A near-sweep in either direction, on the other hand, is signal — a fair coin gives 12/12 about twice' +
    ' in 10,000 tries.');
}
stockfishQuit();
