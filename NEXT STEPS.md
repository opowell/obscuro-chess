What is the key to improving strength? Converting belief accuracy into strength is the open problem, and a bigger corpus is not on its own the answer. See [docs/STRENGTH-PLAN.md](docs/STRENGTH-PLAN.md) for the prioritized roadmap.

## Heuristics to remove

The engine should have no heuristics (see [CLAUDE.md](CLAUDE.md)). These are still in it, most used first, with what could replace each.

- **The particle belief (`src/belief.js`).** Every placement rule in it is hand-picked: anchor-distance weights (`1/(1 + distance)`), `MAX_POSSIBLE` truncation by distance, the move budget, anchor reservation, `THREAT_BIAS`, `MAX_LURKERS`, `RECAPTURE_TYPE_WEIGHT` and phantom-check rejection. P becomes a sample instead of giving up when it outgrows `CAP` or the time guard runs out, and a sample that empties is rebuilt by replaying the game's history. So the particle belief is only used when the tracker is attached mid-game, which has no history to work from, or when three rebuilds in a row all empty. `tryReacquire` reads the particle belief's per-piece square sets, so it has to be reworked before the module can go.
- **The static evaluator (`evaluate` in `src/ChessAgent.js`).** Piece-square tables and pawn-structure terms. Since every child of a refused node is priced by Stockfish (2026-10-01), it scores only positions the engine refuses or does not answer: 3 of 335,000 leaves over a 30-ply game at power 25. It is also `FogChess.evaluateState`, the generic search's default leaf value.
- **`ChessAgent` (`src/ChessAgent.js`).** Alpha-beta search on that evaluator, with the `CHESS_AGENT_SCORING` pessimism, tail and info weights. It is not the Obscuro agent, and nothing in fog-chess, battle-simulator or fow-chess-analyser uses it, but it is a public export, so removing it is a breaking change.

Removed: the move prior's hand-written inputs (the evaluator's piece-square tables, textbook material values, a king hand-set to 1000, a floor of 0.03), replaced on 2026-10-01 by tables fitted entry by entry on 2,872 Chess.com games and a floor fitted on held-out likelihood. Held out: move log-loss 3.002 → 2.661 nats; belief log-loss of the true position 0.97–1.09 ± 0.03 nats better, paired (two runs).

Removed: the hand-picked utility scale (`LEAF_CLAMP` = 1500, `SEARCH_WIN` = 8000), replaced on 2026-10-01 by the engine's own win/draw/loss as values in [−1, +1]. Measured beforehand with `move-quality.mjs` over 30 crawl games, 1,603 positions (null control 282/282 identical): level on expected score lost per move (−0.11 ± 0.37 points of %, z = −0.3), worse by 13.2 ± 2.1 on centipawns lost (z = 6.3), which it gives up where the result is not in doubt. Whether that margin matters in play would take games to show.

Not heuristics: `CAP`, `TIME_GUARD_MS`, the difficulty dial and `ANALYSIS_DEFAULTS` set how much of the principled computation is done, not what it computes.

## The opponent's hidden move is a choice, not a draw

Found 2026-10-01 in a fog-chess game: 1.c4 d5 2.Qa4 d4?? 3.Qxe8. After 2.Qa4 Black's king was capturable along a4–e8 and Black could not see it. Its belief put 6.2% on 2.Qa4 (49% on 2.Nc3/2.Nf3), so 2…d4, worth ~61% in the other positions, beat every move that blocks the diagonal (~46% everywhere). Within that belief the choice was right.

- **It is not the prior.** The learned tables (2026-10-01) put 2.Qa4 at 4.0%: real players play it a little less often than the hand tables implied, and at 4% d4 still wins the calculation.
- **The search hedges; the purification rule throws the hedge away.** Diagnosed 2026-10-02 by replaying the game with the agent and dumping the KLUSS gadget at Black's second move. At power 150 the root strategy was a mix in which every move was worth the same (−0.21 ± 0.003): e7–e6 38–58%, and the moves that block the diagonal (c6, Nc6, Bd7) 30–40% together. That is the equilibrium hedge against a White who may have played 2.Qa4. But `purify` only mixes when the gadget's `pmax` is below `safePmaxThreshold` (0.05); otherwise it commits to the single most likely move, and pmax was 1, so Black played e6 (or d4 at power 25), the deterministic choice an opponent can learn and exploit.
- **pmax is pinned at 1 almost always.** Over 228 decisions of three AI-vs-AI games at power 25, pmax < 0.05 in 50, pmax = 1 in 158 (median 1). The reason is the alternative value of a freshly sampled world, min(ṽ(h), v*), where ṽ(h) is our BEST reply in that world alone. One root move cannot be best in every world, so the opponent "enters" Resolve almost everywhere. Two consequences: the agent almost never mixes, and Maxmargin (weight 1 − pmax), which would focus on the worst-margin world (here Qa4, margin −0.6 to −0.8), never contributes. Resolve itself gives the Qa4 world only its prior weight α(J) ≈ its belief mass (0.045–0.11), the same as every other world White enters.

Measured 2026-10-02 with `scripts/hidden-threats.mjs` on 300 hidden king threats mined from the Chess.com crawl (the side to move has its king capturable by a piece it cannot see, and a defence exists; humans left the king capturable in 38%), dial 30:

| arm | king left capturable | root strategy on safe moves | true-board expected-score loss |
|---|---|---|---|
| shipped (win/draw/loss leaf values) | 52–54% | 44–46% | 4.7–5.9 points |
| always sample the mix (fix 1) | 53% | 45% | — |
| fresh worlds' alternate at the pre-solve strategy (fix 2) | 54% (57% with fix 1) | 43–45% | — |
| centipawn leaf values (as before 2026-10-01) | **14%** | 87% | 4.4 points |

- **Neither fix helps**, so neither ships. The Qa4 game was one where the search hedged and purification discarded it; across 300 real threats the search's own strategy is the problem, not purification.
- **The danger is not lost in the belief or the sample.** On 145 probes the belief put ≥ 50% on a threat and the search's worlds carried ~46% threat reach; the shipped search still left the king capturable 55% of the time there (centipawns: 1%).
- **The value scale is what changed.** Win/draw/loss values left the king capturable ~4× as often as centipawn values (paired: 128 probes only under WDL, 7 only under cp). On the expected-score yardstick the difference is noise (0.34 ± 0.84 points): the extra hangs are where full-information Stockfish rates the best defence as lost anyway (where the best defence keeps ≥ 10%, both hang ~15 of 110).

**The open question is the yardstick, and the utility.** Stockfish's win/draw/loss is a full-information, engine-vs-engine estimate. Under fog a "lost" position keeps swindle chances (the opponent cannot see everything), while a hung king loses for certain — so full-information WDL likely undervalues staying alive. And it is miscalibrated for this game in plain sight: it calls the opening position 94% drawn, while 19 of the crawl's 3,156 fog games were drawn (0.6%). The principled utility is the expected result of a position UNDER FOG: fit P(win, draw, loss | the engine's evaluation, material, …) to the outcomes of the crawl's games (`result` is recorded for every game), and value leaves with that. It replaces both the hand-picked centipawn scale and the full-information WDL, and it is measurable: held-out log-loss of game results, then this probe set and seat-swapped games.
