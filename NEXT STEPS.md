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
- **No fixed belief is safe against a choosing opponent.** The search treats White's hidden move as chance with fixed probabilities. But White chose it, and if Black never guards against Qa4, White should always play it. However the prior is fitted, maximising expected score against it is exploitable wherever a cheap defence exists against a decisive threat. The paper's answer is safe subgame solving: the Resolve/KLUSS gadget lets the opponent pick which world to enter, at its blueprint value. Find out why it did not protect against this here (is the blueprint's value for White's Qa4 infoset too low, is the gadget's alternative computed for this world at all?) before changing anything.

