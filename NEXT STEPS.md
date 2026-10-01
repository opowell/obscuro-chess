What is the key to improving strength? Converting belief accuracy into strength is the open problem, and a bigger corpus is not on its own the answer. See [docs/STRENGTH-PLAN.md](docs/STRENGTH-PLAN.md) for the prioritized roadmap.

## Heuristics to remove

The engine should have no heuristics (see [CLAUDE.md](CLAUDE.md)). These are still in it:

- **The particle belief (`src/belief.js`).** Every placement rule in it is hand-picked: anchor-distance weights (`1/(1 + distance)`), `MAX_POSSIBLE` truncation by distance, the move budget, anchor reservation, `THREAT_BIAS`, `MAX_LURKERS`, `RECAPTURE_TYPE_WEIGHT` and phantom-check rejection. Since P now resamples instead of giving up at `CAP`, the particle belief is only used when the exact tracker has given up for another reason: the time guard trips, a resampled P loses the true position, or the tracker is attached mid-game. Remove each of those causes, then delete the module. `tryReacquire` reads its per-piece square sets, so it has to be reworked too.
- **The move prior's features (`src/movePrior.js`).** The weights are fitted, but the features are hand-chosen: piece-square-table deltas, a flat castling bonus, capture and promotion value.
- **`ChessAgent` (`src/ChessAgent.js`).** Alpha-beta search with a hand-written evaluation (piece-square tables, pawn structure) and the `CHESS_AGENT_SCORING` pessimism, tail and info weights. It is not the Obscuro agent, but it is still exported.
