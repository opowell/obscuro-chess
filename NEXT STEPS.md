What is the key to improving strength? Converting belief accuracy into strength is the open problem, and a bigger corpus is not on its own the answer. See [docs/STRENGTH-PLAN.md](docs/STRENGTH-PLAN.md) for the prioritized roadmap.

## Heuristics to remove

The engine should have no heuristics (see [CLAUDE.md](CLAUDE.md)). These are still in it:

- **The particle belief (`src/belief.js`).** Every placement rule in it is hand-picked: anchor-distance weights (`1/(1 + distance)`), `MAX_POSSIBLE` truncation by distance, the move budget, anchor reservation, `THREAT_BIAS`, `MAX_LURKERS`, `RECAPTURE_TYPE_WEIGHT` and phantom-check rejection. P now becomes a sample instead of giving up when it outgrows `CAP` or the time guard runs out, so the particle belief is only used when a sampled P loses every position consistent with what is seen (it empties), or the tracker is attached mid-game. The first could be recovered without it by replaying the game's observations into a fresh sample. Remove both causes, then delete the module. `tryReacquire` reads its per-piece square sets, so it has to be reworked too.
- **The move prior's features (`src/movePrior.js`).** The weights are fitted, but the features are hand-chosen: piece-square-table deltas, a flat castling bonus, capture and promotion value.
- **`ChessAgent` (`src/ChessAgent.js`).** Alpha-beta search with a hand-written evaluation (piece-square tables, pawn structure) and the `CHESS_AGENT_SCORING` pessimism, tail and info weights. It is not the Obscuro agent, but it is still exported.
