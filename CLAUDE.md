# obscuro-chess

## No heuristics

Do not add heuristics to this engine, and do not fix a heuristic by tuning it or
by adding another one. A heuristic here means a hand-picked rule or constant that
stands in for something the engine could derive: "pieces probably stay near
home", "weight a square by 1/(1 + distance)", "a recapture is 9× likelier with a
pawn", "never place more than two hidden attackers". Each of these has produced
boards or moves that no real game could reach, and each new one has needed
another to cover for it.

What counts as principled instead:

- **Derived from the rules and the observations.** The exact belief P is the set
  of positions that real move histories reach and that match everything seen.
- **Fitted to data.** The move prior π has weights fitted to a game corpus.
- **The paper's method.** The Obscuro search and its Stockfish leaf evaluation.
- **An honest approximation of one of the above, labelled as one.** For example,
  when P outgrows its cap it is resampled without bias (`resample()` in
  `src/exactBelief.js`) and flagged `sampled`. A compute budget like `CAP` limits
  how much of the right thing is done. It does not replace it with a guess.

When something principled is too expensive, approximate it and say so. Do not
substitute a rule of thumb. If a heuristic seems unavoidable, raise it with the
user first.

The heuristics still in the code are listed under "Heuristics to remove" in
[NEXT STEPS.md](NEXT%20STEPS.md).
