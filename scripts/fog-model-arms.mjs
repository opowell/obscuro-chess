// Other fog value models, as agent overlays (`fogModel`, measurement only), so
// self-play.mjs and hidden-threats.mjs can play them against the shipped one
// (src/fogValueModel.js). Frozen copies; see scripts/fit-fog-value.mjs.
const BASE = ['1', 'L', 'Lm', 'm', 'w'];

export const FOG_MODEL_ARMS = {
  // 2026-10-02 (241574): engine score, material on the board, colour.
  fog0: { fogModel: { depth: 4, clip: 0.0005, features: BASE,
    beta: [-0.073256, 0.292423, -0.117159, 0.051220, -0.142990] } },
  // The shipped model plus the INFORMATION STATE (the share of the opponent's
  // pieces each side sees) — the best fit held out (0.5545 against 0.5554 nats,
  // z = −2.3), and rejected in play: on 300 hidden king threats it left the king
  // capturable 19.7% of the time against 15.0% (paired 24 probes against 10),
  // and it scored 46.7% ± 4.6 over 60 seat-swapped pairs. The fit is
  // observational: seeing more goes with attacking, and the search needs what
  // LOOKING is worth, which the coefficient does not measure.
  balseen: { fogModel: { depth: 4, clip: 0.0005, features: [...BASE, 'e', 'em', 'bal', 'seenX', 'seenY'],
    beta: [1.042552, 0.300590, -0.282682, -1.833097, -0.175833, -2.113412, 3.642501, 5.472563, 1.123377, -0.518355] } },
};
