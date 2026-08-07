#!/usr/bin/env node
// ---------------------------------------------------------------------------
// obscuro-chess — one entry point for the demo, the tuning harnesses, and the
// settings system.
//
//   obscuro-chess demo --difficulty 60
//   obscuro-chess config                      # what every parameter is set to
//   obscuro-chess config --set chess.LEAF_CLAMP=900
//   obscuro-chess move-quality --settings sweep.json --arm reach
//
// The settings flags are applied BEFORE the target script is imported, so a
// script sees its parameters already resolved — including the module-level
// constants its imports read at load time.
// ---------------------------------------------------------------------------

import { applyCliSettings, maybePrintConfig, SETTINGS_USAGE } from '../src/cli.js';

const COMMANDS = {
  demo: '../examples/demo.js',
  calibrate: '../scripts/calibrate-belief.mjs',
  'fit-prior': '../scripts/fit-move-prior.mjs',
  'adopt-corpus': '../scripts/adopt-corpus.mjs',
  'move-quality': '../scripts/move-quality.mjs',
  strength: '../scripts/strength-belief.mjs',
};

const USAGE = `
obscuro-chess <command> [flags]

Commands:
  demo            play a fog game between two Obscuro agents
  move-quality    paired cp-loss measurement (the strength harness)
  strength        seat-swapped self-play win rates
  calibrate       belief calibration over recorded sessions
  fit-prior       refit the move prior by MLE
  adopt-corpus    measure a new corpus end to end, and ship it if it wins
  config          print the resolved configuration and exit

${SETTINGS_USAGE}

Every command also takes its own flags; pass --help to one to see them.
`.trim();

const argv = process.argv.slice(2);
if (!argv.length || argv[0] === '--help' || argv[0] === '-h') {
  console.log(USAGE);
  process.exit(argv.length ? 0 : 1);
}

const command = argv[0];
const { rest, printConfig } = applyCliSettings(argv.slice(1));

if (command === 'config') {
  // `config` with no flag means "show me everything", which is the only useful
  // reading of it; --print-changed still narrows to what was overridden.
  await maybePrintConfig(printConfig || 'all');
  process.exit(0);
}

const target = COMMANDS[command];
if (!target) {
  console.error(`obscuro-chess: unknown command "${command}"\n\n${USAGE}`);
  process.exit(1);
}

await maybePrintConfig(printConfig);

// The target scripts read process.argv directly, so hand them the remainder as
// if they had been invoked with it.
process.argv = [process.argv[0], new URL(target, import.meta.url).pathname, ...rest];
await import(target);
