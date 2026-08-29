// onehop — boot.
//
// Loads the weights with a visible progress state, initializes the tensor
// HUD, and hands the model to each figure module as it comes online.
// Figures are added here one at a time as they are built.

import { loadWeights } from "./runtime.mjs";
import { hud, initHUD } from "./hud.mjs";

const loadingEl = document.getElementById("loading");
initHUD(document.getElementById("hud"));

let model = null;
try {
  model = await loadWeights((loaded, total) => {
    const kb = (n) => `${Math.round(n / 1024)} KB`;
    loadingEl.textContent = total
      ? `loading weights · ${kb(loaded)} / ${kb(total)}`
      : `loading weights · ${kb(loaded)}`;
  });
  hud.setModel(model);
  loadingEl.textContent =
    `weights loaded · ${model.config.n_params.toLocaleString("en-US")} params · fp16 · ` +
    `val exact match ${(model.config.val_exact_match * 100).toFixed(1)}%`;
} catch (err) {
  loadingEl.textContent = "weights failed to load — figures are disabled, the text still reads";
  console.error(err);
}

// ---- figures come online here, one per module session ----
{
  // Module 1 is a pure simulation; it does not need the weights.
  const { initOneHop } = await import("./figures/onehop.mjs");
  initOneHop(document.getElementById("fig-onehop"));
}
if (model) {
  const { initSDPA } = await import("./figures/sdpa.mjs");
  initSDPA(document.getElementById("fig-sdpa"), model);
}
{
  // Module 3 demonstrates footnote 4's statistics on synthetic vectors.
  const { initSqrtDk } = await import("./figures/sqrtdk.mjs");
  initSqrtDk(document.getElementById("fig-sqrtdk"));
}
{
  // Module 4 computes PE from the same formula the model uses.
  const { initPE } = await import("./figures/pe.mjs");
  initPE(document.getElementById("fig-pe"));
}
if (model) {
  const { initMask } = await import("./figures/mask.mjs");
  initMask(document.getElementById("fig-mask"), model);
}
if (model) {
  const { initHeads } = await import("./figures/heads.mjs");
  initHeads(document.getElementById("fig-heads"), model);
}
{
  const { initBottleneck } = await import("./figures/bottleneck.mjs");
  initBottleneck(document.getElementById("fig-bottleneck"));
}
{
  const { initCrossover } = await import("./figures/crossover.mjs");
  initCrossover(document.getElementById("fig-crossover"));
}
{
  const { initLrate } = await import("./figures/lrate.mjs");
  initLrate(document.getElementById("fig-lrate"));
}
