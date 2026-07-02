# Notebooks

Planned analyses (empty placeholders until the analytical code in
`src/sharpe_illusion/` is implemented):

- **`01_the_illusion.ipynb`** — fix the true Sharpe ratio at 0.0 / 0.5 / 1.0, simulate
  many PnL paths, and show side by side: zero-edge paths with a high realized SR, and
  positive-edge paths that stay in drawdown for years.
- **`02_estimation_error.ipynb`** — the sampling distribution of the realized SR as a
  function of horizon `T`; the relation `t = SR·√T`; Minimum Track Record Length for
  common SR levels.
- **`03_psr_dsr_mintrl.ipynb`** — Probabilistic, Deflated Sharpe Ratio and MinTRL
  applied to simulated tracks, including a negatively-skewed "short-vol" track and a
  multiple-testing (selection-bias) example.

See [`../docs/literature-overview.md`](../docs/literature-overview.md) for the theory.
