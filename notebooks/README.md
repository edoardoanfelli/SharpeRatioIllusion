# Notebooks

Executed top-to-bottom with `jupyter nbconvert --execute` (outputs are committed, so
they render on GitHub). Seeded with `20260702` for reproducibility.

- **`01_the_illusion.ipynb`** — fix the true Sharpe ratio at 0.0 / 0.5 / 1.0 and watch
  the realized SR and cumulative PnL: 16% of zero-edge paths clear SR 1 in a year, while
  a true-SR-0.5 strategy spends ~95% of its days below a prior peak and should *expect*
  a max drawdown of ~2.5× its annualized volatility (checked against the closed form).
- **`02_estimation_error.ipynb`** — the sampling distribution of the realized SR vs
  horizon (`√T` wall), Lo's (2002) standard-error formula validated against Monte Carlo,
  and the Minimum Track Record Length across SR levels.
- **`03_psr_dsr_mintrl.ipynb`** — PSR on a clean track; how negative skew / fat tails
  lower the PSR at the same headline Sharpe (the short-vol trap); and the Deflated
  Sharpe Ratio catching a best-of-200 selection that fools the naive PSR.

See [`../docs/literature-overview.md`](../docs/literature-overview.md) for the theory.
