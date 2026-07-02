# SharpeRatioIllusion

Testing unprofitable trading strategies and studying **when the Sharpe Ratio (SR)
is — and is not — a reliable indicator of skill**.

The starting point is a well-known but under-appreciated fact: over finite samples
a strategy with *zero* edge can display a high realized Sharpe ratio purely by luck,
while a genuinely profitable strategy can stay in drawdown (cumulative PnL below zero)
for *years*. The distance between the *true* SR and its *estimate* shrinks only as
`√T`, so short and even medium-length track records are dominated by estimation noise.

This repository reproduces that phenomenon by Monte Carlo simulation and then applies
the corrective statistics from the literature — Lo (2002), and the Probabilistic /
Deflated Sharpe Ratio framework of Bailey & López de Prado.

## Core idea

The t-statistic of the mean return is

```
t = SR_hat · √T
```

so distinguishing skill from luck requires `SR² · T` to be large. A true SR of 0.5
needs on the order of 16 years of daily data to reach `t ≈ 2`. That is precisely the
window in which a good strategy "looks dead" and a useless one "looks great".

See [`docs/literature-overview.md`](docs/literature-overview.md) for the full
statistical background and citations.

## Planned structure

```
src/sharpe_illusion/
  simulation.py   # Monte Carlo: fix the TRUE Sharpe ratio, generate PnL paths
  metrics.py      # SR, standard error (Lo 2002), PSR, DSR, MinTRL
  drawdown.py     # drawdown depth / duration under an assumed SR (Bouchaud et al.)
notebooks/
  01_the_illusion.ipynb       # zero-edge/high-SR vs true-edge/multi-year drawdown
  02_estimation_error.ipynb   # distribution of SR_hat vs T; t = SR·√T
  03_psr_dsr_mintrl.ipynb     # de Prado's corrections in practice
docs/                         # literature notes with citations
tests/
```

> Status: scaffolding + literature notes are in place. Analytical implementations
> (`simulation.py`, `metrics.py`, `drawdown.py`) are stubbed with signatures and
> references, to be filled in next.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
