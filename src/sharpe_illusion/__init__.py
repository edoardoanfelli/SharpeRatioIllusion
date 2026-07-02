"""sharpe_illusion — tools for studying when the Sharpe Ratio misleads.

Modules
-------
simulation : Monte Carlo generation of PnL paths from a *fixed true* Sharpe ratio.
metrics    : Sharpe ratio point estimate, its standard error (Lo 2002), and the
             Probabilistic / Deflated Sharpe Ratio and Minimum Track Record Length
             of Bailey & Lopez de Prado (2012, 2014).
drawdown   : drawdown depth and duration statistics under an assumed Sharpe ratio,
             following Rej, Seager & Bouchaud (CFM, 2017).
"""

__version__ = "0.1.0"

__all__ = ["simulation", "metrics", "drawdown"]
