# The Sharpe Ratio Illusion

### A coin-flip strategy can walk in with a Sharpe of 2. A genuinely great one can lose money for four years straight. It's the same fact, seen from two sides.

---

The Sharpe ratio is the number the whole industry runs on. Allocators screen on it,
managers are hired and fired on it, backtests are ranked by it. It has one job:
tell skill apart from luck by dividing how much you earned by how much you bounced
around earning it.

The problem is that, over the track records we actually have — months, a few years —
it is mostly **measuring the bounce, not the skill**. This piece shows what that
looks like, why it happens, and what to report instead. Every chart comes from a
tiny simulation library; the code is linked at the end so you can rerun all of it.

## The setup

The Sharpe ratio is

$$ SR = \frac{\text{average return} - \text{risk-free}}{\text{volatility of returns}} $$

usually annualized. A Sharpe of 1 is a good systematic strategy; 2 is excellent; above
that people start naming buildings after you.

To study it honestly we do the one thing you can never do with a real track record:
**fix the truth and look at the estimate**. We generate return streams whose *true*
Sharpe ratio is exactly 0, 0.5, or 1, run thousands of them, and watch what the
*measured* Sharpe does. If the measured number reliably recovered the truth, the story
would end here. It doesn't.

## Illusion #1 — a year of data can't tell 0 from 1

Here are the realized one-year Sharpe ratios for three populations: a **pure coin flip**
(true SR = 0), a decent strategy (0.5), and an excellent one (1.0).

![Distribution of realized one-year Sharpe ratios for true SR of 0, 0.5 and 1 — they overlap almost completely](figures/fig1_illusion.png)

The three clouds sit almost on top of each other. **16% of the zero-edge strategies —
one in six literal coin flips — post a realized Sharpe above 1.0 over a full year**, the
threshold that gets a manager hired. Plenty clear 2. Nothing is wrong with them; they
just got a good year. Point at any single strategy on the right-hand side of this chart
and you cannot say whether it has an edge or a tailwind.

## Illusion #2 — a great strategy that bleeds for years

Now the mirror image. Take a genuinely good strategy — true Sharpe 0.5, a number most
funds would love — and simply watch its cumulative PnL.

![Cumulative PnL of many true-SR-0.5 paths; the expected line rises steadily but individual paths spend years below zero](figures/fig2_drawdown.png)

The straight blue line is the edge doing its work in expectation. The individual paths
are what a real investor actually experiences, and they are a mess. **18% of these
genuinely-profitable strategies are still underwater three years in.** The highlighted
path is a strategy with a *real, positive edge* that spends **nearly four years in the
red** before it pays off. In the real world it gets redeemed at year two and the edge is
never collected.

This is the result Jean-Philippe Bouchaud and colleagues at Capital Fund Management made
precise in *"You are in a drawdown — when should you start worrying?"*: for realistic
Sharpe ratios, multi-year losing streaks are not a malfunction, they are the **expected
behavior**. A long drawdown is not, by itself, evidence that anything broke.

## Why both happen: the √T wall

Both illusions are the same piece of arithmetic. The signal-to-noise of a track record —
the t-statistic on its average return — is

$$ t = SR \times \sqrt{T} $$

The edge (`SR`) is fixed by the strategy. The only way to accumulate statistical
confidence is to accumulate **time**, and it comes in as a *square root*. To double your
certainty you need four times the data. The estimation error around a Sharpe ratio
shrinks only as `1/√T`, which is agonizingly slow.

## So how long *do* you need?

Turn the equation around and ask how much daily data it takes before you can be 97.5%
confident a strategy's true Sharpe is even *positive* (this is Bailey & López de Prado's
**Minimum Track Record Length**):

![Bar chart of years of daily data needed to trust a Sharpe ratio: SR 0.25 needs 61 years, 0.5 needs 15, 1.0 needs 4, 2.0 needs 1](figures/fig3_mintrl.png)

A genuinely good strategy at **Sharpe 0.5 needs about 15 years** of daily data before its
track record clears the bar. A quarter-Sharpe strategy — still worth money — needs a
human lifetime. Only the rare Sharpe-2 machine can prove itself inside a year. Almost
every track record you will ever evaluate is **too short to trust on the Sharpe ratio
alone**. That is not a data-quality problem you can fix; it is the geometry of the
estimator.

## The trap that makes it worse: searching

Everything above assumed you tested *one* strategy. Nobody tests one strategy. You try
variations — parameters, universes, filters — and report the best. Watch what that does,
using nothing but zero-edge coin flips:

![The best Sharpe ratio found across N zero-edge two-year backtests rises past 1 after ~10 tries and past 2 by 1000 tries](figures/fig4_overfitting.png)

Every strategy on this chart has **zero real edge**. Yet trying just **10 variants**
already hands you a best-of-batch Sharpe above the hire-me line, and searching **1,000 of
them buys a Sharpe above 2 — from pure noise**. This is why a backtest's headline Sharpe
is close to meaningless without knowing how many configurations were tried to find it. It
is also the core of Marcos López de Prado's argument that most published backtests are
selection-bias artifacts.

## What to report instead

The fix is not to abandon the Sharpe ratio but to stop reporting it naked. The literature
gives three concrete upgrades, all implemented in the repo:

1. **Never show a Sharpe without its sample size** — and ideally its standard error.
   `SE(SR) ≈ √((1 + ½·SR²)/n)` (Lo, 2002). The number without `n` is not interpretable.
2. **Report the Probabilistic Sharpe Ratio (PSR)** — the *probability* the true Sharpe
   beats a threshold, given your sample length **and the shape of your returns**. Negative
   skew and fat tails (the profile of every "sell insurance" strategy that looks great
   until it doesn't) push this probability *down* even when the point estimate looks great.
3. **Deflate for your search — the Deflated Sharpe Ratio (DSR)** — judge the winning
   backtest not against zero but against the best Sharpe you'd *expect from luck* given how
   many strategies you tried. The more you searched, the higher the bar.

## The honest caveat

Everything here used clean, well-behaved Gaussian returns — the *friendliest* possible
case. Real returns have fat tails, volatility clustering, and serial correlation, all of
which make the estimation error **larger**, not smaller. The illusion shown here is the
optimistic version.

None of this makes the Sharpe ratio useless. It makes it a **noisy estimate that has to
be reported like one** — with its sample size, its distribution, and an honest count of
how hard you searched. A number that can be manufactured by a coin flip in a good year
deserves exactly that much skepticism.

---

*Code and all figures are reproducible from the open-source repo:
[SharpeRatioIllusion](https://github.com/edoardoanfelli/SharpeRatioIllusion). Simulations
use `numpy`; the Sharpe standard error, PSR, DSR and MinTRL are implemented from the
papers below.*

**Sources**

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4).
- Rej, A., Seager, P., & Bouchaud, J.-P. (2017). *You are in a drawdown. When should you
  start worrying?* Capital Fund Management.
- Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
  Journal of Risk 15(2).
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of
  Portfolio Management 40(5).
