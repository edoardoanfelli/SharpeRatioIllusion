# The Sharpe Ratio Illusion — Literature Overview

This note frames the problem the repository studies and collects the statistical
results we will implement and reproduce. It is deliberately opinionated about *which*
corrections matter and *why*, so the analytical code that follows has a clear target.

---

## 1. The problem in one line

The Sharpe ratio (SR) is a **point estimate of a ratio of two unknown moments**, and
its **estimation error shrinks only as `√T`**. Over the sample lengths available in
practice (months to a few years), that error is large enough that:

- a strategy with **zero true edge** can show a **high realized SR** by luck, and
- a strategy with a **genuinely positive edge** can sit in **cumulative loss for years**.

Everything below quantifies this and gives tools to correct for it.

---

## 2. The estimator and its noise (Lo, 2002)

For returns `r_t` with mean `μ` and standard deviation `σ`, the Sharpe ratio is

```
SR = (μ − r_f) / σ
```

The sample estimator `SR_hat` plugs in sample moments. Andrew Lo's key contribution
is the **standard error** of that estimator.

**IID normal returns:**

```
SE(SR_hat) ≈ sqrt( (1 + ½·SR²) / n )
```

**Non-normal returns** (skewness `g₃`, excess kurtosis via `g₄`):

```
SE(SR_hat) ≈ sqrt( (1 − g₃·SR + (g₄ − 1)/4 · SR²) / n )
```

**Autocorrelated returns:** annualising a per-period SR by `√(periods_per_year)`
is only valid under independence. Lo derives a correction factor built from the
return autocorrelations; positive autocorrelation **inflates** the naive annualised
SR. This matters for any strategy with momentum-like serial dependence.

**Consequence — the t-statistic.** Because `SR_hat / SE(SR_hat) ≈ SR_hat·√n` (IID
case), the t-statistic of the mean return is essentially

```
t = SR_hat · √T
```

Distinguishing skill from luck requires `SR²·T` to be large. Rough daily-data
intuition:

| True (annual) SR | Years for t ≈ 2 |
|------------------|-----------------|
| 1.0              | ~4              |
| 0.5              | ~16             |
| 0.25             | ~64             |

A true SR of 0.5 — a *good* strategy — needs on the order of **16 years** before its
track record is statistically distinguishable from noise. That is the illusion:
during those years, good and useless strategies are hard to tell apart from SR alone.

---

## 3. Drawdowns: the false-negative side (Bouchaud et al., 2017)

Rej, Seager & Bouchaud, *"You are in a drawdown. When should you start worrying?"*
(CFM, 2017) attack the symmetric question: given an **assumed** true SR, how deep and
how long a drawdown is *normal*?

Key results we will reproduce:

- Modelling cumulative PnL as **Brownian motion with drift**, the expected maximum
  drawdown and its duration have tractable forms in the SR and the horizon.
- Even with a solidly positive true SR, the **expected time under water** runs to
  **years**, and the maximum drawdown is a large multiple of the annual volatility.
- Therefore a multi-year losing streak is **not**, by itself, evidence that a
  strategy has stopped working — it is often inside the confidence band of the
  original SR hypothesis.

This is the direct source of the phenomenon the project is named after: a profitable
strategy that "looks broken" for years.

---

## 4. Correcting the estimate: PSR, MinTRL (Bailey & López de Prado, 2012)

Instead of reporting `SR_hat` as if it were the truth, the **Probabilistic Sharpe
Ratio (PSR)** reports a probability.

**PSR** — probability that the true SR exceeds a benchmark `SR*`, given the estimate
over `n` observations and the return distribution's first four moments:

```
PSR = Φ( (SR_hat − SR*) · √(n − 1)
          / sqrt(1 − skew·SR_hat + (kurt − 1)/4 · SR_hat²) )
```

- Negative skew and fat tails (`kurt > 3`) **reduce** PSR — the SR is less
  trustworthy than the point estimate suggests. This is exactly the profile of
  "sell-the-tail" strategies (e.g. short deep out-of-the-money puts) that show a
  gorgeous SR until the tail event arrives.

**Minimum Track Record Length (MinTRL)** — how much data is needed for PSR to reach a
target confidence:

```
MinTRL = 1 + (1 − skew·SR_hat + (kurt − 1)/4 · SR_hat²) · (Φ⁻¹(conf) / (SR_hat − SR*))²
```

This turns "is this track record long enough?" into a number.

---

## 5. Correcting for selection: the Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

The PSR still assumes a **single** strategy was tested. In practice a researcher tries
many configurations and reports the best — **selection bias / backtest overfitting**.

The **Deflated Sharpe Ratio (DSR)** is the PSR evaluated against a benchmark `SR*`
that is **not zero** but the **expected maximum SR achievable by chance** across the
`N` trials actually run. Using the variance of the SR estimates across trials and an
extreme-value (Gumbel) approximation:

```
SR* ≈ sqrt(Var(SR across trials)) · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
```

(`γ` = Euler–Mascheroni constant). The more configurations tried, the higher the bar
a reported SR must clear to be credible. This is the piece that connects the Sharpe
ratio directly to López de Prado's broader thesis on overfitting in quantitative
finance.

---

## 6. What the repository will demonstrate

| Notebook | Claim it makes concrete |
|----------|-------------------------|
| `01_the_illusion`      | With `sr_true = 0`, a non-trivial fraction of paths show a high realized SR; with `sr_true = 0.5`, many paths stay in drawdown for years. |
| `02_estimation_error`  | The sampling distribution of `SR_hat` vs `T`; `t = SR·√T`; MinTRL for common SR levels. |
| `03_psr_dsr_mintrl`    | PSR/DSR/MinTRL applied to simulated tracks, including a skewed "short-vol" track and a multiple-testing example. |

---

## 7. Practical takeaways

1. **Never report a bare SR** without `n` and, ideally, its standard error or PSR.
2. **Skew and kurtosis matter**: a high SR on negatively-skewed returns is a warning,
   not a reassurance.
3. **Count your trials.** If you searched over many strategies, deflate the SR before
   believing it.
4. **A long drawdown is not proof of failure**, and a short winning streak is not
   proof of skill. Both must be read against the horizon `T`.

---

## References

Full bibliographic details in [`references.bib`](references.bib).

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4).
- Rej, A., Seager, P., & Bouchaud, J.-P. (2017). *You are in a drawdown. When should
  you start worrying?* Capital Fund Management.
- Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
  Journal of Risk 15(2).
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio
  Management 40(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
