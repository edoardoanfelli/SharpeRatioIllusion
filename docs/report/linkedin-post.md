# LinkedIn post

> Suggested images to attach, in order: `fig1_illusion.png`, `fig2_drawdown.png`,
> `fig4_overfitting.png`. Keep the first line as the hook (LinkedIn truncates after
> ~2 lines before "…see more").

---

A coin-flip strategy can post a Sharpe ratio of 2. A genuinely great strategy can sit in
a drawdown for seven years.

It's the same fact, seen from two sides — and it's why the number the whole industry
hires and fires on is quieter than it looks.

I ran the experiment you can never run with a real track record: fix the *true* Sharpe
ratio, then watch what the *measured* one does across thousands of simulated histories.

Three things fell out:

→ Over a single year, 1 in 6 strategies with **zero real edge** post a Sharpe above 1.0 —
the "hire me" line. You cannot tell them from the real thing.

→ A genuinely profitable strategy (true Sharpe 0.5) has a 5% chance its *current* drawdown
has already lasted 7+ years. Rej, Seager & Bouchaud (CFM) derived why: drawdown duration
scales as 1/S² — halve the Sharpe and you quadruple the expected time in the hole.

→ It's worse when you search. Trying 1,000 strategy variants with **zero edge** hands you
a best-of-batch Sharpe above 2 — from pure noise. This is the backtest-overfitting trap
López de Prado keeps warning about.

The cause is one line: t = SR × √T. Confidence only accrues with the *square root* of
time, so almost every track record you'll ever see is too short to trust on the Sharpe
alone. A true-Sharpe-0.5 strategy needs ~15 years of daily data to prove itself.

The fix isn't to throw the Sharpe ratio out. It's to stop reporting it naked:

• Always show the sample size (n)
• Report the Probabilistic Sharpe Ratio — accounts for track length + fat tails
• Deflate for how many strategies you tried (Deflated Sharpe Ratio)

Full write-up + open-source code (all figures reproducible) in the comments. 👇

#quant #tradingstrategies #riskmanagement #sharperatio #quantitativefinance
