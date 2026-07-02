"""Calibrate Q_p(x) for E[MDD] of drifted BM by Monte Carlo + Richardson extrapolation.

Q_p(x) = E[MDD] * mu / (2 sigma^2),  x = mu^2 T / (2 sigma^2).
With sigma=1, mu=1: T = 2x, Q_p = E[MDD]/2.
Discrete-sampling bias ~ c/sqrt(N); Richardson: E = 2*E_{4N} - E_N.
"""
import numpy as np

rng = np.random.default_rng(7)

def emdd(T, n_steps, n_paths, batch=500):
    dt = T / n_steps
    total = 0.0
    done = 0
    while done < n_paths:
        b = min(batch, n_paths - done)
        r = rng.normal(dt, np.sqrt(dt), size=(b, n_steps))
        pnl = np.cumsum(r, axis=1)
        dd = np.maximum.accumulate(pnl, axis=1) - pnl
        total += dd.max(axis=1).sum()
        done += b
    return total / n_paths

xs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
               5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0])
N1, N4 = 4096, 16384
P = 12000
print("x, Qp_richardson, small_asym, large_asym")
rows = []
for x in xs:
    T = 2 * x
    e1 = emdd(T, N1, P)
    e4 = emdd(T, N4, P)
    e = 2 * e4 - e1
    qp = e / 2
    rows.append((x, qp))
    print(f"{x:8.3f}  {qp:.5f}   small={0.5*np.sqrt(np.pi*x):.5f}  large={0.25*np.log(x)+0.49088:.5f}")

print("\nPython literal:")
print("_QP_X = np.array([", ", ".join(f"{x:g}" for x, _ in rows), "])")
print("_QP_Y = np.array([", ", ".join(f"{q:.5f}" for _, q in rows), "])")
