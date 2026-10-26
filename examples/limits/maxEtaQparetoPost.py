"""Envelope eta_ub(Q) = min_nu [g(nu) + nu Q] of maxEtaQparetoGCD.py parts, table and figure.

Usage: python maxEtaQparetoPost.py fig.png tag1 [tag2 ...]   (tags under results/, parts in
results/etaQparts/<tag>_etaQ_out_nuKK.npz, NU=16, NULOG=-4,1; missing weights are skipped)
"""
import sys, glob, os
import numpy as np, scipy.io as sio, scipy.linalg as la
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__)); res = os.path.join(here, "results")
nNu, nuLog = 16, (-4.0, 1.0)
Qgrid = np.logspace(1, 3.5, 600)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
styles = {"gGlobal": ("global", ":"), "gGCD": ("GCD", "--"), "gLocal": ("local", "-")}
colors = ["#0072BD", "#D95319"]
for it, tag in enumerate(sys.argv[2:]):
    d = sio.loadmat(os.path.join(res, f"{tag}_in.mat"), squeeze_me=True)
    nuMax = np.max(la.eigh(2*np.asarray(d["R0"], float), np.asarray(d["omW"], float), eigvals_only=True))
    nuList = nuMax*np.logspace(*nuLog, nNu)
    parts = {k: np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(res, "etaQparts", f"{tag}_etaQ_out_nu*.npz")))
             for k in [int(f[-6:-4])]}
    idx = sorted(parts); nu = nuList[idx]
    g = {k: np.array([float(parts[i][k]) for i in idx]) for k in styles}
    N = np.asarray(d["R0"]).shape[0]
    print(f"== {tag}: N = {N}, {len(idx)}/{nNu} weights, nuMax = {nuMax:.4g}")
    viol = [(i, g['gLocal'][j], g['gGCD'][j], g['gGlobal'][j]) for j, i in enumerate(idx)
            if np.isfinite(g['gLocal'][j]) and not (g['gLocal'][j] <= g['gGCD'][j] + 1e-6 <= g['gGlobal'][j] + 2e-6)]
    print(f"   ordering local <= GCD <= global violated at {len(viol)} weights {viol[:3]}")
    for k, (lab, ls) in styles.items():
        ok = np.isfinite(g[k])
        if not ok.any(): continue
        env = np.min(g[k][ok][:, None] + nu[ok][:, None]*Qgrid[None, :], axis=0)
        env = np.clip(env, 0, 1)
        q0 = Qgrid[np.argmax(env > 0)]
        at = {Q: np.interp(Q, Qgrid, env) for Q in (50, 100, 150, 200, 300, 1000)}
        print(f"   {lab:6s}: max eta {g[k][ok][0]:.4f} (nu = {nu[ok][0]:.2e}); eta_ub = 0 below Q = {q0:.1f}; "
              + ", ".join(f"Q={Q}: {v:.3f}" for Q, v in at.items()))
        ax.semilogx(Qgrid, env, ls, color=colors[it], lw=1.4, label=f"N = {N}, {lab}")
ax.axvline(168.2, color="k", lw=0.8, ls="-.", label="lossless local $Q_\\mathrm{lb}$, N = 285")
ax.set_xlabel("Q (radiated + dissipated power)"); ax.set_ylabel(r"radiation efficiency $\eta$")
ax.set_ylim(0, 1); ax.set_xlim(Qgrid[0], Qgrid[-1]); ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=200)
print(f"saved {sys.argv[1]}")
