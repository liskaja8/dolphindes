r"""Greedy local search over RWG edges (removal and addition), closing the gap to the refined global GCD bound.

Copper 2:1 plate (patch21ka1, ka = 1 at 2.45 GHz), RWG basis, Z_mat = Zs L, the total current as the variable.

The start is the best structure of the gap rule in greedyRemoval.py (results/greedyRemoval_best.npz). The target is the global scattering
bound of the full plate, refined by GCD with `NgcdGlobal` iterations: it bounds every structure, so for the current edge set S the gap
D_global - P(S) is minimized by maximizing the realized scattered power P(S).

Moves, all scored in closed form from Z_SS^{-1} (the S-metal structure: Ztot_SS I = V_S):
- Removal of edge i in S: I' = I - (Z_SS^{-1})_{:,i} I_i / (Z_SS^{-1})_ii.
- Addition of edge j not in S (bordered system): with z = Z_Sj, zr = Z_jS, the Schur complement s = Z_jj - zr Z_SS^{-1} z gives
  I'_j = (V_j - zr I) / s and I'_S = I - Z_SS^{-1} z I'_j.
The best improving move is accepted and Z_SS^{-1} is updated by rank one: a downdate for a removal, a bordered update for an addition. The
search stops at a local optimum, where no single removal or addition increases P. The final structure is then certified by GCD on its own
edge set: no sub-structure scatters more than that bound.

Run from examples/limits (after greedyRemoval.py):  python greedyLocalSearch.py
"""
import contextlib
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters

matrixSet = "patch21ka1"
NgcdGlobal = 40 # GCD iterations of the refined global bound
NgcdLocal = 20 # GCD iterations of the certificate on the final edge set
maxMoves = 500
seed = 0

# ---------------------------------------------------------------- problem
def Msym(M):
    return 0.5*(M.conj().T + M)

mu_0, epsilon_0 = 4e-7*np.pi, 8.854e-12
c0, eta_0 = 1/np.sqrt(epsilon_0*mu_0), np.sqrt(mu_0/epsilon_0)
E0, frequency = 1e3, 2.45e9
omega = 2*np.pi*frequency
sigma = 5.96e7
Zs = (1 + 1j)/(sigma*np.sqrt(2/(omega*mu_0*sigma)))

Lmat = np.loadtxt(f"Lmat_{matrixSet}.txt", delimiter=',')
R0 = Msym(np.loadtxt(f"R0_{matrixSet}.txt", delimiter=','))
X0 = Msym(np.loadtxt(f"X0_{matrixSet}.txt", delimiter=','))
Vinc = E0*np.loadtxt(f"V_{matrixSet}.txt", delimiter=',').astype(complex)
Ndes = Lmat.shape[0]
Ztot = R0 + 1j*X0 + Zs*Lmat
Pdip = 3*(c0/frequency)**2/(2*np.pi)*E0**2/(2*eta_0)

def evalObj(I, R=R0):
    return 0.5*np.real(I.conj() @ R @ I)

opt_params = OptimizationHyperparameters(opttol=1e-8, gradConverge=True, min_inner_iter=10, max_restart=20,
                                         penalty_ratio=1e-2, penalty_reduction=0.1, break_iter_period=5, verbose=0)

def scatteringBound(S, Ngcd):
    # global bound and GCD for the structures made of the edges S (current variable)
    np.random.seed(seed)
    Plist = [sp.diags(q*np.ones(len(S), dtype=complex)) for q in [1, 1j]]
    Q = DenseSharedProjQCQP(-0.5*R0[np.ix_(S, S)], np.zeros(len(S), dtype=complex), 0.0, Ztot[np.ix_(S, S)].conj().T,
                            0.5*Vinc[S], Plist, verbose=0)
    Q.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
    Dg = Q.current_dual
    with contextlib.redirect_stdout(io.StringIO()):
        Q.run_gcd(GCDHyperparameters(max_proj_cstrt_num=2 + 2*Ngcd, max_gcd_iter_num=Ngcd, gcd_iter_period=Ngcd + 1, opt_params=opt_params))
    return Dg, Q.current_dual

# ---------------------------------------------------------------- refined global bound and the start
t1 = time.time()
Dglobal2, Dglobal = scatteringBound(np.arange(Ndes), NgcdGlobal)
print(f"full plate: global bound {Dglobal2:.4f} W, refined GCD {NgcdGlobal} bound {Dglobal:.4f} W ({time.time() - t1:.0f}s)", flush=True)

start = np.load("results/greedyRemoval_best.npz")
keep = start["gap_keep"].astype(bool)
print(f"start (greedyRemoval.py, gap rule): {keep.sum()} edges, P {float(start['gap_P']):.4f} W, own GCD bound {float(start['gap_Dgcd']):.4f} W "
      f"(full-plate GCD 20 there {float(start['fullPlateGCD']):.4f} W)", flush=True)

# ---------------------------------------------------------------- local search
S = np.flatnonzero(keep)
Zinv = np.linalg.inv(Ztot[np.ix_(S, S)])
I = Zinv @ Vinc[S]
P = evalObj(I, R0[np.ix_(S, S)])
history = [dict(n=len(S), P=P, move="start")]
t1 = time.time()
for move in range(maxMoves):
    R = R0[np.ix_(S, S)]
    # removals
    M = I[:, None] - Zinv*(I/np.diag(Zinv))[None, :]
    Prem = 0.5*np.real(np.einsum('ij,ik,kj->j', M.conj(), R, M))
    # additions
    out = np.setdiff1d(np.arange(Ndes), S)
    if len(out):
        Zc = Ztot[np.ix_(S, out)] # z = Z_Sj (columns)
        Zr = Ztot[np.ix_(out, S)] # zr = Z_jS (rows)
        Gz = Zinv @ Zc
        sch = np.diag(Ztot)[out] - np.einsum('ij,ji->i', Zr, Gz)
        Ij = (Vinc[out] - Zr @ I)/sch
        IS = I[:, None] - Gz*Ij[None, :] # new currents on S, one column per candidate
        Rss, RSj, Rjj = R, R0[np.ix_(S, out)], np.diag(R0)[out]
        Padd = 0.5*np.real(np.einsum('ij,ik,kj->j', IS.conj(), Rss, IS) + 2*np.real(np.conj(IS)*(RSj*Ij[None, :])).sum(axis=0)
                           + Rjj*np.abs(Ij)**2)
    else:
        Padd = np.array([])
    iBest = int(np.argmax(Prem))
    jBest = int(np.argmax(Padd)) if len(Padd) else -1
    bestRem, bestAdd = Prem[iBest], (Padd[jBest] if jBest >= 0 else -np.inf)
    if max(bestRem, bestAdd) <= P*(1 + 1e-12):
        break
    if bestRem >= bestAdd: # remove S[iBest]
        i = iBest
        I = np.delete(M[:, i], i)
        keepIdx = np.arange(len(S)) != i
        Zinv = Zinv[np.ix_(keepIdx, keepIdx)] - np.outer(Zinv[keepIdx, i], Zinv[i, keepIdx])/Zinv[i, i]
        history.append(dict(n=len(S) - 1, P=bestRem, move=f"remove {S[i]}"))
        S = S[keepIdx]
        P = bestRem
    else: # add out[jBest]
        j = jBest
        g, gr, s = Gz[:, j], Zr[j] @ Zinv, sch[j]
        Zinv = np.block([[Zinv + np.outer(g, gr)/s, -g[:, None]/s], [-gr[None, :]/s, np.array([[1/s]])]])
        I = np.r_[IS[:, j], Ij[j]]
        S = np.r_[S, out[j]]
        history.append(dict(n=len(S), P=bestAdd, move=f"add {out[j]}"))
        P = bestAdd
    # keep the ordering of S sorted (the inverse is permuted accordingly)
    order = np.argsort(S)
    S, I, Zinv = S[order], I[order], Zinv[np.ix_(order, order)]
nRem = sum(h["move"].startswith("remove") for h in history)
nAdd = sum(h["move"].startswith("add") for h in history)
# check the incremental updates against a direct solve
Idirect = np.linalg.solve(Ztot[np.ix_(S, S)], Vinc[S])
print(f"local search: {len(history) - 1} moves ({nRem} removals, {nAdd} additions) in {time.time() - t1:.1f}s; "
      f"local optimum {len(S)} edges, P {P:.4f} W; incremental vs direct current error {np.linalg.norm(I - Idirect)/np.linalg.norm(Idirect):.1e}",
      flush=True)

# ---------------------------------------------------------------- certificate
t1 = time.time()
Dloc2, Dloc = scatteringBound(S, NgcdLocal)
Pstart = float(start["gap_P"])
print(f"final structure: {len(S)} edges, P {P:.4f} W ({P/Pdip:.2f} x resonant dipole); "
      f"gap to the refined global bound {(Dglobal - P)/Dglobal:.3f} (start {(Dglobal - Pstart)/Dglobal:.3f}); "
      f"own GCD {NgcdLocal} bound {Dloc:.4f} W (global {Dloc2:.4f} W), gap to it {(Dloc - P)/Dloc:.3f} ({time.time() - t1:.0f}s)", flush=True)
np.savez("results/greedyLocalSearch_best.npz", keep=np.isin(np.arange(Ndes), S), P=P, Dglobal=Dglobal, Dlocal=Dloc)

# ---------------------------------------------------------------- figure
mesh = sio.loadmat(f"mesh_{matrixSet}.mat")
nodes, tria, bf = mesh["nodes"], mesh["triangles"], mesh["bf"]
edgeLen = np.linalg.norm(nodes[bf[:, 2]] - nodes[bf[:, 3]], axis=1)
e1, e2 = nodes[tria[:, 1]] - nodes[tria[:, 0]], nodes[tria[:, 2]] - nodes[tria[:, 0]]
triArea = 0.5*np.abs(e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0])
triCent = nodes[tria].mean(axis=1)
freeNode = np.array([[np.setdiff1d(tria[t], bf[n, 2:])[0] for t in bf[n, :2]] for n in range(Ndes)])

def currentDensity(Ifull):
    J = np.zeros((len(tria), 2), dtype=complex)
    for s_, sgn in [(0, 1), (1, -1)]:
        rho = sgn*(triCent[bf[:, s_]] - nodes[freeNode[:, s_]])
        np.add.at(J, bf[:, s_], (Ifull*edgeLen/(2*triArea[bf[:, s_]]))[:, None]*rho)
    return J

def structurePlot(ax, keepMask, Ifull, title):
    metal = np.zeros(len(tria), dtype=bool)
    metal[bf[keepMask, :2].ravel()] = True
    J = np.linalg.norm(np.abs(currentDensity(Ifull)), axis=1)
    pc = PolyCollection(nodes[tria[metal]], array=np.maximum(J[metal], 1e-3*J.max()), cmap='viridis',
                        norm=LogNorm(1e-3*J.max(), J.max()), edgecolors='none')
    ax.add_collection(pc)
    ax.add_collection(PolyCollection(nodes[tria[~metal]], facecolors='white', edgecolors='0.85', linewidths=0.3))
    slot = ~keepMask & metal[bf[:, 0]] & metal[bf[:, 1]]
    ax.add_collection(LineCollection(nodes[bf[slot, 2:]], colors='r', linewidths=1.5))
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    return pc

def fullCurrent(keepMask):
    Sk = np.flatnonzero(keepMask)
    If = np.zeros(Ndes, dtype=complex)
    If[Sk] = np.linalg.solve(Ztot[np.ix_(Sk, Sk)], Vinc[Sk])
    return If

fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(2, 2, (1, 2))
ax.plot([h["P"] for h in history], 'o-', ms=3, label='realized P(S)')
for k, h in enumerate(history[1:], 1):
    ax.plot(k, h["P"], 'v' if h["move"].startswith("remove") else '^', c='tab:red' if h["move"].startswith("remove") else 'tab:green', ms=5)
ax.axhline(Dglobal, c='k', lw=0.8, label=f'refined global GCD {NgcdGlobal} bound {Dglobal:.2f} W')
ax.axhline(Dloc, c='tab:purple', ls='--', lw=0.8, label=f'GCD bound on the final edge set {Dloc:.2f} W')
ax.axhline(Pdip, c='gray', ls=':', label='resonant dipole')
ax.set_xlabel('move (red: removal, green: addition)'); ax.set_ylabel('W'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
keep0 = start["gap_keep"].astype(bool)
keep1 = np.isin(np.arange(Ndes), S)
for col, (km, title) in enumerate([(keep0, f"start: {keep0.sum()} edges, P = {Pstart:.2f} W ({Pstart/Dglobal:.2f} of global bound)"),
                                   (keep1, f"local optimum: {keep1.sum()} edges, P = {P:.2f} W ({P/Dglobal:.2f} of global bound)")]):
    ax = fig.add_subplot(2, 2, 3 + col)
    pc = structurePlot(ax, km, fullCurrent(km), title)
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|J| (A/m)')
plt.tight_layout()
out = "results/greedyLocalSearch.png"
plt.savefig(out, dpi=110)
print(f"figure: {out}")
