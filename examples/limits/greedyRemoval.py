r"""Greedy RWG edge removal guided by the scattering dual, with closed-form scores and rank-one downdates.

Copper 2:1 plate (patch21ka1, ka = 1 at 2.45 GHz), RWG basis, Z_mat = Zs L, the total current I as the variable
(A1 = Ztot^H, A2 = 1, s1 = V/2, as in maxScatIvecNoChol.ipynb).

For the present edge set S:
- Bound. The scattering QCQP restricted to S has the Lagrangian L_S = Sym A(lambda) (n_S x n_S) and S_S(lambda) = sum_k lambda_k P_k^H s1, so
  D_lambda(S) = S_S^H L_S^{-1} S_S. Deleting edge i deletes row and column i. A principal submatrix of a PSD matrix is PSD, so lambda
  stays dual feasible, and the bound of S \ {i} at the same multipliers is exact in closed form:
  D_lambda(S \ {i}) = D_lambda(S) - |I*_i|^2 / (L_S^{-1})_ii,  I* = L_S^{-1} S_S.
  Every bound along the path is therefore valid, and the multipliers are re-optimized every `reoptEvery` steps (warm-started dual
  Newton on the GCD projectors restricted to S).
- Realized structure. P(S) is the scattered power of the S-metal structure, Ztot_SS I = V_S. Deleting basis function i updates the solution
  exactly: I' = I - (Z_SS^{-1})_{:,i} I_i / (Z_SS^{-1})_ii, so P(S \ {i}) of all candidates costs one matrix product.
- Downdates. Both inverses are downdated by rank one after each removal: delete row and column i of A^{-1}, subtract b c / beta.

Rule "realized": remove the edge whose removal gives the largest realized P. Rule "gap": remove the edge that minimizes the gap
D_lambda(S \ {i}) - P(S \ {i}). Rule "gap+add": a local search from the best structure of the "gap" path that accepts a removal or an
addition only if it lowers the gap (additions scored at fixed multipliers by the bordered system, valid if L_{S+j} stays PSD), and stops
at a local minimum after one more re-optimization of the multipliers. All three are run. The best structure on each path is reported with its bound from GCD on its own edge set.

Run from examples/limits:  python greedyRemoval.py
"""
import contextlib
import copy
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
Ngcd = 20 # GCD iterations of the initial (full plate) bound and of the final certificate
reoptEvery = 5 # re-optimize the multipliers on S every so many removals
minEdges = 20 # stop the removal path here
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
gcd_params = GCDHyperparameters(max_proj_cstrt_num=2 + 2*Ngcd, max_gcd_iter_num=Ngcd, gcd_iter_period=Ngcd + 1, opt_params=opt_params)

def restrictedQCQP(S, Pcols):
    # scattering QCQP on the edge set S with the diagonal projectors Pcols (columns of length Ndes) restricted to S
    Pd = Pcols[S]
    keep = np.linalg.norm(Pd, axis=0) > 1e-12
    Plist = [sp.diags(Pd[:, k]) for k in np.flatnonzero(keep)]
    Q = DenseSharedProjQCQP(-0.5*R0[np.ix_(S, S)], np.zeros(len(S), dtype=complex), 0.0, Ztot[np.ix_(S, S)].conj().T,
                            0.5*Vinc[S], Plist, verbose=0)
    return Q, keep

def lagrangian(Q):
    lags = Q.current_lags
    return Msym(Q._get_total_A(lags)), Q._get_total_S(lags)

def downdate(Ainv, i):
    # inverse of A with row and column i deleted, from A^{-1}
    keep = np.arange(Ainv.shape[0]) != i
    return Ainv[np.ix_(keep, keep)] - np.outer(Ainv[keep, i], Ainv[i, keep])/Ainv[i, i]

# ---------------------------------------------------------------- initial bound on the full plate
np.random.seed(seed)
Sall = np.arange(Ndes)
Q0, _ = restrictedQCQP(Sall, np.column_stack([np.ones(Ndes), 1j*np.ones(Ndes)]).astype(complex))
Q0.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
Dglob = Q0.current_dual
t1 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    Q0.run_gcd(gcd_params)
Pcols = Q0.Proj.get_Pdata_column_stack().copy() # GCD projectors (Ndes x k), reused on every S
L0, S0 = lagrangian(Q0)
lam0 = np.linalg.eigvalsh(L0)
x0 = np.linalg.solve(L0, S0)
print(f"full plate: global bound {Dglob:.4f} W, GCD {Ngcd} bound {Q0.current_dual:.4f} W ({time.time() - t1:.0f}s, {Pcols.shape[1]} projectors), "
      f"S^H L^-1 S = {np.real(S0.conj() @ x0):.4f} W (formula check), min/max eig L {lam0[0]/lam0[-1]:.1e}", flush=True)

Ifull = np.linalg.solve(Ztot, Vinc)
print(f"full plate structure: P {evalObj(Ifull):.4e} W", flush=True)

# ---------------------------------------------------------------- greedy removal (and addition)
def fullLagrangian(lagsFull):
    # Lagrangian and linear term of the full plate at the multipliers of the GCD projectors (zeros for dropped ones)
    return Msym(Q0._get_total_A(lagsFull)), Q0._get_total_S(lagsFull)

def reoptimize(S, lags):
    # re-optimize the multipliers on S, warm start from lags (feasible: L_S(lags) is a principal submatrix of the PSD full Lagrangian)
    Q, keep = restrictedQCQP(S, Pcols)
    Q.solve_current_dual_problem(method='newton', init_lags=lags[keep], opt_params=opt_params)
    Lr, Sr = lagrangian(Q)
    LinvR = np.linalg.inv(Lr)
    lagsNew = np.zeros(len(keep)); lagsNew[keep] = Q.current_lags
    return np.real(Sr.conj() @ LinvR @ Sr), LinvR, Sr, lagsNew

def greedy(rule, maxMoves=600, tabu=10, startS=None, improvingOnly=False):
    # rule 'realized': remove the edge with the largest realized P; 'gap': remove the edge with the smallest gap D - P;
    # 'gap+add': the removal or addition with the smallest gap. Additions are scored at fixed multipliers by the bordered system:
    # with l = L_Sj, s = L_jj - l^H L_S^{-1} l > 0 (L_{S+j} PSD, so the bound stays valid) the bound grows by
    # |S_j - l^H I*|^2 / s; candidates with s <= 0 would make the multipliers infeasible and are skipped. Recently moved edges are tabu.
    # startS: start from this edge set (multipliers re-optimized there); improvingOnly: accept a move only if it lowers the gap, and when
    # none does, re-optimize the multipliers once more and stop if the gap still cannot be lowered (local minimum).
    allowAdd = rule == "gap+add"
    lags = Q0.current_lags.copy()
    if startS is None:
        S = Sall.copy()
        Lfull, Sfull = L0.copy(), S0.copy()
        Linv = np.linalg.inv(Lfull)
        Svec = Sfull.copy()
        D = np.real(Svec.conj() @ Linv @ Svec)
    else:
        S = np.sort(startS)
        D, Linv, Svec, lags = reoptimize(S, lags)
        Lfull, Sfull = fullLagrangian(lags)
    xs = Linv @ Svec
    Zinv = np.linalg.inv(Ztot[np.ix_(S, S)])
    I = Zinv @ Vinc[S]
    path = [dict(n=len(S), D=D, P=evalObj(I, R0[np.ix_(S, S)]), S=S.copy(), I=I.copy(), move="start")]
    recent = []
    t1 = time.time()
    nRem = nAdd = 0
    justReopt = False
    while len(S) > minEdges and nRem + nAdd < maxMoves:
        R = R0[np.ix_(S, S)]
        # removals: bounds at fixed multipliers (exact, valid) and realized structures
        Drem = D - np.abs(xs)**2/np.real(np.diag(Linv))
        M = I[:, None] - Zinv*(I/np.diag(Zinv))[None, :]
        Prem = 0.5*np.real(np.einsum('ij,ik,kj->j', M.conj(), R, M))
        scoreRem = Prem if rule == "realized" else -(Drem - Prem) # larger is better
        scoreRem = np.where(np.isin(S, recent), -np.inf, scoreRem)
        bestRem = int(np.argmax(scoreRem))
        bestAdd, scoreAdd = -1, -np.inf
        if allowAdd:
            out = np.setdiff1d(np.arange(Ndes), S)
            if len(out):
                Lc = Lfull[np.ix_(S, out)]
                G = Linv @ Lc
                sch = np.real(np.diag(Lfull)[out] - np.einsum('ij,ij->j', Lc.conj(), G))
                rho = Sfull[out] - Lc.conj().T @ xs
                ok = sch > 1e-14*np.abs(np.diag(Lfull)).max()
                Dadd = np.where(ok, D + np.abs(rho)**2/np.where(ok, sch, 1.0), np.inf)
                Zc, Zr = Ztot[np.ix_(S, out)], Ztot[np.ix_(out, S)]
                Gz = Zinv @ Zc
                zs = np.diag(Ztot)[out] - np.einsum('ij,ji->i', Zr, Gz)
                Ij = (Vinc[out] - Zr @ I)/zs
                IS = I[:, None] - Gz*Ij[None, :]
                Padd = 0.5*np.real(np.einsum('ij,ik,kj->j', IS.conj(), R, IS)
                                   + 2*np.real(np.conj(IS)*(R0[np.ix_(S, out)]*Ij[None, :])).sum(axis=0) + np.diag(R0)[out]*np.abs(Ij)**2)
                sAdd = np.where(np.isin(out, recent) | ~ok, -np.inf, -(Dadd - Padd))
                bestAdd = int(np.argmax(sAdd))
                scoreAdd = sAdd[bestAdd]
        if improvingOnly:
            gapNow = D - path[-1]["P"]
            if max(scoreAdd, scoreRem[bestRem]) <= -gapNow*(1 - 1e-12): # no move lowers the gap
                if justReopt:
                    break # local minimum
                try:
                    Dnew, LinvR, Sr, lagsNew = reoptimize(S, lags)
                    if Dnew < D:
                        D, Linv, Svec, lags = Dnew, LinvR, Sr, lagsNew
                        Lfull, Sfull = fullLagrangian(lags)
                        xs = Linv @ Svec
                        path[-1]["D"] = D
                except Exception as err:
                    print(f"    re-optimization failed at |S| = {len(S)} ({err})")
                justReopt = True
                continue
            justReopt = False
        if allowAdd and scoreAdd > scoreRem[bestRem]: # add out[bestAdd]: bordered updates
            j = bestAdd
            g, s_ = G[:, j], sch[j]
            Linv = np.block([[Linv + np.outer(g, g.conj())/s_, -g[:, None]/s_], [-g.conj()[None, :]/s_, np.array([[1/s_]])]])
            gz, gr, zz = Gz[:, j], Zr[j] @ Zinv, zs[j]
            Zinv = np.block([[Zinv + np.outer(gz, gr)/zz, -gz[:, None]/zz], [-gr[None, :]/zz, np.array([[1/zz]])]])
            D = Dadd[j]
            I = np.r_[IS[:, j], Ij[j]]
            Svec = np.r_[Svec, Sfull[out[j]]]
            S = np.r_[S, out[j]]
            moved, nAdd = out[j], nAdd + 1
            move = f"add {moved}"
        else: # remove S[bestRem]: downdates
            i = bestRem
            D = Drem[i]
            I = np.delete(M[:, i], i)
            Zinv = downdate(Zinv, i)
            Linv = downdate(Linv, i)
            Svec = np.delete(Svec, i)
            moved, nRem = S[i], nRem + 1
            S = np.delete(S, i)
            move = f"remove {moved}"
        recent = (recent + [moved])[-tabu:]
        if (nRem + nAdd) % reoptEvery == 0:
            # tighten: re-optimize the multipliers on S, warm start (the old multipliers stay feasible)
            try:
                Dnew, LinvR, Sr, lagsNew = reoptimize(S, lags)
                if Dnew <= D: # keep the fixed-multiplier bound if the re-solve did not improve it
                    D, Linv, Svec, lags = Dnew, LinvR, Sr, lagsNew
                    Lfull, Sfull = fullLagrangian(lags)
            except Exception as err:
                print(f"    re-optimization failed at |S| = {len(S)} ({err}); keeping the fixed multipliers")
        xs = Linv @ Svec
        path.append(dict(n=len(S), D=D, P=evalObj(I, R0[np.ix_(S, S)]), S=S.copy(), I=I.copy(), move=move))
    print(f"rule '{rule}': {nRem} removals, {nAdd} additions in {time.time() - t1:.1f}s", flush=True)
    return path

paths = {rule: greedy(rule) for rule in ["realized", "gap"]}
gapPath = paths["gap"]
startGap = gapPath[int(np.argmax([p["P"] for p in gapPath]))]["S"] # best structure of the removal-only gap path
paths["gap+add"] = greedy("gap+add", startS=startGap, improvingOnly=True)

# ---------------------------------------------------------------- best structures, certified by GCD on their own edge set
best = {}
for rule, path in paths.items():
    k = int(np.argmax([p["P"] for p in path]))
    p = path[k]
    np.random.seed(seed)
    Qb, _ = restrictedQCQP(p["S"], np.column_stack([np.ones(Ndes), 1j*np.ones(Ndes)]).astype(complex))
    Qb.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
    with contextlib.redirect_stdout(io.StringIO()):
        Qb.run_gcd(gcd_params)
    Ib = np.zeros(Ndes, dtype=complex); Ib[p["S"]] = p["I"]
    lamb = np.linalg.eigvalsh(Msym(Qb._get_total_A(Qb.current_lags)))
    best[rule] = dict(p=p, I=Ib, Dgcd=Qb.current_dual, lamRatio=lamb[0]/lamb[-1])
    print(f"rule '{rule}': best structure {p['n']} edges ({Ndes - p['n']} removed), P {p['P']:.4f} W = {p['P']/Q0.current_dual:.3f} of the "
          f"full-plate GCD bound ({p['P']/Pdip:.2f} x resonant dipole, {p['P']/evalObj(Ifull):.0f} x full plate); path bound at that set "
          f"{p['D']:.4f} W; GCD bound on its own edge set {Qb.current_dual:.4f} W, gap {(Qb.current_dual - p['P'])/Qb.current_dual:.3f}, "
          f"min/max eig {lamb[0]/lamb[-1]:.1e}", flush=True)

np.savez("results/greedyRemoval_best.npz", fullPlateGCD=Q0.current_dual,
         **{f"{rule.replace('+', '_')}_keep": np.isin(np.arange(Ndes), b["p"]["S"]) for rule, b in best.items()},
         **{f"{rule.replace('+', '_')}_P": b["p"]["P"] for rule, b in best.items()}, **{f"{rule.replace('+', '_')}_Dgcd": b["Dgcd"] for rule, b in best.items()})
print("best structures saved: results/greedyRemoval_best.npz")

# ---------------------------------------------------------------- figure
mesh = sio.loadmat(f"mesh_{matrixSet}.mat")
nodes, tria, bf = mesh["nodes"], mesh["triangles"], mesh["bf"]
edgeLen = np.linalg.norm(nodes[bf[:, 2]] - nodes[bf[:, 3]], axis=1)
e1, e2 = nodes[tria[:, 1]] - nodes[tria[:, 0]], nodes[tria[:, 2]] - nodes[tria[:, 0]]
triArea = 0.5*np.abs(e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0])
triCent = nodes[tria].mean(axis=1)
freeNode = np.array([[np.setdiff1d(tria[t], bf[n, 2:])[0] for t in bf[n, :2]] for n in range(Ndes)])

def currentDensity(I):
    J = np.zeros((len(tria), 2), dtype=complex)
    for s, sgn in [(0, 1), (1, -1)]:
        rho = sgn*(triCent[bf[:, s]] - nodes[freeNode[:, s]])
        np.add.at(J, bf[:, s], (I*edgeLen/(2*triArea[bf[:, s]]))[:, None]*rho)
    return J

def structurePlot(ax, keep, I, title):
    metal = np.zeros(len(tria), dtype=bool)
    metal[bf[keep, :2].ravel()] = True
    J = np.linalg.norm(np.abs(currentDensity(I)), axis=1)
    pc = PolyCollection(nodes[tria[metal]], array=np.maximum(J[metal], 1e-3*J.max()), cmap='viridis',
                        norm=LogNorm(1e-3*J.max(), J.max()), edgecolors='none')
    ax.add_collection(pc)
    ax.add_collection(PolyCollection(nodes[tria[~metal]], facecolors='white', edgecolors='0.85', linewidths=0.3))
    slot = ~keep & metal[bf[:, 0]] & metal[bf[:, 1]]
    ax.add_collection(LineCollection(nodes[bf[slot, 2:]], colors='r', linewidths=1.5))
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    return pc

fig = plt.figure(figsize=(19, 8))
ax = fig.add_subplot(2, 3, (1, 2))
for rule, path in paths.items():
    ax.semilogy([p["P"] for p in path], '-', label=f"'{rule}': realized P(S)")
    ax.semilogy([p["D"] for p in path], '--', label=f"'{rule}': bound D(S) (fixed / re-optimized multipliers)")
ax.axhline(Q0.current_dual, c='k', lw=0.8, label='full-plate GCD bound'); ax.axhline(Pdip, c='gray', ls=':', label='resonant dipole')
ax.set_xlabel('move'); ax.set_ylabel('W'); ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)
ax = fig.add_subplot(2, 3, 3)
for rule, path in paths.items():
    ax.semilogy([max((p["D"] - p["P"])/p["D"], 1e-6) for p in path], '-', label=f"'{rule}'")
ax.set_xlabel('move'); ax.set_ylabel('relative gap (D - P)/D'); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
for col, (rule, b) in enumerate(best.items()):
    keep = np.zeros(Ndes, dtype=bool); keep[b["p"]["S"]] = True
    ax = fig.add_subplot(2, 3, 4 + col)
    pc = structurePlot(ax, keep, b["I"], f"'{rule}': {b['p']['n']} edges, P = {b['p']['P']:.2f} W "
                       f"({b['p']['P']/Q0.current_dual:.2f} of bound)\nown GCD bound {b['Dgcd']:.2f} W, gap {(b['Dgcd'] - b['p']['P'])/b['Dgcd']:.2f}")
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|J| (A/m)')
plt.tight_layout()
out = "results/greedyRemoval.png"
plt.savefig(out, dpi=110)
print(f"figure: {out}")
