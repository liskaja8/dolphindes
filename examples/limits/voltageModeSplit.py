"""Scattering bound in the significant voltage modes, then a QCQP fit of the free (reactive) voltage for pointwise power balance.

Copper 2:1 plate (patch21ka1, ka = 1 at 2.45 GHz), RWG basis, Z_mat = Zs L, scattered voltage x = V_sca = -Z0 I as the variable.

1. Mode split. H = W^H (Re(Zs) L + R0) W, W = Z0^{-1}, is the loss + radiation operator in the voltage variable. It is also the
   matrix of the real power balance constraint. Its eigenvectors split into
   - significant modes S (eigenvalue > modeTol * max), which carry current, loss and radiation;
   - the plateau P, strongly reactive modes that drive almost no current.
   The near-kernel of the scattering Lagrangian lies in P.
2. Stage 1. The global bound (p = 1, j) and GCD are solved for x = S a only, with the library projector constraints and the reduced
   A1 = S^H A1 and A2 = W S. This problem is checked for strong duality: the bound must equal the attained P, and the residuals must
   vanish.
3. Stage 2. The free voltage x = S a* + P b is fitted to pointwise power balance w_i = conj(I_i) r_i = 0, r = V - Ztot I.
   - The current is frozen at I_a = -W S a*, because P carries almost no current. Then w is affine in b, and sum_i |w_i|^2 is a positive
     definite quadratic in b.
   - The global power balance is kept exactly, with the current not frozen, as two quadratic constraints in b.
   - The result is a QCQP with a positive definite objective and two constraints, solved by its dual from lambda = 0.
4. Pointwise inference (impedance and admittance forms, binary rounding, relaxed designs) of the stage-1 and stage-2 currents. Then a
   figure with the mode spectrum, the complex density and the binary structures.

Run from examples/limits:  python voltageModeSplit.py
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
modeTol = 1e-5 # significant modes: eig(H) > modeTol * max eig(H); 1e-5 keeps the transition modes the reactive balance needs
Ngcd = 20 # GCD iterations in stage 1
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
Zmat, Z0 = Zs*Lmat, R0 + 1j*X0
Ztot, Ldiag = Z0 + Zmat, np.diag(Lmat).copy()
W = np.linalg.inv(Z0)
WH = W.conj().T

def evalObj(I):
    return 0.5*np.real(I.conj() @ R0 @ I)

Ifull = np.linalg.solve(Ztot, Vinc)
Pfull = evalObj(Ifull)
Pdip = 3*(c0/frequency)**2/(2*np.pi)*E0**2/(2*eta_0)

Obj = -0.5*Msym(WH @ R0 @ W) # maximize -x^H Obj x = P
A1 = np.eye(Ndes) + WH @ Zmat.conj().T
A2 = W
s1 = -0.5*Vinc
PlistGlob = [sp.diags(q*np.ones(Ndes, dtype=complex)) for q in [1, 1j]]
C = np.eye(Ndes) + Zmat @ W # r = V - Ztot I = V + C x

opt_params = OptimizationHyperparameters(opttol=1e-10, gradConverge=True, min_inner_iter=10, max_restart=40,
                                         penalty_ratio=1e-2, penalty_reduction=0.1, break_iter_period=5, verbose=0)
gcd_params = GCDHyperparameters(max_proj_cstrt_num=2 + 2*Ngcd, max_gcd_iter_num=Ngcd, gcd_iter_period=Ngcd + 1,
                                opt_params=opt_params)

def VtoI(x):
    return -W @ x

def powerTerms(x):
    I = VtoI(x)
    r = Vinc - Ztot @ I
    return I, r, np.conj(I)*r

def constraintRes(Q, z):
    res = np.real(-z.conj() @ Q._allA_at_v(z) + 2*z.conj() @ Q.Fs)
    res[Q.n_proj_constr:] += Q.c_2j
    return res

def lagSpectrum(Q):
    return np.linalg.eigvalsh(Msym(Q._get_total_A(Q.current_lags)))

# ---------------------------------------------------------------- 1. mode split
H = Msym(WH @ (Zs.real*Lmat + R0) @ W)
lamH, UH = np.linalg.eigh(H)
sig_ = lamH > modeTol*lamH[-1]
S, Pmodes = UH[:, sig_], UH[:, ~sig_]
m, K = S.shape[1], Pmodes.shape[1]
print(f"mode split: {m} significant modes (eig H > {modeTol:.0e} max), {K} plateau modes; "
      f"current carried by the plateau |W P| / |W S| = {np.linalg.norm(W @ Pmodes, 2)/np.linalg.norm(W @ S, 2):.1e}")

# reference: full-space global and GCD scattering bounds
np.random.seed(seed)
Qfull = DenseSharedProjQCQP(Obj, np.zeros(Ndes, dtype=complex), 0.0, A1, s1, PlistGlob, A2=A2, verbose=0)
Qfull.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
print(f"full space, global: D {Qfull.current_dual:.4f} W ({Qfull.current_dual/Pdip:.2f} x resonant dipole)")

# ---------------------------------------------------------------- 2. stage 1: significant modes only
def stage1Report(name, Q):
    a = Q.current_xstar
    x = S @ a
    I = VtoI(x)
    lam = lagSpectrum(Q)
    res = constraintRes(Q, a)
    P = evalObj(I)
    print(f"{name}: {Q.n_proj_constr} constraints, D {Q.current_dual:.4f} W, P(x*) {P:.4f} W (P/D {P/Q.current_dual:.6f}), "
          f"min/max eig {lam[0]/lam[-1]:.1e}, ||c||/D {np.linalg.norm(res)/Q.current_dual:.1e}, "
          f"global residuals/D {constraintRes(Qfull, x)/Q.current_dual}")
    return dict(a=a, x=x, D=Q.current_dual, P=P, lamRatio=lam[0]/lam[-1], strong=abs(P/Q.current_dual - 1) < 1e-4)

np.random.seed(seed)
Q1 = DenseSharedProjQCQP(Msym(S.conj().T @ Obj @ S), np.zeros(m, dtype=complex), 0.0, S.conj().T @ A1, s1, PlistGlob,
                         A2=A2 @ S, verbose=0)
Q1.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
stage1 = {"global": stage1Report("stage 1, global", Q1)}
Q1g = copy.deepcopy(Q1)
t1 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    Q1g.run_gcd(gcd_params)
stage1["GCD"] = stage1Report(f"stage 1, GCD {Ngcd} ({time.time() - t1:.0f}s)", Q1g)

# ---------------------------------------------------------------- 3. stage 2: free voltage in the plateau
def shiftForm(A, s, c, x0, B):
    # -x^H A x + 2 Re(x^H s) + c at x = x0 + B b, as (A, s, c) in b
    Ax0 = A @ x0
    return Msym(B.conj().T @ A @ B), B.conj().T @ (s - Ax0), float(np.real(-x0.conj() @ Ax0 + 2*x0.conj() @ s) + c)

def projectorForm(p, x0, B):
    # Re(-x^H A1 P A2 x + 2 x^H A2^H P^H s1) at x = x0 + B b (exact, current not frozen)
    return shiftForm(Msym(A1 @ (p[:, None]*A2)), A2.conj().T @ (np.conj(p)*s1), 0.0, x0, B)

def generalQCQP(objective, cons):
    A0, s0, cc = objective
    n = len(s0)
    return DenseSharedProjQCQP(A0, s0, cc, np.eye(n, dtype=complex), np.zeros(n, dtype=complex), [], Pstruct=sp.eye(n),
                               B_j=[c_[0] for c_ in cons], s_2j=[c_[1] for c_ in cons], c_2j=np.array([c_[2] for c_ in cons]),
                               verbose=0)

class EvalCap(Exception):
    pass

def solveCapped(Q, lags0, maxEval=3000):
    # the library optimizer has no iteration limit; an infeasible primal makes the dual unbounded below, so get_dual is capped at
    # maxEval calls and the multipliers with the lowest dual seen are kept. Returns whether the cap was hit.
    best = [Q.get_dual(lags0)[0], np.array(lags0, dtype=float).copy()]
    count = [0]
    getDual = Q.get_dual
    def cappedDual(l, *args, **kwargs):
        count[0] += 1
        if count[0] > maxEval:
            raise EvalCap
        out = getDual(l, *args, **kwargs)
        D = out[0] if not kwargs.get("penalty_vectors") else getDual(l)[0]
        if D < best[0]:
            best[0], best[1] = D, np.array(l, dtype=float).copy()
        return out
    Q.get_dual = cappedDual
    capped = False
    try:
        Q.solve_current_dual_problem(method='newton', init_lags=lags0, opt_params=opt_params)
    except EvalCap:
        capped = True
    finally:
        del Q.get_dual
    if capped or Q.current_dual > best[0]:
        Q.current_lags = best[1]
        Q.current_dual = Q.get_dual(best[1])[0]
        Q.current_xstar = Q._get_xstar(best[1])[0]
    return capped

def stage2(x0):
    # frozen current I_a: w_i(b) = conj(I_a,i) (r0 + C P b)_i, minimize sum |w_i|^2 under the exact global power balance
    Ia = VtoI(x0)
    r0 = Vinc + C @ x0
    G = np.conj(Ia)[:, None]*(C @ Pmodes) # d w / d b
    g0 = np.conj(Ia)*r0 # w at b = 0
    scale = np.linalg.norm(g0)**2
    objective = (Msym(G.conj().T @ G)/scale, -(G.conj().T @ g0)/scale, -scale/scale) # -|g0 + G b|^2 / |g0|^2
    cons = [projectorForm(p, x0, Pmodes) for p in [np.ones(Ndes, dtype=complex), 1j*np.ones(Ndes, dtype=complex)]]
    Q = generalQCQP(objective, cons)
    capped = solveCapped(Q, np.zeros(2))
    b = Q.current_xstar
    bFree = np.linalg.lstsq(G, -g0, rcond=None)[0] # unconstrained least squares, for comparison
    lam = lagSpectrum(Q)
    fit = np.real(b.conj() @ objective[0] @ b - 2*b.conj() @ objective[1]) - objective[2] # attained objective
    return dict(Q=Q, b=b, x=x0 + Pmodes @ b, xFree=x0 + Pmodes @ bFree, bound=-Q.current_dual, fit=fit, lamRatio=lam[0]/lam[-1], capped=capped,
                frozenErr=np.linalg.norm(W @ Pmodes @ b)/np.linalg.norm(Ia),
                res=np.linalg.norm(constraintRes(Q, b)))

def wStats(x):
    I, r, w = powerTerms(x)
    return np.sum(np.abs(w))/np.abs(I.conj() @ Vinc), np.abs(w).max()/np.abs(I.conj() @ Vinc)

results = {}
for name, r1 in stage1.items():
    t1 = time.time()
    r2 = stage2(r1["x"])
    results[name] = (r1, r2)
    w0, wm0 = wStats(r1["x"])
    w1, wm1 = wStats(r2["x"])
    wf, _ = wStats(r2["xFree"])
    I2 = VtoI(r2["x"])
    print(f"stage 2 from stage-1 {name} ({time.time() - t1:.1f}s): plateau dim {K}, fit bound / attained "
          f"{r2['bound']/r2['fit']:.6f}{' (capped: dual unbounded, fit infeasible)' if r2['capped'] else ''}, min/max eig {r2['lamRatio']:.1e}, "
          f"||c||_QCQP {r2['res']:.1e}\n"
          f"    frozen-current error |W P b| / |I_a| {r2['frozenErr']:.1e}; sum|w|/|I^H V|: stage 1 {w0:.2e} -> stage 2 {w1:.2e} "
          f"(unconstrained least squares {wf:.2e}); max|w_i|/|I^H V| {wm0:.1e} -> {wm1:.1e}; "
          f"P {evalObj(VtoI(r1['x'])):.4f} -> {evalObj(I2):.4f} W; global residuals/D {constraintRes(Qfull, r2['x'])/r1['D']}")

# ---------------------------------------------------------------- 4. pointwise inference
dMin = 1e-6
thsLog = np.logspace(-6, 0, 61)[:-1]

def complexDensity(I):
    u = Vinc - Ztot @ I + Zs*Ldiag*I
    return Zs*Ldiag*I/u

def inferDimp(I):
    nz = I != 0
    rr = np.zeros(Ndes, dtype=complex)
    rr[nz] = (Vinc - Ztot @ I)[nz]/I[nz]
    t = np.maximum(0, np.real(np.conj(Zs)*rr)/(np.abs(Zs)**2*Ldiag))
    return np.where(nz, 1/(1 + t), 0.0)

def inferDadm(I):
    return np.clip(complexDensity(I).real, 0, 1)

def realizedCurrent(d):
    keep = d >= dMin
    I = np.zeros(Ndes, dtype=complex)
    Zd = Ztot + np.diag(Zs*Ldiag*(1/np.maximum(d, dMin) - 1))
    I[keep] = np.linalg.solve(Zd[np.ix_(keep, keep)], Vinc[keep])
    return I

def bestBinary(d):
    designs = [d > t for t in thsLog]
    Is = [realizedCurrent(k.astype(float)) for k in designs]
    Pb = np.array([evalObj(I) for I in Is])
    k = int(np.argmax(Pb))
    return dict(keep=designs[k], I=Is[k], P=Pb[k])

final = {"full-space global x*": Qfull.current_xstar}
for name, (r1, r2) in results.items():
    final[f"stage 1 {name}"] = r1["x"]
    final[f"stage 2 from {name}"] = r2["x"]
inference = {}
Dref = Qfull.current_dual
for name, x in final.items():
    I = VtoI(x)
    d = complexDensity(I)
    out = dict(d=d)
    for form, infer in [("impedance", inferDimp), ("admittance", inferDadm)]:
        dd = infer(I)
        out[form] = dict(relaxed=evalObj(realizedCurrent(dd)), **bestBinary(dd))
    inference[name] = out
    print(f"{name:26s}: |d| median/max {np.median(np.abs(d)):.1e}/{np.abs(d).max():.1e}, |d - 1| < 0.1: {np.count_nonzero(np.abs(d - 1) < 0.1):3d}; "
          + "; ".join(f"{f}: relaxed {out[f]['relaxed']/Dref:.4f}, binary {out[f]['P']/Dref:.4f} ({out[f]['P']:.3e} W)" for f in ["impedance", "admittance"]))
print(f"(relative to the full-space global bound {Dref:.4f} W; full plate {Pfull:.3e} W)")

# ---------------------------------------------------------------- figure
mesh = sio.loadmat(f"mesh_{matrixSet}.mat")
nodes, tria, bf = mesh["nodes"], mesh["triangles"], mesh["bf"]

def edgePlot(ax, values, title, cmap='RdBu_r', clim=None):
    lc = LineCollection(nodes[bf[:, 2:]], array=values, cmap=cmap, linewidths=2.5)
    if clim is not None:
        lc.set_clim(*clim)
    ax.add_collection(lc)
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    plt.colorbar(lc, ax=ax, shrink=0.8, format='%.0e')

def structurePlot(ax, keep, title):
    metal = np.zeros(len(tria), dtype=bool)
    metal[bf[keep, :2].ravel()] = True
    ax.add_collection(PolyCollection(nodes[tria[metal]], facecolors='0.4', edgecolors='none'))
    ax.add_collection(PolyCollection(nodes[tria[~metal]], facecolors='white', edgecolors='0.85', linewidths=0.3))
    slot = ~keep & metal[bf[:, 0]] & metal[bf[:, 1]]
    ax.add_collection(LineCollection(nodes[bf[slot, 2:]], colors='r', linewidths=1.5))
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)

names = list(final)
fig = plt.figure(figsize=(16, 2.5*(len(names) + 1)))
ax = fig.add_subplot(len(names) + 1, 1, 1)
ax.semilogy(np.arange(1, Ndes + 1), lamH[::-1], 'k.', ms=3)
ax.axvline(m + 0.5, ls=':', c='gray'); ax.set_xlabel('mode of W^H (Re Zs L + R0) W (descending)'); ax.set_ylabel('eigenvalue')
ax.set_title(f"{m} significant modes | {K} plateau modes", fontsize=9); ax.grid(True, which='both', alpha=0.3)
for row, name in enumerate(names):
    r = inference[name]
    dmax = np.quantile(np.abs(r["d"]), 0.99)
    form = max(["impedance", "admittance"], key=lambda f: r[f]["P"])
    edgePlot(fig.add_subplot(len(names) + 1, 3, 3*(row + 1) + 1), r["d"].real, f"{name}: Re d", clim=(-dmax, dmax))
    edgePlot(fig.add_subplot(len(names) + 1, 3, 3*(row + 1) + 2), r["d"].imag, f"{name}: Im d", clim=(-dmax, dmax))
    structurePlot(fig.add_subplot(len(names) + 1, 3, 3*(row + 1) + 3), r[form]["keep"],
                  f"{form} form, binary P/D = {r[form]['P']/Dref:.4f}")
plt.tight_layout()
out = "results/voltageModeSplit.png"
plt.savefig(out, dpi=110)
print(f"figure: {out}")
