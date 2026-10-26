"""Scraping study (generated from scraping.ipynb for batch runs, e.g. Narval).

Environment: OPTTOL (dual solver, default 1e-9), GCD_OPTTOL (default 1e-6), NGCD (default 100),
SEED (default: SLURM_ARRAY_TASK_ID or 0). Writes results/scraping_<TAG>.npz and figures.
"""
import os
import matplotlib
matplotlib.use('Agg')
OPTTOL = float(os.environ.get('OPTTOL', 1e-9))
GCD_OPTTOL = float(os.environ.get('GCD_OPTTOL', 1e-6))
SEED = int(os.environ.get('SEED', os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
TAG = os.environ.get('TAG', f"seed{SEED}")
os.makedirs('results', exist_ok=True)
import numpy as _np
_np.random.seed(SEED)
print(f"OPTTOL {OPTTOL:g}, GCD_OPTTOL {GCD_OPTTOL:g}, SEED {SEED}, TAG {TAG}", flush=True)

import numpy as np
import scipy.sparse as sp
import time
from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
import copy
import matplotlib.pyplot as plt
from dolphindes.cvxopt.gcd import GCDHyperparameters

matrixSet = "patch21ka1"   # "dipole" | "patch21ka1": operators loaded from *_{matrixSet}.txt (both ka = 1)

# Physical constants
epsilon_0 = 8.854e-12 # F/m
mu_0 = 4e-7 * np.pi # H/m
c = 1/np.sqrt(epsilon_0 * mu_0) # speed of light in vacuum

E0 = 1e3 # plane wave amplitude in V/m

frequency = 2.45e9 # frequency in Hz
wavelength = c / frequency # wavelength in m
print(f"Wavelength: {wavelength} m")

ka = 1 # electrical size of the antenna
k = 2 * np.pi / wavelength # wavenumber
a = ka / k # antenna circumradius
print(f"Antenna circumradius: {a} m")

omega = 2 * np.pi * frequency # angular frequency

conductivity_reduction_factor = 1 # factor to reduce conductivity for testing purposes
copper_conductivity = conductivity_reduction_factor*5.96e7 # S/m
copper_permittivity = 1 + 1j * copper_conductivity / (omega * epsilon_0) # copper permittivity in dimensionless units
print(f"Copper permittivity: {copper_permittivity:.2e}")

# Calculate surface impedance
delta = np.sqrt(2 / (omega * mu_0 * copper_conductivity)) # skin depth in m
Zs = (1 + 1j) / (copper_conductivity * delta)
print(f"Surface impedance: {Zs} Ω")

# Load matrices from text files
Lmat = np.loadtxt(f"Lmat_{matrixSet}.txt", delimiter=',') # lossy matrix

Ndes = Lmat.shape[0] # number of design variables
print(f"Matrix set: {matrixSet}, number of design variables: {Ndes}")

Lchol = np.linalg.cholesky(Lmat).conj().T
LcholInv = np.linalg.inv(Lchol)

def Msym(M):
    return(0.5*(M.conj().T + M))

def Mchol(M):
    return(Msym(LcholInv.conj().T @ M @ LcholInv))

R0 = np.loadtxt(f"R0_{matrixSet}.txt", delimiter=',') # radiated power matrix
R0 = Mchol(R0)

X0 = np.loadtxt(f"X0_{matrixSet}.txt", delimiter=',') # reactive power matrix
X0 = Mchol(X0)

Vinc = np.loadtxt(f"V_{matrixSet}.txt", delimiter=',') # voltage excitation vector
Vinc = E0*Vinc
def Vchol(v):
    return(LcholInv.conj().T @ v)
Vinc = Vchol(Vinc)

Zmat = Zs * np.eye(Ndes) # material impedance matrix
Z0 = R0 + 1j*X0 # free-space impedance matrix
Ztot = Z0 + Zmat # total impedance matrix

def evalObjI(Ivec):
    # radiated power in the current basis
    return(0.5*np.real(Ivec.conj().T @ R0 @ Ivec))

# full plate performance
Ifull = np.linalg.solve(Ztot, Vinc) # current distribution for full plate
Pfull = evalObjI(Ifull) # objective for full plate
print(f"Objective for full plate: {Pfull:.6e} W")

print(f"cond(Z0) = {np.linalg.cond(Z0):.3e}")
W = np.linalg.inv(Z0)   # Z0^{-1}
WH = W.conj().T

Obj = -0.5 * Msym(WH @ R0 @ W)
obj = np.zeros(Ndes, dtype=complex)
obj0 = 0.0

def VtoI(Vsca):
    return(-W @ Vsca)

def evalObj(Vsca):
    return(-np.real(Vsca.conj().T @ Obj @ Vsca) + 2*np.real(Vsca.conj().T @ obj) + obj0)

Vfull = -Z0 @ Ifull # scattered voltage for full plate
print(f"Objective for full plate in Vsca: {evalObj(Vfull):.6e} W (I basis {Pfull:.6e} W)")

A1 = np.eye(Ndes) + np.conj(Zs) * WH
A2 = W
s1 = -0.5 * Vinc

qList = [1, -1j]   # real and imaginary part (real and reactive power), p = q 1
Plist = [sp.diags(np.conj(q)*np.ones(Ndes, dtype=complex)) for q in qList]   # P' = diag(p^*)

def projRes(x, Pp):
    # Re(-x^H A1 P' A2 x + 2 x^H A2^H P'^H s1)
    return(np.real(-x.conj() @ A1 @ (Pp @ (A2 @ x)) + 2*x.conj() @ (A2.conj().T @ (Pp.conj().T @ s1))))

def directRes(Vsca, p):
    # Re[I^H diag(p) (V + Vsca - Zs I)] evaluated directly
    I = VtoI(Vsca)
    return(np.real(np.sum(p*np.conj(I)*(Vinc + Vsca - Zs*I))))

print("power conservation residuals at full plate:",
      [f"{projRes(Vfull, Pp):.2e} (direct {directRes(Vfull, q*np.ones(Ndes)):.2e})" for Pp, q in zip(Plist, qList)])

# check: shared projector form equals the direct constraint for a random projector at a random point
pRand = np.random.randn(Ndes) + 1j*np.random.randn(Ndes)
xRand = np.random.randn(Ndes) + 1j*np.random.randn(Ndes)
print(f"random p, random x: shared form {projRes(xRand, sp.diags(np.conj(pRand))):.6e}, direct {directRes(xRand, pRand):.6e}")

QCQP = DenseSharedProjQCQP(Obj, obj, obj0, A1, s1, Plist, A2=A2, verbose=1)

lags_init = np.array([1.0, 0.0])   # lambda = 1 on the real power constraint
print(f"initial lags dual feasible: {QCQP.is_dual_feasible(lags_init)}")

opt_params = OptimizationHyperparameters(opttol=OPTTOL,
    gradConverge=True,
    min_inner_iter=10,
    max_restart=10,
    penalty_ratio=1e-2,
    penalty_reduction=0.1,
    break_iter_period=5,
    verbose=0)

t1 = time.time()
result = QCQP.solve_current_dual_problem(method='newton', init_lags=lags_init, opt_params=opt_params)
print(f"bound: {result[0]:.6e} W, time: {time.time()-t1:.2f}s, Lagrange multipliers: {QCQP.current_lags}")

Ngcd = int(os.environ.get('NGCD', 100))
cacheFile = None   # e.g. "scraping_gcd100.npz"

import os
if cacheFile is not None and os.path.exists(cacheFile):
    st = np.load(cacheFile)
    PlistG = [sp.diags(st['Pdata'][:, k]) for k in range(st['Pdata'].shape[1])]
    gQCQP = DenseSharedProjQCQP(Obj, obj, obj0, A1, s1, PlistG, A2=A2, verbose=0)
    gQCQP.current_lags = st['lags']
    gQCQP.current_dual = gQCQP.get_dual(gQCQP.current_lags)[0]
    gQCQP.current_xstar = gQCQP._get_xstar(gQCQP.current_lags)[0]
else:
    gQCQP = copy.deepcopy(QCQP)
    gQCQP.verbose = 0
    t1 = time.time()
    gQCQP.run_gcd(GCDHyperparameters(max_proj_cstrt_num=2 + 2*Ngcd, max_gcd_iter_num=Ngcd,
                                     gcd_iter_period=Ngcd + 1, gcd_tol=GCD_OPTTOL, opt_params=opt_params))
    print(f"GCD time {time.time()-t1:.1f}s")
    if cacheFile is not None:
        np.savez(cacheFile, Pdata=gQCQP.Proj.get_Pdata_column_stack(), lags=gQCQP.current_lags)
gcdLags = gQCQP.current_lags.copy()
print(f"GCD bound {gQCQP.current_dual:.6e} W, {gQCQP.n_proj_constr} projectors")

Rtot = Msym(Ztot)
Xtot = (Ztot - Ztot.conj().T) / 2j
Crt = np.linalg.cholesky(Rtot).conj().T

def ellipsoid(V):
    Ic = 0.5 * np.linalg.solve(Rtot, V)
    r = 0.5 * np.sqrt(np.real(V.conj() @ np.linalg.solve(Rtot, V)))
    return Ic, r

def measures(x, V):
    # relative distance from the real-power ellipsoid of V, normalized reactive residual
    I = VtoI(x)
    Ic, r = ellipsoid(V)
    d = (np.linalg.norm(Crt @ (I - Ic)) - r) / r
    cim = np.real(I.conj() @ Xtot @ I + I.conj() @ (1j * V)) / r**2
    return d, cim

def project(x, V):
    # radial projection of x onto the real-power ellipsoid of V
    I = VtoI(x)
    Ic, r = ellipsoid(V)
    Ip = Ic + r / np.linalg.norm(Crt @ (I - Ic)) * (I - Ic)
    return -Z0 @ Ip

Ic, r = ellipsoid(Vinc)
print(f"r^2 = {r**2:.6e} W")
for name, x in [("full plate", Vfull), ("global constraints", QCQP.current_xstar), (f"GCD {Ngcd}", gQCQP.current_xstar)]:
    d, cim = measures(x, Vinc)
    print(f"{name:20s} d = {d:+.3e}, c_im/r^2 = {cim:+.3e}")

def scrape(target, J=8, tau=1e-2):
    # target: "original" (ellipsoid of Vinc) or "own" (ellipsoid of the current V_j)
    Q = copy.deepcopy(gQCQP)
    Q.s1 = -0.5 * Vinc
    Q.compute_precomputed_values()
    lags = gcdLags.copy()
    Q.solve_current_dual_problem('newton', init_lags=lags, opt_params=opt_params)
    lags, x, V, D = Q.current_lags.copy(), Q.current_xstar.copy(), Vinc.copy(), Q.current_dual
    q = lambda lg: Q.Proj.Pdiags.conj() @ lg[:Q.n_proj_constr]
    assert np.allclose(Q._get_total_S(lags), A2.conj().T @ (q(lags) * (-0.5 * V)))   # S(lambda) V
    hist = []
    def record(j, eta, lin):
        dO, cO = measures(x, Vinc); dJ, cJ = measures(x, V)
        A = Msym(Q._get_total_A(lags))
        hist.append(dict(j=j, eta=eta, D=D, lin=lin, d=dO, cim=cO, dOwn=dJ, cimOwn=cJ,
                         e=np.linalg.norm(V - Vinc)/np.linalg.norm(Vinc), mineig=np.linalg.eigvalsh(A)[0],
                         A=A, s=Q._get_total_S(lags), x=x.copy()))
    record(0, 0.0, D)
    for j in range(1, J + 1):
        A = Q._get_total_A(lags)
        xp = project(x, Vinc if target == "original" else V)
        ds = A @ (xp - x)
        dV = -2 * (Z0.conj().T @ ds) / q(lags)
        eta = 1.0
        while True:
            Vn = V + eta * dV
            Q.s1 = -0.5 * Vn
            Q.compute_precomputed_values()
            lin = D + 2 * eta * np.real(x.conj() @ ds)
            Q.solve_current_dual_problem('newton', init_lags=lags, opt_params=opt_params)
            if abs(Q.current_dual - lin) <= tau * abs(D) or eta < 1e-3:
                break
            eta /= 2
        lags, x, V, D = Q.current_lags.copy(), Q.current_xstar.copy(), Vn, Q.current_dual
        record(j, eta, lin)
    return hist

def show(hist, title):
    print(title)
    print(f"{'j':>2} {'eta':>5} {'D_j (W)':>12} {'lin (W)':>12} {'d(V)':>10} {'cim(V)/r2':>10} {'d(Vj)':>10} {'cim(Vj)/r2':>10} {'e_j':>9} {'min eig':>9}")
    for h in hist:
        print(f"{h['j']:2d} {h['eta']:5.2f} {h['D']:12.6e} {h['lin']:12.6e} {h['d']:+10.3e} {h['cim']:+10.3e} "
              f"{h['dOwn']:+10.3e} {h['cimOwn']:+10.3e} {h['e']:9.2e} {h['mineig']:9.2e}")

t1 = time.time()
histO = scrape("original")
show(histO, "target: ellipsoid of the original excitation V")
histJ = scrape("own")
show(histJ, "\ntarget: ellipsoid of the perturbed excitation V_j")
print(f"\n{time.time()-t1:.1f}s")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for hist, lab, m in [(histO, "target V", 'o-'), (histJ, r"target $V_j$", 's--')]:
    j = [h['j'] for h in hist]
    axes[0].semilogy(j, [abs(h['d']) for h in hist], m, label=f"{lab}: $|d|$ w.r.t. $V$")
    axes[0].semilogy(j, [abs(h['dOwn']) for h in hist], m, alpha=0.4, label=f"{lab}: $|d|$ w.r.t. $V_j$")
    axes[1].plot(j, [h['cim'] for h in hist], m, label=lab)
    axes[2].plot(j, [h['D'] for h in hist], m, label=f"{lab}: $D_j$")
axes[0].axhline(1e-3, c='gray', ls=':', label='GCD accuracy')
axes[0].set_xlabel('scraping step $j$'); axes[0].set_ylabel('relative distance from ellipsoid'); axes[0].legend(fontsize=7)
axes[1].set_xlabel('scraping step $j$'); axes[1].set_ylabel(r'$c_\mathrm{im}/r^2$'); axes[1].legend(fontsize=7)
axes[2].axhline(gQCQP.current_dual, c='k', ls='--', lw=0.8, label='GCD bound $D_0$')
axes[2].set_xlabel('scraping step $j$'); axes[2].set_ylabel('dual value (W)'); axes[2].legend(fontsize=7)
for ax in axes: ax.grid(True)
plt.tight_layout(); plt.savefig(f'results/scraping_{TAG}_fig1.png', dpi=150); plt.close()

tauD, tauV = 1e-3, 1e-3
om = np.array([np.exp(-h['e']/tauV) + tauD/(abs(h['d']) + tauD) for h in histO])
mu = om / om.sum()
Amu = sum(m * h['A']/np.linalg.norm(h['A']) for m, h in zip(mu, histO))
smu = sum(m * h['s']/np.linalg.norm(h['A']) for m, h in zip(mu, histO))
ev = np.linalg.eigvalsh(Amu)
xbar = np.linalg.solve(Amu, smu)
print("weights mu_j:", np.array2string(mu, precision=3))
print(f"A_mu eigenvalues: min {ev[0]:.3e}, max {ev[-1]:.3e} (single A_j: min {min(np.linalg.eigvalsh(h['A']/np.linalg.norm(h['A']))[0] for h in histO):.3e})")
d, cim = measures(xbar, Vinc)
print(f"chained center: d = {d:+.3e}, c_im/r^2 = {cim:+.3e}, P = {evalObj(xbar):.6e} W, "
      f"relative difference from x_0: {np.linalg.norm(xbar - histO[0]['x'])/np.linalg.norm(histO[0]['x']):.3e}")

from scipy.optimize import minimize

m0 = 1 / np.sqrt(Ndes)

def chain(hist, mu_tot, only=None, tauD=1e-3, tauV=1e-3):
    # normalized chained Lagrangian A_mu, s_mu and the constant making each term a squared norm
    if only is None:
        om = np.array([np.exp(-h['e']/tauV) + tauD/(abs(h['d']) + tauD) for h in hist])
    else:
        om = np.array([1.0 if j in only else 0.0 for j in range(len(hist))])
    mu = mu_tot * om / om.sum()
    nA = [np.linalg.norm(h['A']) for h in hist]
    Amu = sum(m * h['A']/n for m, h, n in zip(mu, hist, nA))
    smu = sum(m * h['s']/n for m, h, n in zip(mu, hist, nA))
    cst = sum(m * np.real(h['x'].conj() @ h['s'])/n for m, h, n in zip(mu, hist, nA))
    return Amu, smu, cst

def chainedObjective(Amu, smu, cst):
    K = np.linalg.inv(m0 * np.eye(Ndes) + Amu)
    def f(rho):
        S = np.diag(rho.astype(complex))
        M = Zs * np.eye(Ndes) + S @ Z0
        I = np.linalg.solve(M, S @ Vinc)
        x = -Z0 @ I
        xt = K @ (m0 * x + smu)
        val = m0 * np.linalg.norm(x - xt)**2 + np.real(xt.conj() @ Amu @ xt) - 2*np.real(xt.conj() @ smu) + cst
        w = np.linalg.solve(M.conj().T, Z0.conj().T @ (m0 * (x - xt)))
        return val, -2 * np.real(np.conj(w) * (Vinc - Z0 @ I))
    return f

def realizedObj(rho):
    S = np.diag(rho.astype(complex))
    return evalObjI(np.linalg.solve(Zs * np.eye(Ndes) + S @ Z0, S @ Vinc))

ths = np.linspace(0.05, 0.95, 19)
def bestBinary(rho):
    P = [realizedObj((rho > t).astype(float)) for t in ths]
    k = int(np.argmax(P))
    return P[k], ths[k]

# gradient check
f = chainedObjective(*chain(histO, 1.0))
rho = np.random.rand(Ndes); e = np.zeros(Ndes); e[5] = 1e-6
print(f"gradient check: adjoint {f(rho)[1][5]:.6e}, finite difference {(f(rho + e)[0] - f(rho - e)[0])/2e-6:.6e}")

# starts: separable current fit of x_0, uniform, random
I0 = VtoI(histO[0]['x']); b0 = (Vinc + histO[0]['x'])/Zs
rng = np.random.default_rng(0)
starts = [np.clip(np.real(np.conj(b0)*I0)/np.abs(b0)**2, 0, 1), 0.5*np.ones(Ndes)] + [rng.random(Ndes) for _ in range(8)]

seedP, (seedB, seedT) = realizedObj(starts[0]), bestBinary(starts[0])
print(f"direct inference seed (separable current fit of x_0, unfitted): P(rho) = {seedP:.3e} W, "
      f"best binary {seedB:.3e} W (threshold {seedT:.2f}), {seedB/gQCQP.current_dual:.3%} of D_0", flush=True)

t1 = time.time()
fitRes = {}
print(f"dual bound D_0 = {gQCQP.current_dual:.4f} W, full plate {Pfull:.4f} W")
print(f"{'case':12s} {'mu':>5s} {'med P(rho)':>11s} {'med binary':>11s} {'max binary':>11s} {'max P(rho)':>11s} "
      f"{'seed P(rho)':>11s} {'seed binary':>11s}")
for mu_tot in [0.1, 1.0, 10.0]:
    for label, only in [('chained', None), ('single j=0', [0]), ('single j=J', [len(histO) - 1])]:
        f = chainedObjective(*chain(histO, mu_tot, only))
        out = []
        for r0 in starts:
            rho = minimize(f, r0, jac=True, method='L-BFGS-B', bounds=[(0, 1)]*Ndes,
                           options=dict(maxiter=3000, ftol=1e-14, gtol=1e-12)).x
            out.append((realizedObj(rho), *bestBinary(rho), rho))
        fitRes[(label, mu_tot)] = out
        P, B = np.array([o[0] for o in out]), np.array([o[1] for o in out])
        print(f"{label:12s} {mu_tot:5g} {np.median(P):11.3e} {np.median(B):11.3e} {B.max():11.3e} {P.max():11.3e} "
              f"{P[0]:11.3e} {B[0]:11.3e}", flush=True)
print(f"{time.time()-t1:.0f}s")

fig, ax = plt.subplots(figsize=(8, 3.5))
labels = ['chained', 'single j=0', 'single j=J']
for k, label in enumerate(labels):
    for i, mu_tot in enumerate([0.1, 1.0, 10.0]):
        B = np.maximum([o[1] for o in fitRes[(label, mu_tot)]], 1e-7)
        xs = np.full(len(B), 4*i + k) + 0.1*rng.standard_normal(len(B))
        ax.semilogy(xs[1:], B[1:], 'o', color=f"C{k}", alpha=0.7, label=label if i == 0 else None)
        # starts[0]: separable current fit of x_0 (direct inference seed)
        ax.semilogy(4*i + k, B[0], '*', color=f"C{k}", mec='k', ms=13, zorder=3,
                    label='direct inference seed' if i == 0 and k == 0 else None)
ax.axhline(seedB, c='k', ls='-.', lw=0.8, label='seed, unfitted')
ax.axhline(gQCQP.current_dual, c='k', ls='--', lw=0.8, label='bound $D_0$')
ax.axhline(Pfull, c='gray', ls=':', label='full plate')
ax.set_xticks([1, 5, 9]); ax.set_xticklabels([r'$\mu = 0.1$', r'$\mu = 1$', r'$\mu = 10$'])
ax.set_ylabel('best binary objective (W)'); ax.legend(fontsize=8); ax.grid(True)
plt.tight_layout(); plt.savefig(f'results/scraping_{TAG}_fig2.png', dpi=150); plt.close()

# ---- save results
keys = ['j', 'eta', 'D', 'lin', 'd', 'cim', 'dOwn', 'cimOwn', 'e', 'mineig']
out = {f"orig_{k}": np.array([h[k] for h in histO]) for k in keys}
out.update({f"own_{k}": np.array([h[k] for h in histJ]) for k in keys})
out['fit_cases'] = np.array([f"{lab}|{m:g}" for (lab, m) in fitRes])
out['fit_P'] = np.array([[o[0] for o in v] for v in fitRes.values()])
out['fit_bin'] = np.array([[o[1] for o in v] for v in fitRes.values()])
out['fit_rho'] = np.array([[o[3] for o in v] for v in fitRes.values()])
out['D0'] = gQCQP.current_dual
out['seed_rho'] = starts[0]
out['seed_P'] = seedP
out['seed_bin'] = seedB
out['Amu_mineig'] = ev[0]
out['xbar_d'] = measures(xbar, Vinc)[0]
out.update(OPTTOL=OPTTOL, GCD_OPTTOL=GCD_OPTTOL, SEED=SEED)
np.savez(f"results/scraping_{TAG}.npz", **out)
print(f"saved results/scraping_{TAG}.npz", flush=True)
