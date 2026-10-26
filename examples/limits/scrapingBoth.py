import os
import matplotlib
matplotlib.use("Agg")
NGCD = int(os.environ.get("NGCD", 100))
SEED = int(os.environ.get("SEED", 0))
TAG = os.environ.get("TAG", f"ngcd{NGCD}_seed{SEED}")
import numpy as np
np.random.seed(SEED)
print(f"NGCD {NGCD}, SEED {SEED}, TAG {TAG}", flush=True)
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

opt_params = OptimizationHyperparameters(opttol=1e-6,
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

Ngcd = NGCD
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
                                     gcd_iter_period=Ngcd + 1, opt_params=opt_params))
    print(f"GCD time {time.time()-t1:.1f}s")
    if cacheFile is not None:
        np.savez(cacheFile, Pdata=gQCQP.Proj.get_Pdata_column_stack(), lags=gQCQP.current_lags)
gcdLags = gQCQP.current_lags.copy()
print(f"GCD bound {gQCQP.current_dual:.6e} W, {gQCQP.n_proj_constr} projectors")

import scipy.linalg as sla

Rtot = Msym(Ztot)
Xtot = (Ztot - Ztot.conj().T) / 2j
Xtot = Msym(Xtot)
g = sla.eigh(Xtot.real, Rtot.real, eigvals_only=True)
thHi = np.arctan(1/g.max())
thScan = np.linspace(0, thHi, 400)[1:-1]
condV = [np.linalg.cond(Msym(W.conj().T @ (np.cos(t)*Rtot - np.sin(t)*Xtot) @ W)) for t in thScan]
thOpt = thScan[int(np.argmin(condV))]
print(f"rotated real power positive definite up to {np.degrees(thHi):.4e} deg; best-conditioned angle {np.degrees(thOpt):.4e} deg, "
      f"cond W^H H W {min(condV):.2e} (theta = 0: {np.linalg.cond(Msym(W.conj().T @ Rtot @ W)):.2e})")

def Hth(th):
    return np.cos(th)*Rtot - np.sin(th)*Xtot

Cth = {th: np.linalg.cholesky(Hth(th)).conj().T for th in [0.0, thOpt]}

def ellipsoid(V, th=0.0):
    H = Hth(th)
    Ic = 0.5 * np.linalg.solve(H, np.exp(1j*th)*V)
    r = 0.5 * np.sqrt(np.real(V.conj() @ np.linalg.solve(H, V)))
    return Ic, r

def measures(x, V):
    # signed relative distances from the real-power and the rotated ellipsoid of V, normalized reactive residual
    I = VtoI(x)
    out = []
    for th in [0.0, thOpt]:
        Ic, r = ellipsoid(V, th)
        out.append((np.linalg.norm(Cth[th] @ (I - Ic)) - r) / r)
    Ic, r = ellipsoid(V)
    cim = np.real(I.conj() @ Xtot @ I + I.conj() @ (1j * V)) / r**2
    return out[0], out[1], cim

# check: a structure (the full plate) lies on every ellipsoid
print("full plate: d_0 {:+.1e}, d_theta {:+.1e}, c_im/r^2 {:+.1e}".format(*measures(Vfull, Vinc)))

def projectRadial(x, V):
    I = VtoI(x)
    Ic, r = ellipsoid(V)
    Ip = Ic + r / np.linalg.norm(Cth[0.0] @ (I - Ic)) * (I - Ic)
    return -Z0 @ Ip

metricBoth = {"both, A0 metric": Msym(WH @ Rtot @ W), "both, rotated metric": Msym(WH @ Hth(thOpt) @ W)}

def projectBoth(x0, V, M):
    # argmin (x - x0)^H M (x - x0) s.t. the global real and reactive power of V (two-constraint QCQP, solved by its dual)
    M = M/np.linalg.norm(M, 2)
    Qp = DenseSharedProjQCQP(M, M @ x0, -np.real(x0.conj() @ M @ x0), A1, -0.5*V, Plist, A2=A2, verbose=0)
    Qp.solve_current_dual_problem(method='newton', init_lags=np.zeros(2), opt_params=opt_params)
    return Qp.current_xstar

for name, x in [("full plate", Vfull), ("global constraints", QCQP.current_xstar), (f"GCD {Ngcd}", gQCQP.current_xstar)]:
    print(f"{name:20s} d_0 {measures(x, Vinc)[0]:+.3e}, d_theta {measures(x, Vinc)[1]:+.3e}, c_im/r^2 {measures(x, Vinc)[2]:+.3e}")
for name, M in metricBoth.items():
    xp = projectBoth(gQCQP.current_xstar, Vinc, M)
    print(f"GCD current projected ({name}): d_0 {measures(xp, Vinc)[0]:+.1e}, d_theta {measures(xp, Vinc)[1]:+.1e}, c_im/r^2 {measures(xp, Vinc)[2]:+.1e}, "
          f"move {np.linalg.norm(xp - gQCQP.current_xstar)/np.linalg.norm(gQCQP.current_xstar):.2e}")

def scrape(target, J=8, tau=1e-2):
    Q = copy.deepcopy(gQCQP)
    Q.s1 = -0.5 * Vinc
    Q.compute_precomputed_values()
    Q.solve_current_dual_problem('newton', init_lags=gcdLags.copy(), opt_params=opt_params)
    lags, x, V, D = Q.current_lags.copy(), Q.current_xstar.copy(), Vinc.copy(), Q.current_dual
    q = lambda lg: Q.Proj.Pdiags.conj() @ lg[:Q.n_proj_constr]
    hist = []
    def record(j, eta, lin):
        d0, dth, cim = measures(x, Vinc)
        A = Msym(Q._get_total_A(lags))
        hist.append(dict(j=j, eta=eta, D=D, lin=lin, d=d0, dth=dth, cim=cim, e=np.linalg.norm(V - Vinc)/np.linalg.norm(Vinc),
                         mineig=np.linalg.eigvalsh(A)[0]/np.linalg.eigvalsh(A)[-1], x=x.copy(), A=A, s=Q._get_total_S(lags)))
    record(0, 0.0, D)
    for j in range(1, J + 1):
        A = Q._get_total_A(lags)
        xp = projectRadial(x, Vinc) if target == "radial" else projectBoth(x, Vinc, metricBoth[target])
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
    print(f"{'j':>2} {'eta':>5} {'D_j (W)':>12} {'lin (W)':>12} {'d_0':>10} {'d_theta':>10} {'cim/r2':>10} {'e_j':>9} {'min/max eig':>11}")
    for h in hist:
        print(f"{h['j']:2d} {h['eta']:5.2f} {h['D']:12.6e} {h['lin']:12.6e} {h['d']:+10.3e} {h['dth']:+10.3e} {h['cim']:+10.3e} {h['e']:9.2e} {h['mineig']:11.2e}")

hists = {}
for target in ["radial", "both, A0 metric", "both, rotated metric"]:
    t1 = time.time()
    hists[target] = scrape(target)
    show(hists[target], f"\ntarget: {target} ({time.time() - t1:.0f}s)")

fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
for (target, hist), m in zip(hists.items(), ['o-', 's--', '^-.']):
    j = [h['j'] for h in hist]
    axes[0].semilogy(j, [abs(h['d']) for h in hist], m, label=target)
    axes[1].semilogy(j, [abs(h['dth']) for h in hist], m, label=target)
    axes[2].semilogy(j, [abs(h['cim']) for h in hist], m, label=target)
    axes[3].plot(j, [h['e'] for h in hist], m, label=target)
for ax, lab in zip(axes, [r'$|d_0|$ (real-power ellipsoid)', r'$|d_\theta|$ (rotated ellipsoid)', r'$|c_\mathrm{im}|/r^2$', r'source change $e_j$']):
    ax.set_xlabel('scraping step $j$'); ax.set_ylabel(lab); ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'results/scrapingBoth_{TAG}_history.png', dpi=120); plt.close('all')

from scipy.optimize import lsq_linear

def fitDensity(Vsca):
    b = (Vinc + Vsca)/Zs
    I = VtoI(Vsca)
    return np.clip(np.real(np.conj(b)*I)/np.abs(b)**2, 0, 1)

def fitDensityVoltage(Vsca):
    Vtot = Vinc + Vsca
    M = Z0 * Vtot[None, :]
    r = -Zs * Vsca
    return lsq_linear(np.vstack([M.real, M.imag]), np.concatenate([r.real, r.imag]), bounds=(0, 1), method='bvls', tol=1e-12).x

def realizedObj(rho):
    S = np.diag(rho.astype(complex))
    return evalObjI(np.linalg.solve(Zs*np.eye(Ndes) + S @ Z0, S @ Vinc))

thsBin = np.linspace(0.05, 0.95, 19)
D0 = gQCQP.current_dual
cases = {"GCD x* (j = 0)": hists["radial"][0]["x"], **{f"scraped, {t} (j = {h[-1]['j']})": h[-1]["x"] for t, h in hists.items()}}
for name, x in cases.items():
    rhoC = Zs*VtoI(x)/(Vinc + x)
    parts = []
    for fname, fit in [("current fit", fitDensity), ("voltage fit", fitDensityVoltage)]:
        rho = fit(x)
        parts.append(f"{fname}: relaxed P/D {realizedObj(rho)/D0:.4f}, best binary {max(realizedObj((rho > t).astype(float)) for t in thsBin)/D0:.4f}")
    print(f"{name:42s}: max|complex density| {np.abs(rhoC).max():.1e}; " + "; ".join(parts), flush=True)

# ---- chained density fit (scraping.ipynb), for every scraping target ----
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


import time
fitOut = {}
seedOut = {}
D0 = gQCQP.current_dual
print(f"dual bound D_0 = {D0:.4f} W, full plate {Pfull:.4f} W", flush=True)
for target, hist in hists.items():
    I0 = VtoI(hist[0]['x']); b0 = (Vinc + hist[0]['x'])/Zs
    rng = np.random.default_rng(SEED)
    starts = [np.clip(np.real(np.conj(b0)*I0)/np.abs(b0)**2, 0, 1), 0.5*np.ones(Ndes)] + [rng.random(Ndes) for _ in range(8)]
    f = chainedObjective(*chain(hist, 1.0))
    rho = np.random.rand(Ndes); e = np.zeros(Ndes); e[5] = 1e-6
    print(f"[{target}] gradient check: adjoint {f(rho)[1][5]:.6e}, finite difference {(f(rho + e)[0] - f(rho - e)[0])/2e-6:.6e}", flush=True)
    seedP, (seedB, seedT) = realizedObj(starts[0]), bestBinary(starts[0])
    seedOut[target] = (seedP, seedB)
    print(f"[{target}] direct inference seed (separable current fit of x_0, unfitted): P(rho) = {seedP:.3e} W, "
          f"best binary {seedB:.3e} W (threshold {seedT:.2f}), {seedB/D0:.3%} of D_0", flush=True)
    print(f"[{target}] {'case':12s} {'mu':>5s} {'med P(rho)':>11s} {'med binary':>11s} {'max binary':>11s} {'max P(rho)':>11s} "
          f"{'seed P(rho)':>11s} {'seed binary':>11s}", flush=True)
    t1 = time.time()
    for mu_tot in [0.1, 1.0, 10.0]:
        for label, only in [('chained', None), ('single j=0', [0]), ('single j=J', [len(hist) - 1])]:
            f = chainedObjective(*chain(hist, mu_tot, only))
            out = []
            for r0 in starts:
                rho = minimize(f, r0, jac=True, method='L-BFGS-B', bounds=[(0, 1)]*Ndes,
                               options=dict(maxiter=3000, ftol=1e-14, gtol=1e-12)).x
                out.append((realizedObj(rho), *bestBinary(rho), rho))
            fitOut[(target, label, mu_tot)] = out
            P, B = np.array([o[0] for o in out]), np.array([o[1] for o in out])
            print(f"[{target}] {label:12s} {mu_tot:5g} {np.median(P):11.3e} {np.median(B):11.3e} {B.max():11.3e} {P.max():11.3e} "
                  f"{P[0]:11.3e} {B[0]:11.3e}", flush=True)
    print(f"[{target}] chained fits {time.time()-t1:.0f}s", flush=True)

save = dict(D0=D0, Pfull=Pfull, ngcd=NGCD, seed=SEED)
for target, hist in hists.items():
    k = target.replace(", ", "_").replace(" ", "")
    for key in ['D', 'd', 'dth', 'cim', 'e', 'mineig', 'eta']:
        save[f"hist_{k}_{key}"] = np.array([h[key] for h in hist])
for target, (sP, sB) in seedOut.items():
    k = target.replace(", ", "_").replace(" ", "")
    save[f"seed_{k}_P"] = sP; save[f"seed_{k}_B"] = sB
for (target, label, mu_tot), out in fitOut.items():
    k = f"fit_{target.replace(', ', '_').replace(' ', '')}_{label.replace(' ', '').replace('=', '')}_mu{mu_tot:g}"
    save[k + "_P"] = np.array([o[0] for o in out]); save[k + "_B"] = np.array([o[1] for o in out])
    save[k + "_rhoSeed"] = out[0][3]
    save[k + "_rhoBest"] = out[int(np.argmax([o[1] for o in out]))][3]
np.savez(f"results/scrapingBoth_{TAG}.npz", **save)
print(f"saved results/scrapingBoth_{TAG}.npz", flush=True)
