"""Shared setup for scraping with a GCD rerun after each source change.

Used by scrapingInitGCD.py (initial GCD, saved to results/scrapingInitGCD_<TAG>.npz) and by the strategy
notebooks scrapingGCD_{Aphi,realPower,rotated,pointwise}.ipynb, which load that state, scrape the source,
rerun GCD warm (projectors kept) and fit the density to the chained dual solutions.
Copper 2:1 patch, ka = 1, scattered voltage x = Vsca in the whitened basis (as scrapingBoth.py).

Environment: OPTTOL (dual solver, default 1e-9), GCD_OPTTOL (default 1e-6).
"""
import os
import glob
import copy
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.optimize import minimize
from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters

OPTTOL = float(os.environ.get('OPTTOL', 1e-9))
GCD_OPTTOL = float(os.environ.get('GCD_OPTTOL', 1e-6))
os.makedirs('results', exist_ok=True)

matrixSet = "patch21ka1"   # operators loaded from *_{matrixSet}.txt

# Physical constants
epsilon_0 = 8.854e-12 # F/m
mu_0 = 4e-7 * np.pi # H/m
c = 1/np.sqrt(epsilon_0 * mu_0) # speed of light in vacuum
E0 = 1e3 # plane wave amplitude in V/m
frequency = 2.45e9 # frequency in Hz
omega = 2 * np.pi * frequency # angular frequency

copper_conductivity = 5.96e7 # S/m
delta = np.sqrt(2 / (omega * mu_0 * copper_conductivity)) # skin depth in m
Zs = (1 + 1j) / (copper_conductivity * delta) # surface impedance

Lmat = np.loadtxt(f"Lmat_{matrixSet}.txt", delimiter=',') # lossy matrix
Ndes = Lmat.shape[0]
Lchol = np.linalg.cholesky(Lmat).conj().T
LcholInv = np.linalg.inv(Lchol)

def Msym(M):
    return(0.5*(M.conj().T + M))

def Mchol(M):
    return(Msym(LcholInv.conj().T @ M @ LcholInv))

R0 = Mchol(np.loadtxt(f"R0_{matrixSet}.txt", delimiter=',')) # radiated power matrix
X0 = Mchol(np.loadtxt(f"X0_{matrixSet}.txt", delimiter=',')) # reactive power matrix
Vinc = LcholInv.conj().T @ (E0*np.loadtxt(f"V_{matrixSet}.txt", delimiter=',')) # whitened excitation

Z0 = R0 + 1j*X0 # free-space impedance matrix
Ztot = Z0 + Zs * np.eye(Ndes) # total impedance matrix

def evalObjI(Ivec):
    # radiated power in the current basis
    return(0.5*np.real(Ivec.conj().T @ R0 @ Ivec))

Ifull = np.linalg.solve(Ztot, Vinc)
Pfull = evalObjI(Ifull)

W = np.linalg.inv(Z0)   # Z0^{-1}
WH = W.conj().T

Obj = -0.5 * Msym(WH @ R0 @ W)
obj = np.zeros(Ndes, dtype=complex)
obj0 = 0.0

def VtoI(Vsca):
    return(-W @ Vsca)

Vfull = -Z0 @ Ifull # scattered voltage for full plate

A1 = np.eye(Ndes) + np.conj(Zs) * WH
A2 = W
s1 = -0.5 * Vinc

qList = [1, -1j]   # real and imaginary part (real and reactive power), p = q 1
Plist = [sp.diags(np.conj(q)*np.ones(Ndes, dtype=complex)) for q in qList]   # P' = diag(p^*)

opt_params = OptimizationHyperparameters(opttol=OPTTOL,
    gradConverge=True,
    min_inner_iter=10,
    max_restart=10,
    penalty_ratio=1e-2,
    penalty_reduction=0.1,
    break_iter_period=5,
    verbose=0)

# ---- global real power, rotated real power, ellipsoids
Rtot = Msym(Ztot)
Xtot = Msym((Ztot - Ztot.conj().T) / 2j)
g = sla.eigh(Xtot.real, Rtot.real, eigvals_only=True)
thHi = np.arctan(1/g.max())
thScan = np.linspace(0, thHi, 400)[1:-1]
condV = [np.linalg.cond(Msym(WH @ (np.cos(t)*Rtot - np.sin(t)*Xtot) @ W)) for t in thScan]
thOpt = thScan[int(np.argmin(condV))]

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

def pointViolation(x, V):
    # local complex violation conj(I_i) (V_i + x_i - Zs I_i) of the power balance, relative to |conj(I) V|
    I = VtoI(x)
    return np.linalg.norm(np.conj(I) * (V + x - Zs*I)) / np.linalg.norm(np.conj(I) * V)

def projectBoth(x0, V, M):
    # argmin (x - x0)^H M (x - x0) s.t. the global real and reactive power of V (two-constraint QCQP, solved by its dual)
    M = M/np.linalg.norm(M, 2)
    Qp = DenseSharedProjQCQP(M, M @ x0, -np.real(x0.conj() @ M @ x0), A1, -0.5*V, Plist, A2=A2, verbose=0)
    Qp.solve_current_dual_problem(method='newton', init_lags=np.zeros(2), opt_params=opt_params)
    return Qp.current_xstar

# ---- initial GCD state
def globalQCQP():
    QCQP = DenseSharedProjQCQP(Obj, obj, obj0, A1, s1, Plist, A2=A2, verbose=0)
    QCQP.solve_current_dual_problem(method='newton', init_lags=np.array([1.0, 0.0]), opt_params=opt_params)
    return QCQP

def saveInitialGCD(Q, path, **extra):
    np.savez(path, Pdata=Q.Proj.get_Pdata_column_stack(), lags=Q.current_lags, D0=Q.current_dual,
             xstar=Q.current_xstar, Vinc=Vinc, OPTTOL=OPTTOL, GCD_OPTTOL=GCD_OPTTOL, **extra)

def loadInitialGCD(path=None):
    if path is None:
        path = max(glob.glob('results/scrapingInitGCD_*.npz'), key=os.path.getmtime)
    st = np.load(path)
    PlistG = [sp.diags(st['Pdata'][:, k]) for k in range(st['Pdata'].shape[1])]
    Q = DenseSharedProjQCQP(Obj, obj, obj0, A1, s1, PlistG, A2=A2, verbose=0)
    Q.current_lags = st['lags'].copy()
    Q.current_dual = Q.get_dual(Q.current_lags)[0]
    Q.current_xstar = Q._get_xstar(Q.current_lags)[0]
    print(f"loaded {path}: {Q.n_proj_constr} projectors, D_0 = {Q.current_dual:.6f} W (saved {float(st['D0']):.6f} W)")
    return Q, int(st['ngcd']) if 'ngcd' in st.files else None

# ---- scraping steps: each returns the source change dV for the current x, V, lags
def mult(Q, lags):
    # q = sum_k lambda_k p_k, so that s(lambda; V) = -1/2 W^H diag(q) V
    return Q.Proj.Pdiags.conj() @ lags[:Q.n_proj_constr]

def metricStep(metric, target='own'):
    # project x onto both global constraints in the metric M(A_phi) and move the stationary point there
    def step(Q, x, V, lags):
        A = Q._get_total_A(lags)
        xp = projectBoth(x, V if target == 'own' else Vinc, metric(A))
        ds = A @ (xp - x)
        return -2 * (Z0.conj().T @ ds) / mult(Q, lags)
    return step

def pointwiseStep(Q, x, V, lags):
    # at fixed x, the source change removing every local violation conj(I_i) (V_i + x_i - Zs I_i)
    return -(V + x - Zs*VtoI(x))

# ---- scraping with a warm GCD rerun
def scrapeGCD(Q0, step, ngcdRe, jmax, ngcdMax, tau=1e-2, eta0=1.0, etaMin=1e-8, strongTol=1e-4, dropTol=0.3):
    Q = copy.deepcopy(Q0)
    lags, x, V, D = Q.current_lags.copy(), Q.current_xstar.copy(), Vinc.copy(), Q.current_dual
    D0 = D
    hist = []
    def record(j, eta, Dfix, lin, stop=''):
        d0, dth, cim = measures(x, Vinc)
        dOwn, dthOwn, cimOwn = measures(x, V)
        A = Msym(Q._get_total_A(lags))
        ev = np.linalg.eigvalsh(A)
        hist.append(dict(j=j, eta=eta, D=D, Dfix=Dfix, lin=lin, d=d0, dth=dth, cim=cim, dOwn=dOwn, dthOwn=dthOwn,
                         cimOwn=cimOwn, viol=pointViolation(x, V), e=np.linalg.norm(V - Vinc)/np.linalg.norm(Vinc),
                         mineig=ev[0]/ev[-1], nproj=Q.n_proj_constr, x=x.copy(), V=V.copy(), A=A,
                         s=Q._get_total_S(lags), stop=stop))
        h = hist[-1]
        print(f"j {j:3d} eta {eta:.2e} D {D:.5f} (fixed {Dfix:.5f}, lin {lin:.5f}) e {h['e']:.2e} | "
              f"own: d {dOwn:+.2e} cim {cimOwn:+.2e} viol {h['viol']:.2e} | orig: d {d0:+.2e} cim {cim:+.2e} | "
              f"mineig {h['mineig']:.1e} nproj {h['nproj']} {stop}", flush=True)
    record(0, 0.0, D, D)
    for j in range(1, jmax + 1):
        t1 = time.time()
        dV = step(Q, x, V, lags)
        sOld = Q._get_total_S(lags)
        eta = eta0
        while True:
            # dual value at fixed projectors, from the previous multipliers, against its first-order prediction
            Vn = V + eta * dV
            Q.s1 = -0.5 * Vn
            Q.compute_precomputed_values()
            lin = D + 2 * np.real(x.conj() @ (Q._get_total_S(lags) - sOld))
            Q.solve_current_dual_problem('newton', init_lags=lags, opt_params=opt_params)
            if abs(Q.current_dual - lin) <= tau * abs(D) or eta < etaMin:
                break
            eta /= 2
        if eta < etaMin:
            hist[-1]['stop'] = 'linearity check failed'
            print(f"    no step size above {etaMin:g} passes the linearity check, stop", flush=True)
            break
        Dfix = Q.current_dual
        Q.run_gcd(GCDHyperparameters(max_proj_cstrt_num=ngcdMax, max_gcd_iter_num=ngcdRe, gcd_iter_period=ngcdRe + 1,
                                     gcd_tol=GCD_OPTTOL, opt_params=opt_params))
        lags, x, V, D = Q.current_lags.copy(), Q.current_xstar.copy(), Vn, Q.current_dual
        dOwn, _, cimOwn = measures(x, V)
        stop = ''
        if D < (1 - dropTol) * D0:
            stop = f'bound dropped below {1 - dropTol:.0%} of D_0'
        elif abs(dOwn) <= strongTol and abs(cimOwn) <= strongTol:
            stop = 'strongly dual'
        record(j, eta, Dfix, lin, stop)
        print(f"    step {time.time() - t1:.0f}s", flush=True)
        if stop:
            break
    return hist

# ---- chained density fit (as scraping.ipynb / scrapingBoth.py)
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

def chainedFit(hist, seed=0, mus=(0.1, 1.0, 10.0)):
    # starts: separable current fit of x_0 (direct inference seed), uniform, 8 random
    I0 = VtoI(hist[0]['x']); b0 = (Vinc + hist[0]['x'])/Zs
    rng = np.random.default_rng(seed)
    starts = [np.clip(np.real(np.conj(b0)*I0)/np.abs(b0)**2, 0, 1), 0.5*np.ones(Ndes)] + [rng.random(Ndes) for _ in range(8)]
    D0 = hist[0]['D']
    print(f"{'case':12s} {'mu':>5s} {'med binary':>11s} {'max binary':>11s} {'seed':>11s} {'uniform':>11s}  (% of D_0)")
    fitOut = {}
    for mu_tot in mus:
        for label, only in [('chained', None), ('single j=0', [0]), ('single j=J', [len(hist) - 1])]:
            f = chainedObjective(*chain(hist, mu_tot, only))
            out = []
            for r0 in starts:
                rho = minimize(f, r0, jac=True, method='L-BFGS-B', bounds=[(0, 1)]*Ndes,
                               options=dict(maxiter=3000, ftol=1e-14, gtol=1e-12)).x
                out.append((realizedObj(rho), *bestBinary(rho), rho))
            fitOut[(label, mu_tot)] = out
            B = 100*np.array([o[1] for o in out])/D0
            print(f"{label:12s} {mu_tot:5g} {np.median(B):11.3f} {B.max():11.3f} {B[0]:11.3f} {B[1]:11.3f}", flush=True)
    return fitOut

# ---- output
histKeys = ['j', 'eta', 'D', 'Dfix', 'lin', 'd', 'dth', 'cim', 'dOwn', 'dthOwn', 'cimOwn', 'viol', 'e', 'mineig', 'nproj']

def plotHistory(hist, title, fname=None):
    import matplotlib.pyplot as plt
    j = [h['j'] for h in hist]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    axes[0].semilogy(j, [abs(h['d']) for h in hist], 'o-', label=r'$\mathbf{V}$')
    axes[0].semilogy(j, [abs(h['dOwn']) for h in hist], 's--', label=r'$\mathbf{V}_j$')
    axes[0].set_title(r'$|d|$ (real-power ellipsoid)')
    axes[1].semilogy(j, [abs(h['cim']) for h in hist], 'o-', label=r'$\mathbf{V}$')
    axes[1].semilogy(j, [abs(h['cimOwn']) for h in hist], 's--', label=r'$\mathbf{V}_j$')
    axes[1].set_title(r'$|c_\mathrm{im}|/r^2$')
    axes[2].semilogy(j, [h['viol'] for h in hist], 'o-'); axes[2].set_title('pointwise violation (own source)')
    axes[3].plot(j, [h['e'] for h in hist], 'o-'); axes[3].set_title(r'source change $e_j$')
    axes[4].plot(j, [h['D']/hist[0]['D'] for h in hist], 'o-', label='GCD rerun')
    axes[4].plot(j, [h['Dfix']/hist[0]['D'] for h in hist], 's--', label='fixed projectors')
    axes[4].set_title(r'$D_j/D_0$')
    for ax in axes:
        ax.grid(True); ax.set_xlabel('$j$')
    for ax in [axes[0], axes[1], axes[4]]:
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    return fig

def saveResults(path, hist, fitOut=None, **extra):
    save = {f"hist_{k}": np.array([h[k] for h in hist]) for k in histKeys}
    save['hist_stop'] = hist[-1]['stop']
    save['x'] = np.array([h['x'] for h in hist])
    save['V'] = np.array([h['V'] for h in hist])
    if fitOut:
        save['fit_cases'] = np.array([f"{lab}|{m:g}" for (lab, m) in fitOut])
        save['fit_P'] = np.array([[o[0] for o in v] for v in fitOut.values()])
        save['fit_bin'] = np.array([[o[1] for o in v] for v in fitOut.values()])
        save['fit_rhoBest'] = np.array([v[int(np.argmax([o[1] for o in v]))][3] for v in fitOut.values()])
    save.update(extra)
    np.savez(path, **save)
    print(f"saved {path}")
