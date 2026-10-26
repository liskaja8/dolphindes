"""Scattering bound with regional power balance constraints, the number of regions set by the range of the dissipated power operator.

Copper 2:1 plate (patch21ka1, ka = 1 at 2.45 GHz), RWG basis, Z_mat = Zs L, scattered voltage x = V_sca = -Z0 I as the variable.

1. Number of regions. H = W^H (Re(Zs) L + R0) W is the dissipated (loss + radiation) power operator in the voltage variable, W = Z0^{-1}.
   Its effective range dimension is the number of eigenvalues above the largest gap of its log spectrum (73 for this mesh). The rest
   (160 modes) is the plateau of strongly reactive voltage modes that carry almost no current.
2. Regions. The basis functions are clustered by their edge midpoints into that many compact regions (k-means with a deterministic
   farthest-point initialization).
3. Bound. The scattered power is maximized subject to the real and reactive power balance of every region,
   Re sum_{i in R} conj(p) w_i = 0 for p = 1 and p = j, with w_i = conj(I_i)(V - Ztot I)_i. That is 2 x nReg diagonal projector
   constraints, whose sum contains the global constraints, solved by the dual (Newton). The dual is checked for strong duality: the bound
   must equal the attained P, the residuals must vanish, and the Lagrangian is reported (definite or hard case).
4. Inference per region. Each region is void or metal as a whole.
   - Admittance form: the least-squares density of the region, d_R = clip(Re(sum conj(u_i) Zs L_ii I_i) / sum |u_i|^2, 0, 1), where
     u = V - Ztot I + Zs L_ii I.
   - Impedance form: the region's power balance, sum_{i in R} w_i = Zs (1/d_R - 1) sum L_ii |I_i|^2 for a uniform density, projected onto
     passive d_R in [0, 1].
   Both are rounded over the thresholds and the best binary region design is kept. The relaxed region design (no rounding) and the
   pointwise inference are reported for comparison.

Run from examples/limits:  python regionPowerBalance.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from matplotlib.collections import LineCollection, PolyCollection

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters

matrixSet = "patch21ka1"
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

def VtoI(x):
    return -W @ x

Ifull = np.linalg.solve(Ztot, Vinc)
Pfull = evalObj(Ifull)
Pdip = 3*(c0/frequency)**2/(2*np.pi)*E0**2/(2*eta_0)

Obj = -0.5*Msym(WH @ R0 @ W)
A1 = np.eye(Ndes) + WH @ Zmat.conj().T
A2 = W
s1 = -0.5*Vinc
opt_params = OptimizationHyperparameters(opttol=1e-10, gradConverge=True, min_inner_iter=10, max_restart=40,
                                         penalty_ratio=1e-2, penalty_reduction=0.1, break_iter_period=5, verbose=0)

def localViolation(I):
    return np.conj(I)*(Vinc - Ztot @ I)

def constraintRes(Q, z):
    res = np.real(-z.conj() @ Q._allA_at_v(z) + 2*z.conj() @ Q.Fs)
    res[Q.n_proj_constr:] += Q.c_2j
    return res

def lagSpectrum(Q):
    return np.linalg.eigvalsh(Msym(Q._get_total_A(Q.current_lags)))

# ---------------------------------------------------------------- 1. range dimension of the dissipated power operator
H = Msym(WH @ (Zs.real*Lmat + R0) @ W)
lamH = np.linalg.eigvalsh(H)[::-1]
gaps = np.diff(np.log10(lamH))
nReg = int(np.argmin(gaps[5:-5])) + 6 # eigenvalues above the largest drop of the log spectrum
print(f"dissipated power operator W^H (Re(Zs) L + R0) W: range dimension {nReg} (eigenvalue drop {lamH[nReg - 1]:.2e} -> {lamH[nReg]:.2e}), "
      f"plateau {Ndes - nReg}")

# ---------------------------------------------------------------- 2. regions from the mesh
mesh = sio.loadmat(f"mesh_{matrixSet}.mat")
nodes, tria, bf = mesh["nodes"], mesh["triangles"], mesh["bf"]
mid = 0.5*(nodes[bf[:, 2]] + nodes[bf[:, 3]])

def kmeans(points, k, nIter=200):
    # deterministic farthest-point initialization, then Lloyd iterations; empty clusters are reseeded at the farthest point
    centers = [points[np.argmin(points[:, 0] + points[:, 1])]]
    for _ in range(k - 1):
        dist = np.min(np.linalg.norm(points[:, None, :] - np.array(centers)[None], axis=2), axis=1)
        centers.append(points[np.argmax(dist)])
    centers = np.array(centers)
    for _ in range(nIter):
        label = np.argmin(np.linalg.norm(points[:, None, :] - centers[None], axis=2), axis=1)
        new = centers.copy()
        for j in range(k):
            if np.any(label == j):
                new[j] = points[label == j].mean(axis=0)
            else:
                new[j] = points[np.argmax(np.min(np.linalg.norm(points[:, None, :] - centers[None], axis=2), axis=1))]
        if np.allclose(new, centers):
            break
        centers = new
    return label

region = kmeans(mid, nReg)
sizes = np.bincount(region, minlength=nReg)
print(f"{nReg} regions from the edge midpoints: {sizes.min()}-{sizes.max()} basis functions per region (mean {sizes.mean():.1f})")
Ind = np.zeros((Ndes, nReg))
Ind[np.arange(Ndes), region] = 1.0

# ---------------------------------------------------------------- 3. bounds
def solve(Plist, init_lags, name):
    np.random.seed(seed)
    Q = DenseSharedProjQCQP(Obj, np.zeros(Ndes, dtype=complex), 0.0, A1, s1, Plist, A2=A2, verbose=0)
    t1 = time.time()
    Q.solve_current_dual_problem(method='newton', init_lags=init_lags, opt_params=opt_params)
    x = Q.current_xstar
    I = VtoI(x)
    lam = lagSpectrum(Q)
    res = constraintRes(Q, x)
    P = evalObj(I)
    w = localViolation(I)
    regW = Ind.T @ w # regional complex power balance
    print(f"{name} ({time.time() - t1:.0f}s): {Q.n_proj_constr} constraints, D {Q.current_dual:.4f} W ({Q.current_dual/Pdip:.2f} x resonant dipole), "
          f"P(x*) {P:.4f} W (P/D {P/Q.current_dual:.6f}), min/max eig {lam[0]/lam[-1]:.1e}, ||c||/D {np.linalg.norm(res)/Q.current_dual:.1e}, "
          f"max regional |sum w| / |I^H V| {np.abs(regW).max()/np.abs(I.conj() @ Vinc):.1e}, max pointwise |w_i| / |I^H V| "
          f"{np.abs(w).max()/np.abs(I.conj() @ Vinc):.1e}", flush=True)
    return Q

Qglob = solve([sp.diags(q*np.ones(Ndes, dtype=complex)) for q in [1, 1j]], np.array([1.0, 0.0]), "global (2 constraints)")
PlistReg = [sp.diags(q*Ind[:, j].astype(complex)) for q in [1, 1j] for j in range(nReg)]
Qreg = solve(PlistReg, np.r_[np.ones(nReg), np.zeros(nReg)], f"regional ({nReg} regions)")

# ---------------------------------------------------------------- 4. inference per region
dMin = 1e-6
thsLog = np.logspace(-6, 0, 61)[:-1]

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

def regionAdmittance(I):
    # least-squares uniform density of each region: Zs L_ii I_i = d_R u_i
    u = Vinc - Ztot @ I + Zs*Ldiag*I
    num = Ind.T @ np.real(np.conj(u)*Zs*Ldiag*I)
    den = Ind.T @ np.abs(u)**2
    return np.clip(num/np.maximum(den, 1e-300), 0, 1)

def regionImpedance(I):
    # regional power balance: sum w_i = Zs (1/d_R - 1) sum L_ii |I_i|^2, projected onto passive d_R
    w = localViolation(I)
    t = np.maximum(0, Ind.T @ np.real(np.conj(Zs)*w)/(np.abs(Zs)**2*np.maximum(Ind.T @ (Ldiag*np.abs(I)**2), 1e-300)))
    return 1/(1 + t)

def pointwiseImpedance(I):
    nz = I != 0
    rr = np.zeros(Ndes, dtype=complex)
    rr[nz] = (Vinc - Ztot @ I)[nz]/I[nz]
    t = np.maximum(0, np.real(np.conj(Zs)*rr)/(np.abs(Zs)**2*Ldiag))
    return np.where(nz, 1/(1 + t), 0.0)

def pointwiseAdmittance(I):
    u = Vinc - Ztot @ I + Zs*Ldiag*I
    return np.clip(np.real(np.conj(u)*Zs*Ldiag*I)/np.abs(u)**2, 0, 1)

inference = {}
for qname, Q in [("global", Qglob), ("regional", Qreg)]:
    I = VtoI(Q.current_xstar)
    for form, infer, perRegion in [("region admittance", regionAdmittance, True), ("region impedance", regionImpedance, True),
                                   ("pointwise impedance", pointwiseImpedance, False), ("pointwise admittance", pointwiseAdmittance, False)]:
        dR = infer(I)
        d = dR[region] if perRegion else dR # density per basis function
        b = bestBinary(d)
        inference[qname, form] = dict(d=d, dR=dR if perRegion else None, relaxed=evalObj(realizedCurrent(d)), **b)
        r = inference[qname, form]
        print(f"{qname:8s} {form:20s}: relaxed P/D {r['relaxed']/Qreg.current_dual:.4f} ({r['relaxed']:.3e} W), binary P/D {r['P']/Qreg.current_dual:.4f} "
              f"({r['P']:.3e} W, kept {r['keep'].sum()}/{Ndes} basis functions"
              + (f", {len(np.unique(region[r['keep']]))}/{nReg} regions)" if perRegion else ")"), flush=True)
print(f"(P/D relative to the regional bound {Qreg.current_dual:.4f} W; full plate {Pfull:.3e} W, resonant dipole {Pdip:.3f} W)")

# ---------------------------------------------------------------- figure
def edgePlot(ax, values, title, cmap='viridis', clim=None):
    lc = LineCollection(nodes[bf[:, 2:]], array=values, cmap=cmap, linewidths=2.5)
    if clim is not None:
        lc.set_clim(*clim)
    ax.add_collection(lc)
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    return lc

def structurePlot(ax, keep, title):
    metal = np.zeros(len(tria), dtype=bool)
    metal[bf[keep, :2].ravel()] = True
    ax.add_collection(PolyCollection(nodes[tria[metal]], facecolors='0.4', edgecolors='none'))
    ax.add_collection(PolyCollection(nodes[tria[~metal]], facecolors='white', edgecolors='0.85', linewidths=0.3))
    slot = ~keep & metal[bf[:, 0]] & metal[bf[:, 1]]
    ax.add_collection(LineCollection(nodes[bf[slot, 2:]], colors='r', linewidths=1.5))
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max()); ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)

fig, axes = plt.subplots(3, 3, figsize=(15, 8.5))
perm = np.random.default_rng(1).permutation(nReg) # shuffled colors so that neighbouring regions differ
edgePlot(axes[0, 0], perm[region], f"{nReg} regions (k-means of the edge midpoints)", cmap='tab20')
ax = axes[0, 1]
ax.semilogy(np.arange(1, Ndes + 1), lamH, 'k.', ms=3); ax.axvline(nReg + 0.5, ls=':', c='gray')
ax.set_xlabel('mode (descending)'); ax.set_ylabel('eig W^H (Re Zs L + R0) W'); ax.grid(True, which='both', alpha=0.3)
ax.set_title(f"range dimension {nReg}", fontsize=8)
ax = axes[0, 2]
for qname, Q in [("global", Qglob), ("regional", Qreg)]:
    lam = lagSpectrum(Q)
    ax.semilogy(np.arange(1, Ndes + 1), lam/lam[-1], '.', ms=3, label=f"{qname}, D = {Q.current_dual:.3f} W")
ax.set_xlabel('index'); ax.set_ylabel('eig L* / max'); ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)
for row, qname in [(1, "global"), (2, "regional")]:
    best = max([f for (q, f) in inference if q == qname and f.startswith("region")], key=lambda f: inference[qname, f]["P"])
    r = inference[qname, best]
    lc = edgePlot(axes[row, 0], r["d"], f"{qname} dual current, {best}: relaxed d (P/D {r['relaxed']/Qreg.current_dual:.4f})", clim=(0, 1))
    plt.colorbar(lc, ax=axes[row, 0], shrink=0.8)
    structurePlot(axes[row, 1], r["keep"], f"{qname}, {best}: binary P/D = {r['P']/Qreg.current_dual:.4f}")
    bestPw = max([f for (q, f) in inference if q == qname and f.startswith("pointwise")], key=lambda f: inference[qname, f]["P"])
    structurePlot(axes[row, 2], inference[qname, bestPw]["keep"], f"{qname}, {bestPw}: binary P/D = {inference[qname, bestPw]['P']/Qreg.current_dual:.4f}")
plt.tight_layout()
out = "results/regionPowerBalance.png"
plt.savefig(out, dpi=110)
print(f"figure: {out}")
