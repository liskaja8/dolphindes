"""Upper bound on the radiated power of a PEC structure fed by a port, global constraints.

Same design region and port as minQportGCD.py (in.mat from minQportGCD.m), but the
port voltage Vp is the only excitation and the port current is free. For a lossless
structure the radiated (scattered) power equals the input power,
P = 1/2 I^H R0 I = 1/2 Vp Re[I_p], so a real input impedance Zin0 requires
P = Vp^2/(2 Zin0) (5 mW for 1 V and 100 Ohm). If the bound is below this value,
Zin0 is infeasible.

R0 is singular (only a few radiating modes at small ka), so the maximization of
the convex objective has no dual feasible point in the full RWG space. The current
is restricted to the range of R0, I = U c, U the eigenvectors of R0 with eigenvalues
above RTOL*max (sweep), where Lambda_R = U^H R0 U > 0:

    max  P = 1/2 c^H Lambda_R c
    s.t. Re[c^H P^H (U^H V - Zc c)] = 0,  P in {1, j}   (global real/reactive power)
         Im[I_p] = 0                                   (optional, RES=1: resonant port)

with Zc = U^H Z0 U (A1 = Zc^H, A2 = 1, s1 = U^H V/2). The real power multiplier
lambda = 1 gives the Lagrange matrix Lambda_R/2 > 0. The real power constraint alone
gives P <= 1/2 Vp^2 [R0^+]_pp (R0^+ the pseudoinverse on the same range).

Usage
-----
    python maxPportGlobal.py in.mat [Zin0]
"""

import sys

import numpy as np
import scipy.io as sio
import scipy.linalg as la
import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters


def main(argv):
    data = sio.loadmat(argv[1], squeeze_me=True)
    Zin0 = float(argv[2]) if len(argv) >= 3 else 100.0
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    p = int(data["port"])
    Vp = float(data["Vp"])
    N = R0.shape[0]
    Preq = Vp**2 / (2 * Zin0)
    V = np.zeros(N, dtype=complex)
    V[p] = Vp
    Z0 = R0 + 1j * X0
    w, Uall = la.eigh(R0)
    print(f"N = {N}, port {p}, Vp = {Vp} V, required P = {Preq*1e3:.4g} mW for Zin = {Zin0} Ohm")
    print(f"R0 eigenvalues / max: {', '.join(f'{x:.1e}' for x in w[::-1][:8] / w[-1])}, ...")
    opt_params = OptimizationHyperparameters(opttol=1e-11, gradConverge=True, min_inner_iter=10,
        max_restart=10, penalty_ratio=1e-2, penalty_reduction=0.1, break_iter_period=5, verbose=0)

    print(f"{'RTOL':>7} {'K':>4} {'Re only':>11} {'Re+Im':>11} {'Re+Im+res':>11}  (mW)  "
          f"Zin at Re+Im+res optimum")
    for rtol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        keep = w > rtol * w[-1]
        U = Uall[:, keep]
        K = U.shape[1]
        Lam = w[keep]
        Zc = U.conj().T @ Z0 @ U
        Zc = 0.5 * (Zc + Zc.T)
        v = U.conj().T @ V
        PreOnly = 0.5 * np.real(v.conj() @ (v / Lam))      # 1/2 Vp^2 [R0^+]_pp
        A0 = -0.5 * np.diag(Lam).astype(complex)
        s0 = np.zeros(K, dtype=complex)
        IK = np.ones(K, dtype=complex)
        Plist = [sp.diags(IK), sp.diags(1j * IK)]
        res = []
        for resonant in (False, True):
            if resonant:  # Im[a c] = 0, a = U[p, :]
                a = U[p, :].astype(complex)
                gen = dict(B_j=[np.zeros((K, K), dtype=complex)], s_2j=[1j * a.conj() / 2],
                           c_2j=[0.0])
                init = np.array([1.0, 0.0, 0.0])
            else:
                gen = {}
                init = np.array([1.0, 0.0])
            Q = DenseSharedProjQCQP(A0, s0, 0.0, Zc.conj().T, v / 2, Plist,
                                    A2=np.eye(K, dtype=complex), verbose=0, **gen)
            Q.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
            I = U @ Q.current_xstar
            res.append((Q.current_dual, Vp / I[p] if abs(I[p]) > 0 else np.nan))
        print(f"{rtol:7.0e} {K:4d} {PreOnly*1e3:11.4g} {res[0][0]*1e3:11.4g} {res[1][0]*1e3:11.4g}"
              f"  {'' if res[1][0] >= Preq else 'INFEASIBLE '}Zin = {res[1][1]:.4g}")


if __name__ == "__main__":
    main(sys.argv)
