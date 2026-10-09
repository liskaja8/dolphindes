"""Minimum Q of a PEC structure fed by a port of given input impedance, dual bound with GCD.

The design region (AToM RWG basis, normalize=true -> Ohm) is fed by a delta gap
at basis function p with the input voltage Vp. The input impedance Zin0 fixes
the input current I0 = Vp/Zin0 and the radiated power P = Vp I0/2, so for a
self-resonant current Q = I^H omW I/(2 I^H R0 I) = I^H omW I/(2 I0).

Degree of freedom: the scattered voltage y = Vsca = -Z0 I (PEC, no Lmat), so
I = -G y with G = Z0^{-1}. The port row of the PEC equation, (Z0 I)_p = Vp,
reads y_p = -Vp, hence y_p is eliminated, x = y_r, I = -G[:, r] x + G[:, p] Vp.

    max  -Q = -x^H A0 x + 2 Re[x^H s0] + c0
    s.t. Re[I_r^H P (V_r + x)] = 0, V_r = 0     (PEC: metal V + Vsca = 0, void I = 0)
         I_p = I0                                (Zin = Zin0, two linear constraints)

The constraints are written directly from the AToM equations (Re[.] algebra,
no U = -jZ mapping needed): Re[x^H P^H (-G_rr x + g_r)] with A1 = G_rr^H,
A2 = 1, s1 = g_r/2. The projector constraints start from the global Re/Im pair
and are refined by GCD; the port constraint is a fixed general constraint
(B_j = 0). All dual solves use the dense Cholesky factorization of the
Lagrange matrix (DenseSharedProjQCQP, Newton).

Usage
-----
    python minQportGCD.py in.mat out.mat     # in.mat from minQportGCD.m
    python minQportGCD.py                    # minQport_patch21ka1_in.mat -> *_out.mat
    LOCAL_ONLY=1 python minQportGCD.py ...   # add IoptLocal to an existing out.mat

in.mat:  R0, X0, omW [N x N] (normalized, Ohm), port (0 based), I0, Vp, Ngcd
out.mat: QlbGlobal, QlbGCD, QlbLocal, Iopt, Vsca (GCD dual optimal),
         IoptLocal, VscaLocal (all-local dual optimal), lags*, Pdiags,
         minEigGCD, QIopt, ZinIopt, residuals
"""

import copy
import dataclasses
import os
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la
import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters


def Msym(M):
    return 0.5 * (M.conj().T + M)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    inFile = argv[1] if len(argv) >= 2 else os.path.join(here, "minQport_patch21ka1_in.mat")
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    omW = np.asarray(data["omW"], dtype=float)
    p = int(data["port"])
    I0 = float(data["I0"])
    Vp = float(data["Vp"])
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    doLocal = os.environ.get("LOCAL", "1") == "1"
    N = R0.shape[0]
    r = np.setdiff1d(np.arange(N), [p])
    Qscale = 1 / (2 * Vp * I0)  # Q = Qscale * I^H omW I
    print(f"N = {N}, port {p}, Vp = {Vp} V, I0 = {I0} A, Zin = {Vp/I0} Ohm, Ngcd = {Ngcd}, "
          f"opttol = {os.environ.get('OPTTOL', 1e-8)}, maxproj = {os.environ.get('MAXPROJ', 'all')}")

    Z0 = R0 + 1j * X0
    G = np.linalg.inv(Z0)
    Gr = G[:, r]
    g = G[:, p] * Vp  # I = -Gr x + g

    # objective -Q
    A0 = Qscale * Msym(Gr.conj().T @ omW @ Gr)
    s0 = Qscale * Gr.conj().T @ omW @ g
    c0 = -Qscale * np.real(g.conj() @ omW @ g)
    la.cho_factor(A0)  # zero multipliers must be dual feasible (Cholesky of A0)

    # PEC constraints Re[x^H P^H (-G_rr x + g_r)] = 0
    A1 = G[np.ix_(r, r)].conj().T
    s1 = g[r] / 2

    # port current I_p = a x + b + I0 = I0 -> Re/Im[a x + b] = 0
    a = -G[p, r]
    b = g[p] - I0
    B_j = [np.zeros((N - 1, N - 1), dtype=complex)] * 2
    s_2j = [a.conj() / 2, 1j * a.conj() / 2]
    c_2j = [np.real(b), np.imag(b)]

    opt_params = OptimizationHyperparameters(opttol=float(os.environ.get("OPTTOL", 1e-8)),
        gradConverge=True,
        min_inner_iter=10,
        max_restart=10,
        penalty_ratio=1e-2,
        penalty_reduction=0.1,
        break_iter_period=5,
        verbose=0)

    # inner GCD re-solves at GCD_OPTTOL, the final solves at OPTTOL (tight tolerance
    # matters for xstar, the Lagrange matrix is nearly singular at the optimum)
    gcd_opt_params = dataclasses.replace(
        opt_params, opttol=float(os.environ.get("GCD_OPTTOL", opt_params.opttol)))

    def makeQCQP(Plist):
        return DenseSharedProjQCQP(A0, s0, c0, A1, s1, Plist, A2=np.eye(N - 1, dtype=complex),
                                   B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0)

    def current(QCQP):
        return -Gr @ QCQP.current_xstar + g

    def report(name, QCQP):
        I = current(QCQP)
        Qd = -QCQP.current_dual
        QI = 0.5 * np.real(I.conj() @ omW @ I + abs(I.conj() @ X0 @ I)) / np.real(I.conj() @ R0 @ I)
        print(f"{name}: Q_lb = {Qd:.6g}, Q(Iopt) = {QI:.6g}, Zin(Iopt) = {Vp/I[p]:.4f}, "
              f"{QCQP.n_proj_constr} projectors")
        return Qd

    def localQCQP():
        Pl = [sp.csc_array(([q], ([i], [i])), shape=(N - 1, N - 1), dtype=complex)
              for i in range(N - 1) for q in (1, 1j)]
        return makeQCQP(Pl)

    def fullBasis(QCQP):
        Vsca = np.empty(N, dtype=complex)
        Vsca[r] = QCQP.current_xstar
        Vsca[p] = -Vp
        return current(QCQP).reshape(-1, 1), Vsca.reshape(-1, 1)

    if os.environ.get("LOCAL_ONLY", "0") == "1":
        # re-solve the all-local bound and add its dual optimum to an existing out.mat,
        # cold start (zero lags) unless WARM=1 (from lagsLocal); the Lagrange matrix
        # is nearly singular at the optimum, so xstar depends on the solver path
        out = {}
        if os.path.isfile(outFile):
            out = sio.loadmat(outFile, squeeze_me=True)
            out = {k: v for k, v in out.items() if not k.startswith("__")}
        lQCQP = localQCQP()
        warm = os.environ.get("WARM", "0") == "1" and "lagsLocal" in out
        init = np.asarray(out["lagsLocal"]) if warm else np.zeros(2 * (N - 1) + 2)
        t1 = time.time()
        lQCQP.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
        out["QlbLocal"] = report(f"local, {'warm' if warm else 'cold'} start "
                                 f"({time.time()-t1:.1f}s)", lQCQP)
        out["lagsLocal"] = np.asarray(lQCQP.current_lags).reshape(-1, 1)
        out["IoptLocal"], out["VscaLocal"] = fullBasis(lQCQP)
        sio.savemat(outFile, out)
        print(f"saved {outFile}")
        return out

    # global bound (Re/Im power), zero multipliers are feasible
    Ir = np.ones(N - 1, dtype=complex)
    QCQP = makeQCQP([sp.diags(Ir), sp.diags(1j * Ir)])
    t1 = time.time()
    QCQP.solve_current_dual_problem(method="newton", init_lags=np.zeros(4), opt_params=opt_params)
    QlbGlobal = report(f"global ({time.time()-t1:.1f}s)", QCQP)
    lagsGlobal = QCQP.current_lags

    # GCD refinement
    gQCQP = copy.deepcopy(QCQP)
    # MAXPROJ caps the projector number (older ones are merged), default keeps all
    maxProj = int(os.environ.get("MAXPROJ", 2 + 2 * Ngcd))
    gcd_params = GCDHyperparameters(max_proj_cstrt_num=maxProj,
        max_gcd_iter_num=Ngcd,
        gcd_iter_period=Ngcd + 1,
        method="newton",
        opt_params=gcd_opt_params)
    t1 = time.time()
    gQCQP.run_gcd(gcd_params)
    gQCQP.solve_current_dual_problem(method="newton", init_lags=gQCQP.current_lags,
                                     opt_params=opt_params)
    QlbGCD = report(f"GCD ({time.time()-t1:.1f}s)", gQCQP)

    # all-local bound (Re/Im per BF), reference for the GCD tightness
    QlbLocal = np.nan
    lagsLocal = np.zeros(0)
    IoptLocal = VscaLocal = np.zeros((N, 1), dtype=complex)
    if doLocal:
        lQCQP = localQCQP()
        t1 = time.time()
        lQCQP.solve_current_dual_problem(method="newton", init_lags=np.zeros(2 * (N - 1) + 2),
                                         opt_params=opt_params)
        QlbLocal = report(f"local ({time.time()-t1:.1f}s)", lQCQP)
        lagsLocal = lQCQP.current_lags
        IoptLocal, VscaLocal = fullBasis(lQCQP)

    # dual optimum of GCD, back to the full RWG basis
    x = gQCQP.current_xstar
    Vsca = np.empty(N, dtype=complex)
    Vsca[r] = x
    Vsca[p] = -Vp
    Iopt = current(gQCQP)
    Lh = Msym(gQCQP._get_total_A(gQCQP.current_lags))
    minEig = np.min(np.linalg.eigvalsh(Lh))
    V = np.zeros(N, dtype=complex)
    V[p] = Vp
    res = Iopt.conj() * (V + Vsca)
    print(f"min eig Lagrange matrix {minEig:.3e}, |I_p - I0| = {abs(Iopt[p]-I0):.2e}, "
          f"global Re/Im residual {np.sum(res.real):.2e} {np.sum(res.imag):.2e} "
          f"(of Vp I0 = {Vp*I0:.2e}), max local |I^*(V+Vsca)| {np.max(abs(res)):.2e}")
    QIopt = 0.5 * np.real(Iopt.conj() @ omW @ Iopt + abs(Iopt.conj() @ X0 @ Iopt)) \
        / np.real(Iopt.conj() @ R0 @ Iopt)

    out = dict(QlbGlobal=QlbGlobal, QlbGCD=QlbGCD, QlbLocal=QlbLocal,
               Iopt=Iopt.reshape(-1, 1), Vsca=Vsca.reshape(-1, 1),
               lagsGlobal=np.asarray(lagsGlobal).reshape(-1, 1),
               lagsGCD=np.asarray(gQCQP.current_lags).reshape(-1, 1),
               lagsLocal=np.asarray(lagsLocal).reshape(-1, 1),
               IoptLocal=IoptLocal, VscaLocal=VscaLocal,
               Pdiags=gQCQP.Proj.get_Pdata_column_stack(), minEigGCD=minEig,
               QIopt=QIopt, ZinIopt=Vp / Iopt[p], localResidual=res.reshape(-1, 1),
               Ngcd=Ngcd)
    sio.savemat(outFile, out)
    print(f"saved {outFile}")
    return out


if __name__ == "__main__":
    main(sys.argv)
