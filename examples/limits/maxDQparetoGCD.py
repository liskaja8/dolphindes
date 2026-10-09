"""Directivity-Q Pareto bound of a PEC structure fed by a matched port, dual bound with GCD.

The design region (AToM RWG basis, normalize=false, 1/l_n scaled -> Ohm) is fed
by a delta gap at basis function p with the input voltage Vp. The input
impedance Zin0 fixes the input current I0 = Vp/Zin0 and the radiated power
P = Vp I0/2, so for a self-resonant current

    D = 4 pi U/P = I^H Ud I,   Ud = 4 pi (Fth^H Fth + Fph^H Fph)/(2 Z0vac)/P,
    Q = I^H omW I/(2 I^H R0 I) = Qs I^H omW I,   Qs = 1/(2 Vp I0).

The maximum directivity of a PEC structure is unbounded (R0 is only positive
semidefinite), the stored energy omW > 0 regularizes it. The Pareto front is
bounded by the weighted problems

    g(nu) >= max  D - nu Q = -I^H M I,   M = nu Qs omW - Ud,

so every structure satisfies D <= g(nu) + nu Q for all nu, and
D_ub(Q) = min_nu [g(nu) + nu Q] (Lagrangian envelope of the front).

Degree of freedom: the scattered voltage y = Vsca = -Z0 I as in minQportGCD.py
(PEC, no Lmat), I = -G y, G = Z0^{-1}; y_p = -Vp is eliminated, x = y_r,
I = -G[:, r] x + G[:, p] Vp.

    max  -x^H A0 x + 2 Re[x^H s0] + c0      A0 = Gr^H M Gr, s0 = Gr^H M g, c0 = -g^H M g
    s.t. Re[x^H P^H (-G_rr x + g_r)] = 0    A1 = G_rr^H, A2 = 1, s1 = g_r/2
         I_p = I0                           (Zin = Zin0, two linear constraints)

For nu >= nuMax = max eig(Ud, Qs omW), A0 > 0 and the zero multipliers are dual
feasible; below nuMax the global real power multiplier is increased until the
Lagrange matrix is positive definite.

Usage
-----
    python maxDQparetoGCD.py in.mat out.mat  # in.mat from maxDQparetoGCD.m
    NGCD=20 NU=12 NULOG=-2.5,1.5 python maxDQparetoGCD.py ...

in.mat:  R0, X0, omW [N x N] (Ohm), Fth, Fph [1 x N], port (0 based), I0, Vp, Ngcd
out.mat: nu, nuMax, gGlobal, gGCD [1 x nNu], Iopt, Vsca [N x nNu] (GCD dual optimal),
         Dopt, Qopt, ZinOpt, lagsGlobal, lagsGCD (cells), Lstar [nNu x N-1 x N-1],
         minEigGCD, Ngcd
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

Z0VAC = 376.730313668


def Msym(M):
    return 0.5 * (M.conj().T + M)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    inFile = argv[1] if len(argv) >= 2 else os.path.join(here, "maxDQpareto_patch21ka1_gcd20_in.mat")
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    omW = np.asarray(data["omW"], dtype=float)
    Fth = np.asarray(data["Fth"], dtype=complex).ravel()
    Fph = np.asarray(data["Fph"], dtype=complex).ravel()
    p = int(data["port"])
    I0 = float(data["I0"])
    Vp = float(data["Vp"])
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    nNu = int(os.environ.get("NU", 12))
    nuLog = [float(s) for s in os.environ.get("NULOG", "-2.5,1.5").split(",")]
    N = R0.shape[0]
    r = np.setdiff1d(np.arange(N), [p])
    P = Vp * I0 / 2
    Qs = 1 / (2 * Vp * I0)

    Z0 = R0 + 1j * X0
    G = np.linalg.inv(Z0)
    Gr = G[:, r]
    g = G[:, p] * Vp  # I = -Gr x + g

    Ud = Msym(4 * np.pi * (np.outer(Fth.conj(), Fth) + np.outer(Fph.conj(), Fph)) / (2 * Z0VAC) / P)
    nuMax = np.max(la.eigh(Ud, Qs * omW, eigvals_only=True))
    nuList = nuMax * np.logspace(nuLog[0], nuLog[1], nNu)
    print(f"N = {N}, port {p}, Vp = {Vp} V, I0 = {I0} A, Zin = {Vp/I0} Ohm, Ngcd = {Ngcd}, "
          f"nuMax = max D/Q (arbitrary current) = {nuMax:.6g}, {nNu} weights "
          f"nuMax*10^[{nuLog[0]}, {nuLog[1]}]")

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
    gcd_opt_params = dataclasses.replace(
        opt_params, opttol=float(os.environ.get("GCD_OPTTOL", opt_params.opttol)))

    def DQ(I):
        D = np.real(I.conj() @ Ud @ I) * P / (0.5 * np.real(I.conj() @ R0 @ I))
        Q = 0.5 * np.real(I.conj() @ omW @ I + abs(I.conj() @ X0 @ I)) / np.real(I.conj() @ R0 @ I)
        return D, Q

    Ir = np.ones(N - 1, dtype=complex)
    out = {k: [] for k in ["gGlobal", "gGCD", "Iopt", "Vsca", "Dopt", "Qopt", "ZinOpt",
                           "lagsGlobal", "lagsGCD", "Lstar", "minEigGCD", "lam0"]}
    for iNu, nu in enumerate(nuList):
        M = Msym(nu * Qs * omW - Ud)
        A0 = Msym(Gr.conj().T @ M @ Gr)
        s0 = Gr.conj().T @ M @ g
        c0 = -np.real(g.conj() @ M @ g)

        def makeQCQP(Plist):
            return DenseSharedProjQCQP(A0, s0, c0, A1, s1, Plist, A2=np.eye(N - 1, dtype=complex),
                                       B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0)

        def report(name, QCQP):
            I = -Gr @ QCQP.current_xstar + g
            D, Q = DQ(I)
            gd = QCQP.current_dual
            print(f"  {name}: g = {gd:.6g}, D(Iopt) = {D:.5g}, Q(Iopt) = {Q:.5g}, "
                  f"D - nu Q = {D - nu*Q:.5g}, Zin(Iopt) = {Vp/I[p]:.3f}, "
                  f"{QCQP.n_proj_constr} projectors", flush=True)
            return gd

        # global bound, dual feasible start by the real power multiplier
        QCQP = makeQCQP([sp.diags(Ir), sp.diags(1j * Ir)])
        init = np.zeros(4)
        lam = 0.0
        while not QCQP.is_dual_feasible(init):
            lam = 1e-3 * nu if lam == 0 else 2 * lam
            init[0] = lam
            if lam > 1e12:
                raise RuntimeError(f"no dual feasible start for nu = {nu:.4g}")
        print(f"nu = {nu:.5g} ({iNu+1}/{nNu}), feasible start lambda_Re = {lam:.3g}", flush=True)
        t1 = time.time()
        QCQP.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
        gGlobal = report(f"global ({time.time()-t1:.1f}s)", QCQP)

        # GCD refinement
        gQCQP = copy.deepcopy(QCQP)
        gcd_params = GCDHyperparameters(max_proj_cstrt_num=2 + 2 * Ngcd,
            max_gcd_iter_num=Ngcd,
            gcd_iter_period=Ngcd + 1,
            method="newton",
            opt_params=gcd_opt_params)
        t1 = time.time()
        gQCQP.run_gcd(gcd_params)
        gQCQP.solve_current_dual_problem(method="newton", init_lags=gQCQP.current_lags,
                                         opt_params=opt_params)
        gGCD = report(f"GCD ({time.time()-t1:.1f}s)", gQCQP)

        x = gQCQP.current_xstar
        I = -Gr @ x + g
        Vsca = np.empty(N, dtype=complex)
        Vsca[r] = x
        Vsca[p] = -Vp
        Lstar = Msym(gQCQP._get_total_A(gQCQP.current_lags))
        D, Q = DQ(I)
        out["gGlobal"].append(gGlobal)
        out["gGCD"].append(gGCD)
        out["Iopt"].append(I)
        out["Vsca"].append(Vsca)
        out["Dopt"].append(D)
        out["Qopt"].append(Q)
        out["ZinOpt"].append(Vp / I[p])
        out["lagsGlobal"].append(np.asarray(QCQP.current_lags))
        out["lagsGCD"].append(np.asarray(gQCQP.current_lags))
        out["Lstar"].append(Lstar)
        out["minEigGCD"].append(np.min(np.linalg.eigvalsh(Lstar)))
        out["lam0"].append(lam)

    res = dict(nu=nuList, nuMax=nuMax, Ngcd=Ngcd,
               gGlobal=np.array(out["gGlobal"]), gGCD=np.array(out["gGCD"]),
               Iopt=np.array(out["Iopt"]).T, Vsca=np.array(out["Vsca"]).T,
               Dopt=np.array(out["Dopt"]), Qopt=np.array(out["Qopt"]),
               ZinOpt=np.array(out["ZinOpt"]), minEigGCD=np.array(out["minEigGCD"]),
               lam0=np.array(out["lam0"]), Lstar=np.array(out["Lstar"]),
               lagsGlobal=np.array(out["lagsGlobal"], dtype=object),
               lagsGCD=np.array(out["lagsGCD"], dtype=object))
    sio.savemat(outFile, res)
    print(f"saved {outFile}")
    return res


if __name__ == "__main__":
    main(sys.argv)
