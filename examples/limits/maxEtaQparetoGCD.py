"""Radiation efficiency-Q Pareto bound of a copper structure fed by a port, dual bound with GCD.

Same design region and port as minQportGCD.py (in.mat from minQportGCD.m, with the
copper loss matrix Lmat and surface impedance Zs). The input voltage Vp and the input
impedance Zin0 fix the input current I0 = Vp/Zin0 and the input power
Pin = Vp I0/2 = Prad + Ploss. With x = I/I0, v = V/I0 and Z = Z0 + Zs Lmat,

    eta = Prad/Pin = x^H R0 x/Zin0,
    Q   = 2 w W/Pin = x^H omW x/(2 Zin0)      (self-resonant: Zin = Zin0 real),

so Q counts the radiated and the dissipated power. Every structure satisfies
eta <= g(nu) + nu Q with

    g(nu) >= max  eta - nu Q = -x^H M x,   M = nu omW/(2 Zin0) - R0/Zin0,
             s.t. Re[x^H P^H (v - Z x)] = 0      A1 = Z^H, A2 = 1, s1 = v/2
                  x_p = 1                        (Zin = Zin0, two linear constraints)

and the front is bounded by eta_ub(Q) = min_nu [g(nu) + nu Q]. Every structure
(subset of enabled basis functions, port kept) satisfies conj(x_i)(v - Z x)_i = 0 at
every basis function. The projector constraints start from the global Re/Im pair
and are refined by GCD, optionally all local (LOCAL=1). Copper makes
Re[Z] = R0 + Re[Zs] Lmat > 0, so the global real power multiplier
lambda >= 1/Zin0 gives a dual feasible start for every nu.

Usage
-----
    python maxEtaQparetoGCD.py in.mat out.mat
    NGCD=50 NU=16 NULOG=-5,0 LOCAL=1 python maxEtaQparetoGCD.py ...
    NU_IDX=k python maxEtaQparetoGCD.py in.mat out.mat   # weight k only (0 based, job array)
                                                       # -> out_nuKK.npz next to out.mat
    MERGE=1 python maxEtaQparetoGCD.py in.mat out.mat    # out_nu*.npz -> out.mat

out.mat: nu, nuMax, gGlobal, gGCD, gLocal [1 x nNu], Iopt, IoptLocal [N x nNu] (A),
         etaOpt, Qopt, ZinOpt (GCD dual optimal current), etaLocal, QLocal, lags*,
         minEigGCD, lam0, Zs, Ngcd
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


KEYS = ["gGlobal", "gGCD", "gLocal", "Iopt", "IoptLocal", "etaOpt", "Qopt", "ZinOpt",
        "etaLocal", "QLocal", "lagsGlobal", "lagsGCD", "lagsLocal", "minEigGCD", "lam0"]


def main(argv):
    inFile = argv[1]
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_etaQ_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    omW = np.asarray(data["omW"], dtype=float)
    Lmat = np.asarray(data["Lmat"], dtype=float)
    Zs = complex(data["Zs"])
    p = int(data["port"])
    I0 = float(data["I0"])
    Vp = float(data["Vp"])
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    doLocal = os.environ.get("LOCAL", "0") == "1"
    nNu = int(os.environ.get("NU", 16))
    nuLog = [float(s) for s in os.environ.get("NULOG", "-5,0").split(",")]
    N = R0.shape[0]
    Zin0 = Vp / I0

    Z = R0 + 1j * X0 + Zs * Lmat
    Rloss = np.real(Zs) * Lmat
    v = np.zeros(N, dtype=complex)
    v[p] = Zin0  # V/I0

    # max eta/Q of an arbitrary current (no constraints), the weights are relative to it
    nuMax = np.max(la.eigh(2 * R0, omW, eigvals_only=True))
    nuList = nuMax * np.logspace(nuLog[0], nuLog[1], nNu)
    partFile = outFile.replace(".mat", "_nu{:02d}.npz")
    if os.environ.get("MERGE", "0") == "1":
        parts = [np.load(partFile.format(k), allow_pickle=True) for k in range(nNu)]
        out = {key: [part[key][()] for part in parts] for key in KEYS}
        return save(outFile, nuList, nuMax, Ngcd, Zs, I0, out)
    idxList = [int(os.environ["NU_IDX"])] if "NU_IDX" in os.environ else range(nNu)

    print(f"N = {N}, port {p}, Vp = {Vp} V, I0 = {I0} A, Zin = {Zin0} Ohm, Zs = {Zs:.4e} Ohm, "
          f"Ngcd = {Ngcd}, local = {doLocal}, opttol = {os.environ.get('OPTTOL', 1e-8)}, "
          f"nuMax = max eta/Q (arbitrary current) = {nuMax:.6g}, {nNu} weights "
          f"nuMax*10^[{nuLog[0]}, {nuLog[1]}]", flush=True)

    A1 = Z.conj().T
    s1 = v / 2
    ep = np.zeros(N, dtype=complex)
    ep[p] = 1
    B_j = [np.zeros((N, N), dtype=complex)] * 2
    s_2j = [ep / 2, 1j * ep / 2]
    c_2j = [-1.0, 0.0]

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

    def etaQ(x):
        """Efficiency and Q of a current (Q including the tuning reactance), x = I/I0."""
        Prad = np.real(x.conj() @ R0 @ x)
        Ploss = np.real(x.conj() @ Rloss @ x)
        eta = Prad / (Prad + Ploss)
        Q = 0.5 * np.real(x.conj() @ omW @ x + abs(x.conj() @ X0 @ x)) / (Prad + Ploss)
        return eta, Q

    In = np.ones(N, dtype=complex)
    out = {k: [] for k in KEYS}
    for iNu in idxList:
        nu = nuList[iNu]
        A0 = Msym(nu * omW / (2 * Zin0) - R0 / Zin0)

        def makeQCQP(Plist):
            return DenseSharedProjQCQP(A0, np.zeros(N, dtype=complex), 0.0, A1, s1, Plist,
                                       A2=np.eye(N, dtype=complex),
                                       B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0)

        def report(name, QCQP):
            x = QCQP.current_xstar
            eta, Q = etaQ(x)
            gd = QCQP.current_dual
            print(f"  {name}: g = {gd:.6g}, eta(Iopt) = {eta:.5g}, Q(Iopt) = {Q:.5g}, "
                  f"eta - nu Q = {eta - nu*Q:.5g}, Zin(Iopt) = {Zin0/x[p]:.3f}, "
                  f"{QCQP.n_proj_constr} projectors", flush=True)
            return gd

        # global bound, dual feasible start by the real power multiplier
        QCQP = makeQCQP([sp.diags(In), sp.diags(1j * In)])
        init = np.zeros(4)
        lam = 0.0
        while not QCQP.is_dual_feasible(init):
            lam = 1 / Zin0 if lam == 0 else 2 * lam
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

        # all-local bound (Re/Im per BF), warm start from the real power multiplier
        gLocal = np.nan
        xL = np.zeros(N, dtype=complex)
        lagsL = np.zeros(0)
        if doLocal:
            Pl = [sp.csc_array(([q], ([i], [i])), shape=(N, N), dtype=complex)
                  for i in range(N) for q in (1, 1j)]
            lQCQP = makeQCQP(Pl)
            initL = np.zeros(2 * N + 2)
            initL[0:2 * N:2] = lam  # lam on every local real power constraint = lam globally
            t1 = time.time()
            lQCQP.solve_current_dual_problem(method="newton", init_lags=initL,
                                             opt_params=opt_params)
            gLocal = report(f"local ({time.time()-t1:.1f}s)", lQCQP)
            xL = lQCQP.current_xstar
            lagsL = np.asarray(lQCQP.current_lags)

        x = gQCQP.current_xstar
        eta, Q = etaQ(x)
        etaL, QL = etaQ(xL) if doLocal else (np.nan, np.nan)
        Lstar = Msym(gQCQP._get_total_A(gQCQP.current_lags))
        for k, val in [("gGlobal", gGlobal), ("gGCD", gGCD), ("gLocal", gLocal),
                       ("Iopt", I0 * x), ("IoptLocal", I0 * xL), ("etaOpt", eta), ("Qopt", Q),
                       ("ZinOpt", Zin0 / x[p]), ("etaLocal", etaL), ("QLocal", QL),
                       ("lagsGlobal", np.asarray(QCQP.current_lags)),
                       ("lagsGCD", np.asarray(gQCQP.current_lags)), ("lagsLocal", lagsL),
                       ("minEigGCD", np.min(np.linalg.eigvalsh(Lstar))), ("lam0", lam)]:
            out[k].append(val)

    if "NU_IDX" in os.environ:
        part = partFile.format(idxList[0])
        np.savez(part, nuIdx=idxList[0], **{k: np.asarray(v[0]) for k, v in out.items()})
        print(f"saved {part}")
        return out
    return save(outFile, nuList, nuMax, Ngcd, Zs, I0, out)


def save(outFile, nuList, nuMax, Ngcd, Zs, I0, out):
    res = dict(nu=nuList, nuMax=nuMax, Ngcd=Ngcd, Zs=Zs, I0=I0,
               **{k: np.array(out[k]) for k in ["gGlobal", "gGCD", "gLocal", "etaOpt", "Qopt",
                                                "ZinOpt", "etaLocal", "QLocal", "minEigGCD",
                                                "lam0"]},
               Iopt=np.array(out["Iopt"]).T, IoptLocal=np.array(out["IoptLocal"]).T,
               **{k: np.array(out[k], dtype=object) for k in ["lagsGlobal", "lagsGCD",
                                                              "lagsLocal"]})
    sio.savemat(outFile, res)
    print(f"saved {outFile}")
    return res


if __name__ == "__main__":
    main(sys.argv)
