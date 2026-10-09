"""Maximum broadside gain of a copper structure fed by a matched port, dual bound with GCD.

The design region (AToM RWG basis, normalize=false) is fed by a delta gap at
basis function p with the input voltage Vp. The input impedance Zin0 fixes the
input current I0 = Vp/Zin0 and the input power Pin = Vp I0/2 (global real power
constraint), so the gain G = 4 pi U/Pin is the realized gain into Zin0.

Degree of freedom: the current in the RWG basis (non-diagonal material
operator, main.tex Sec. "Inference with Non-Diagonal Material Operator"),
scaled by the port current, x = I/I0, with Z = Z0 + Zs L and v = V/I0:

    max  G = x^H Ug x                 Ug = 4 pi I0^2/Pin (Fth^H Fth + Fph^H Fph)/(2 Z0vac)
    s.t. Re[x^H P^H (v - Z x)] = 0    A1 = Z^H, A2 = 1, s1 = v/2
         x_p = 1                      (Zin = Zin0, two linear constraints)

Every structure (subset of enabled basis functions, port kept) satisfies
conj(x_i)(v - Z x)_i = 0 at every basis function. The projector constraints
start from the global Re/Im pair and are refined by GCD; the port constraint is
a fixed general constraint (B_j = 0). The objective is convex, so the zero
multipliers are not dual feasible. With gMax, the maximum gain of an arbitrary
current (generalized eigenvalue of Ug and Re[Z]/Zin0), Ug <= gMax/Zin0 Re[Z], so
the global real power multiplier lambda = 2 gMax/Zin0 is a dual feasible start.

Usage
-----
    python maxGainGCD.py in.mat out.mat      # in.mat from maxGainGCD.m
    python maxGainGCD.py                     # maxGain_patch21ka1_in.mat -> *_out.mat
    NGCD=50 LOCAL=1 python maxGainGCD.py ... # GCD iterations, add all-local bound

in.mat:  R0, X0, Lmat [N x N], Zs, Fth, Fph [1 x N], port (0 based), I0, Vp, Ngcd
out.mat: GlbGlobal, GlbGCD, GlbLocal, Iopt (A, GCD dual optimal), xopt = Iopt/I0,
         xstar (solver xopt before the null-space shift),
         Lstar (Hermitian Lagrange matrix in x), lags*, Pdiags, minEigGCD,
         GIopt, ZinIopt, IoptLocal, gMax (generalized eigenvalue, no constraints)
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
from scipy.optimize import least_squares

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters

Z0VAC = 376.730313668


def Msym(M):
    return 0.5 * (M.conj().T + M)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    inFile = argv[1] if len(argv) >= 2 else os.path.join(here, "maxGain_patch21ka1_in.mat")
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    L = np.asarray(data["Lmat"], dtype=float)
    Zs = complex(data["Zs"])
    Fth = np.asarray(data["Fth"], dtype=complex).ravel()
    Fph = np.asarray(data["Fph"], dtype=complex).ravel()
    p = int(data["port"])
    I0 = float(data["I0"])
    Vp = float(data["Vp"])
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    doLocal = os.environ.get("LOCAL", "1") == "1"
    N = R0.shape[0]
    Pin = Vp * I0 / 2
    print(f"N = {N}, port {p}, Vp = {Vp} V, I0 = {I0} A, Zin = {Vp/I0} Ohm, Zs = {Zs:.4e} Ohm, "
          f"Ngcd = {Ngcd}, opttol = {os.environ.get('OPTTOL', 1e-8)}")

    Z = R0 + 1j * X0 + Zs * L
    v = np.zeros(N, dtype=complex)
    v[p] = Vp / I0

    # gain G = 4 pi U/Pin = x^H Ug x
    Ug = 4 * np.pi * I0**2 / Pin * (np.outer(Fth.conj(), Fth) + np.outer(Fph.conj(), Fph)) / (2 * Z0VAC)
    Ug = Msym(Ug)
    Rz = np.real(Z)  # Hermitian part of A1 = Z^H (Z complex symmetric)
    Rz = 0.5 * (Rz + Rz.T)
    # max gain of an arbitrary current: G = 4 pi U/(I^H Rz I/2) = x^H Ug x Pin/(I0^2 x^H Rz x/2)
    gMax = np.max(la.eigh(Ug, Rz * I0**2 / (2 * Pin), eigvals_only=True))
    print(f"unconstrained max gain (generalized eig) {gMax:.6g}")

    # objective max x^H Ug x -> A0 = -Ug
    A0 = -Ug
    s0 = np.zeros(N, dtype=complex)
    c0 = 0.0

    # constraints Re[x^H P^H (v - Z x)] = 0
    A1 = Z.conj().T
    s1 = v / 2

    # port x_p = 1 -> Re/Im[e_p^T x - 1] = 0
    a = np.zeros(N, dtype=complex)
    a[p] = 1
    B_j = [np.zeros((N, N), dtype=complex)] * 2
    s_2j = [a.conj() / 2, 1j * a.conj() / 2]
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

    def makeQCQP(Plist):
        return DenseSharedProjQCQP(A0, s0, c0, A1, s1, Plist, A2=np.eye(N, dtype=complex),
                                   B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0)

    def gain(x):
        return np.real(x.conj() @ Ug @ x)

    def report(name, QCQP):
        x = QCQP.current_xstar
        Gd = QCQP.current_dual
        res = x.conj() * (v - Z @ x)
        print(f"{name}: G_ub = {Gd:.6g}, G(xopt) = {gain(x):.6g}, Zin(xopt) = {Vp/(I0*x[p]):.4f}, "
              f"{QCQP.n_proj_constr} projectors, global residual {abs(np.sum(res)):.2e}")
        return Gd

    # dual feasible start: lambda Re[Z] - Ug > 0 for the global real power constraint,
    # Ug <= gMax I0^2/(2 Pin) Rz = gMax/Zin0 Rz
    lam0 = 2.0 * gMax * I0 / Vp

    # global bound (Re/Im power)
    In = np.ones(N, dtype=complex)
    QCQP = makeQCQP([sp.diags(In), sp.diags(1j * In)])
    init = np.zeros(4)
    init[0] = lam0
    la.cho_factor(QCQP._get_total_A(init))  # check dual feasibility of the start
    t1 = time.time()
    QCQP.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
    GlbGlobal = report(f"global ({time.time()-t1:.1f}s)", QCQP)
    lagsGlobal = QCQP.current_lags

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
    GlbGCD = report(f"GCD ({time.time()-t1:.1f}s)", gQCQP)

    # all-local bound (Re/Im per BF), lambda_i = lam0 on all real power constraints
    GlbLocal = np.nan
    lagsLocal = np.zeros(0)
    IoptLocal = np.zeros((N, 1), dtype=complex)
    if doLocal:
        Pl = [sp.csc_array(([q], ([i], [i])), shape=(N, N), dtype=complex)
              for i in range(N) for q in (1, 1j)]
        lQCQP = makeQCQP(Pl)
        init = np.zeros(2 * N + 2)
        init[0:2 * N:2] = lam0
        t1 = time.time()
        lQCQP.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
        GlbLocal = report(f"local ({time.time()-t1:.1f}s)", lQCQP)
        lagsLocal = lQCQP.current_lags
        IoptLocal = (I0 * lQCQP.current_xstar).reshape(-1, 1)

    # dual optimum of GCD. The Lagrange matrix is singular at the optimum (convex
    # objective), xstar is fixed only up to its null space. Shift xstar along the
    # null vectors n (eig < NULLTOL max eig, n_p = 0) to minimize the residuals of
    # the projector constraints; the Lagrangian, and so the fit J, is unchanged.
    xstar = gQCQP.current_xstar
    Lstar = Msym(gQCQP._get_total_A(gQCQP.current_lags))
    eL, WL = np.linalg.eigh(Lstar)
    minEig = eL[0]
    Nn = WL[:, eL < float(os.environ.get("NULLTOL", 1e-9)) * eL[-1]]
    Nn = Nn[:, np.abs(Nn[p, :]) < 1e-6]
    Pd = gQCQP.Proj.get_Pdata_column_stack()

    def projResidual(x):
        return np.real(Pd.conj().T @ (x.conj() * (v - Z @ x)))

    k = Nn.shape[1]
    if k > 0:
        fit = least_squares(lambda c: projResidual(xstar + Nn @ (c[:k] + 1j * c[k:])) * I0 / Vp,
                            np.zeros(2 * k), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        x = xstar + Nn @ (fit.x[:k] + 1j * fit.x[k:])
    else:
        x = xstar
    print(f"null space of L*: {k} vectors, max projector residual {np.max(abs(projResidual(xstar))):.3e} "
          f"-> {np.max(abs(projResidual(x))):.3e}, G {gain(xstar):.4g} -> {gain(x):.4g}")
    res = x.conj() * (v - Z @ x)
    print(f"min eig Lagrange matrix {minEig:.3e}, |x_p - 1| = {abs(x[p]-1):.2e}, "
          f"global Re/Im residual {np.sum(res.real):.2e} {np.sum(res.imag):.2e} "
          f"(of Zin0 = {Vp/I0:.1f}), max local |x^*(v-Zx)| {np.max(abs(res)):.2e}")

    out = dict(GlbGlobal=GlbGlobal, GlbGCD=GlbGCD, GlbLocal=GlbLocal, gMax=gMax,
               Iopt=(I0 * x).reshape(-1, 1), xopt=x.reshape(-1, 1), xstar=xstar.reshape(-1, 1),
               Lstar=Lstar,
               lagsGlobal=np.asarray(lagsGlobal).reshape(-1, 1),
               lagsGCD=np.asarray(gQCQP.current_lags).reshape(-1, 1),
               lagsLocal=np.asarray(lagsLocal).reshape(-1, 1), IoptLocal=IoptLocal,
               Pdiags=Pd, minEigGCD=minEig, nNull=k,
               GIopt=gain(x), ZinIopt=Vp / (I0 * x[p]), localResidual=res.reshape(-1, 1),
               Ngcd=Ngcd)
    sio.savemat(outFile, out)
    print(f"saved {outFile}")
    return out


if __name__ == "__main__":
    main(sys.argv)
