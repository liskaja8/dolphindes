"""Dual bound with GCD refinement for MATLAB/AToM operators (scattered-voltage DOF).

Maximizes the scattered power normalized by the power incident on the design
domain, P_sca/P_inc, with the scattered voltage V_sca = -Z0 I as the degree of
freedom (VscatBound.ipynb). Evaluates the global bound (real and reactive power
constraints), runs Ngcd GCD iterations and returns the dual optimal scattered
voltage together with the Lagrange matrix.

Usage
-----
    python dualGCDforMTB.py in.mat out.mat   # called from TSGAdualPareto.m
    python dualGCDforMTB.py                  # standalone test, *_dipole.txt

in.mat (AToM RWG basis, normalize=false):
    R0, X0, Lmat [N x N], V [N x 1] (per unit field), Zs, E0, Pinc, Ngcd

out.mat (AToM RWG basis, J = (Vsca - Vopt)^H LV (Vsca - Vopt), Vsca = -Z0 I):
    Vopt [N x 1]  dual optimal scattered voltage
    LV [N x N]    Lagrange matrix (quadratic part of the Lagrangian) for Vsca
    Iopt [N x 1]  current related to Vopt, Iopt = -Z0^{-1} Vopt
    dualGlobal, dualGCD, lagsGlobal, lagsGCD, Pdiags, minEigGCD
"""

import copy
import os
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters


def Msym(M):
    return 0.5 * (M.conj().T + M)


def load_inputs(argv):
    if len(argv) >= 2:
        data = sio.loadmat(argv[1], squeeze_me=True)
        R0 = np.asarray(data["R0"], dtype=float)
        X0 = np.asarray(data["X0"], dtype=float)
        Lmat = np.asarray(data["Lmat"], dtype=float)
        V = np.asarray(data["V"], dtype=complex).ravel()
        Zs = complex(data["Zs"])
        E0 = float(data["E0"])
        Pinc = float(data["Pinc"])
        Ngcd = int(data["Ngcd"])
        return R0, X0, Lmat, V, Zs, E0, Pinc, Ngcd

    # standalone test: shortened dipole, unnormalized (Pinc = 1, bound in W)
    here = os.path.dirname(os.path.abspath(__file__))
    load = lambda name: np.loadtxt(os.path.join(here, name), delimiter=",")
    epsilon_0 = 8.854e-12
    mu_0 = 4e-7 * np.pi
    omega = 2 * np.pi * 2.45e9
    sigma = 5.96e7
    delta = np.sqrt(2 / (omega * mu_0 * sigma))
    Zs = (1 + 1j) / (sigma * delta)
    return (load("R0_dipole.txt"), load("X0_dipole.txt"), load("Lmat_dipole.txt"),
            load("V_dipole.txt").astype(complex), Zs, 1e3, 1.0, 20)


def main(argv):
    R0, X0, Lmat, V, Zs, E0, Pinc, Ngcd = load_inputs(argv)
    Ndes = Lmat.shape[0]
    print(f"N = {Ndes}, Zs = {Zs:.4e} Ohm, E0 = {E0}, Pinc = {Pinc:.6e} W, Ngcd = {Ngcd}")

    # Cholesky whitening, Zmat = Zs*L -> Zs*1, I^ = Lchol I, V^ = Lchol^{-H} V
    Lchol = np.linalg.cholesky(Lmat).conj().T
    LcholInv = np.linalg.inv(Lchol)

    def Mchol(M):
        return Msym(LcholInv.conj().T @ M @ LcholInv)

    R0h = Mchol(R0)
    X0h = Mchol(X0)
    Vinc = LcholInv.conj().T @ (E0 * V)
    Z0h = R0h + 1j * X0h
    Ztot = Z0h + Zs * np.eye(Ndes)

    # full structure (reference)
    Ifull = np.linalg.solve(Ztot, Vinc)
    Pfull = 0.5 * np.real(Ifull.conj() @ R0h @ Ifull) / Pinc
    print(f"full structure: Psca/Pinc = {Pfull:.6e}")

    # objective in Vsca: Psca/Pinc = 1/2 Vsca^H W^H R0 W Vsca / Pinc, W = Z0^{-1}
    W = np.linalg.inv(Z0h)
    WH = W.conj().T
    A0 = -0.5 * Msym(WH @ R0h @ W) / Pinc
    s0 = np.zeros(Ndes, dtype=complex)
    c0 = 0.0

    # global constraints Re[I^H diag(p) (V + Vsca - Zs I)] = 0, p = {1, j}
    A1 = np.eye(Ndes) + np.conj(Zs) * WH
    A2 = W
    s1 = -0.5 * Vinc
    qList = [1, -1j]
    Plist = [sp.diags(np.conj(q) * np.ones(Ndes, dtype=complex)) for q in qList]

    opt_params = OptimizationHyperparameters(opttol=1e-6,
        gradConverge=True,
        min_inner_iter=10,
        max_restart=10,
        penalty_ratio=1e-2,
        penalty_reduction=0.1,
        break_iter_period=5,
        verbose=0)

    # global bound, any multiplier > 1/(2 Pinc) on the real power is dual feasible
    QCQP = DenseSharedProjQCQP(A0, s0, c0, A1, s1, Plist, A2=A2, verbose=0)
    t1 = time.time()
    QCQP.solve_current_dual_problem(method="newton", init_lags=np.array([1.0 / Pinc, 0.0]),
                                    opt_params=opt_params)
    dualGlobal = QCQP.current_dual
    lagsGlobal = QCQP.current_lags
    print(f"global bound: {dualGlobal:.6e} ({time.time()-t1:.2f}s), lags {lagsGlobal}")

    # GCD refinement, no merging and no early termination
    gQCQP = copy.deepcopy(QCQP)
    gcd_params = GCDHyperparameters(max_proj_cstrt_num=2 + 2 * Ngcd,
        max_gcd_iter_num=Ngcd,
        gcd_iter_period=Ngcd + 1,
        opt_params=opt_params)
    t1 = time.time()
    gQCQP.run_gcd(gcd_params)
    gQCQP.solve_current_dual_problem(method="newton", init_lags=gQCQP.current_lags,
                                     opt_params=opt_params)
    dualGCD = gQCQP.current_dual
    print(f"GCD bound: {dualGCD:.6e} ({time.time()-t1:.2f}s, {gQCQP.n_proj_constr} projectors)")

    # dual optimal scattered voltage and Lagrange matrix (whitened)
    Vh = gQCQP.current_xstar
    Lh = Msym(gQCQP._get_total_A(gQCQP.current_lags))
    minEig = np.min(np.linalg.eigvalsh(Lh))
    print(f"min eigenvalue of the Lagrange matrix: {minEig:.3e}")

    # back to RWG basis: Vsca = Lchol^H V^, J = (Vsca-Vopt)^H LV (Vsca-Vopt)
    Vopt = Lchol.conj().T @ Vh
    LV = Msym(LcholInv @ Lh @ LcholInv.conj().T)
    Z0 = R0 + 1j * X0
    Iopt = -np.linalg.solve(Z0, Vopt)

    # checks in the AToM convention, Z I = E0 V, Z = R0 + jX0 + Zs Lmat
    Z = Z0 + Zs * Lmat
    Vexc = E0 * V
    def res(I):
        return abs(I.conj() @ (Z @ I) - I.conj() @ Vexc) / abs(I.conj() @ Vexc)
    print(f"AToM power balance residual of Iopt: {res(Iopt):.3e} (conj: {res(Iopt.conj()):.3e})")
    PsOpt = 0.5 * np.real(Iopt.conj() @ R0 @ Iopt) / Pinc
    print(f"Psca(Iopt)/Pinc = {PsOpt:.6e}, dual GCD = {dualGCD:.6e}, full = {Pfull:.6e}")

    out = dict(Vopt=Vopt.reshape(-1, 1), LV=LV, Iopt=Iopt.reshape(-1, 1),
               dualGlobal=dualGlobal, dualGCD=dualGCD, Pfull=Pfull,
               lagsGlobal=np.asarray(lagsGlobal).reshape(-1, 1),
               lagsGCD=np.asarray(gQCQP.current_lags).reshape(-1, 1),
               Pdiags=gQCQP.Proj.get_Pdata_column_stack(), minEigGCD=minEig)
    if len(argv) >= 3:
        sio.savemat(argv[2], out)
        print(f"saved {argv[2]}")
    return out


if __name__ == "__main__":
    main(sys.argv)
