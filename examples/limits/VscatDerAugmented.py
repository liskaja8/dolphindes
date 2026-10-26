"""Maximum scattered power with local power and frequency-derivative balance, dual bound with GCD.

RWG basis with the non-diagonal material operator Z = Z0 + Zs L (main.tex, Sec.
"Inference with Non-Diagonal Material Operator"), no Cholesky whitening. The
variable is the scattered voltage augmented by its frequency derivative,

    x = [x1; x2] = [Vsca; omega Vsca'],    Vsca = -Z0 I,

so that I = -W x1, omega I' = W D W x1 - W x2, with W = Z0^{-1}, D = omega Z0'.
Every structure (subset of enabled RWG functions, material independent of
frequency) satisfies at every basis function i

    r1_i = (V - Z I)_i = 0,
    r2_i = (omega V' - (D + Zs/2 L) I - Z omega I')_i = 0     (omega Zs' = Zs/2)

or i_i = i'_i = 0, so all four tests conj(t_a,i) r_b,i, t = [I; omega I'], vanish.
With P a 2N x 2N matrix of per-basis-function 2 x 2 blocks, the constraint
Re[t^H P r] = 0 is the Dolphindes shared projector constraint

    Re(-x^H A1 P' A2 x + 2 x^H A2^H P'^H s1) = 0,    P' = P^H,
    A2 = [[W, 0], [-W D W, W]]  (A2 x = -t),
    A1 = 1 + A2^H Zt^H,  Zt = [[Zs L, 0], [Zs/2 L, Zs L]],  s1 = -[V; omega V']/2,

and the scattered power 1/2 I^H R0 I has A0 = -1/2 blockdiag(W^H R0 W, 0).
The baseline (power balance only) is the N x N block of the same form.

The derivative block is scaled by s (x2 = omega Vsca'/s, omega -> omega/s in
D, omega Zs', omega V'), since the derivative balance is indefinite in I with a
Schur complement ~ Q^2 times the power balance; s = sqrt(max|eig| / min eig)
makes lambda_pp = lambda_dd = 1 dual feasible. The 8 global constraints (block
patterns pp, dd, pd, dp times 1, j) are passed in a Hadamard basis so that each
projector fills Pstruct (Projectors.get_Pdata_column_stack requires it).
The global power pair is rotated, q = e^{j theta}, -j e^{j theta} (THETA=opt: the angle
maximizing min eig W^H Herm(e^{j theta} Z) W, VscatBoundRotation.ipynb), in both blocks.

Direct inference: the density d_i is the least-squares fit of the admittance
relations of the current and of its derivative,

    Lam_ii i_i = d_i u_i,                 u = V - (Z - Lam) I,
    Lam_ii (omega i'_i + i_i/2) = d_i omega u'_i,
    omega u' = omega V' - (D + Zs/2 (L - diag L)) I - (Z - Lam) omega I',

Lam = Zs diag(L_ii), in the current residual, followed by a threshold sweep.

Usage
-----
    python VscatDerAugmented.py in.mat out.mat      # in.mat from VscatDerAugmented.m
    NGCD=40 THETA=opt SEED=0 OPTTOL=1e-9 GCD_OPTTOL=1e-6 python VscatDerAugmented.py ...

in.mat:  R0, X0, Lmat, D [N x N], V, dV (= omega V') [N x 1], Zs, E0, Ngcd
out.mat: PlbBase, PlbBaseGCD (power balance only), PlbGlobal, PlbGCD (lifted),
         PlbLocal (nan, all 8N local constraints not evaluated), Pfull,
         xopt, Iopt, dIopt (= omega I'), PIopt, d, dPw, dDer, Pd, PdPw, PdDer,
         sweepN, sweepP, sweepPpw, PsweepBest, PsweepBestPw, Sbest, Ibest,
         Lstar, lags*, Pdata, minEigGCD, nNull, residual norms
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
from scipy.optimize import minimize_scalar

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters


def Msym(M):
    return 0.5 * (M.conj().T + M)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    inFile = argv[1] if len(argv) >= 2 else os.path.join(here, "results", "VscatDerAug_patch21ka1_gcd20_in.mat")
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    R0 = np.asarray(data["R0"], dtype=float)
    X0 = np.asarray(data["X0"], dtype=float)
    L = np.asarray(data["Lmat"], dtype=float)
    D = np.asarray(data["D"], dtype=complex)
    E0 = float(data["E0"])
    V = E0 * np.asarray(data["V"], dtype=complex).ravel()
    dV = E0 * np.asarray(data["dV"], dtype=complex).ravel()
    Zs = complex(data["Zs"])
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    opttol = float(os.environ.get("OPTTOL", 1e-9))
    gcdtol = float(os.environ.get("GCD_OPTTOL", 1e-6))
    N = R0.shape[0]
    print(f"N = {N}, Zs = {Zs:.4e} Ohm, E0 = {E0} V/m, Ngcd = {Ngcd}, opttol = {opttol}, GCD opttol = {gcdtol}")

    Z0 = R0 + 1j * X0
    Z = Z0 + Zs * L
    W = np.linalg.inv(Z0)
    WH = W.conj().T
    print(f"cond(Z0) = {np.linalg.cond(Z0):.3e}, ||D||/||Z0|| = {np.linalg.norm(D)/np.linalg.norm(Z0):.3f}")

    def Psca(I):
        return 0.5 * np.real(I.conj() @ R0 @ I)

    # ---------------------------------------------------------------- structures
    def solveStructure(S):
        """Current and omega I' of the structure with the RWG functions S."""
        I = np.zeros(N, dtype=complex)
        dI = np.zeros(N, dtype=complex)
        ZS = la.lu_factor(Z[np.ix_(S, S)])
        I[S] = la.lu_solve(ZS, V[S])
        dI[S] = la.lu_solve(ZS, (dV - (D + 0.5 * Zs * L) @ I)[S])
        return I, dI

    allBF = np.arange(N)
    Ifull, dIfull = solveStructure(allBF)
    Pfull = Psca(Ifull)
    print(f"full plate: P = {Pfull:.6e} W")

    # ---------------------------------------------------------------- derivative scaling
    # The derivative balance, with omega I' eliminated, is strongly indefinite in I
    # (its Schur complement scales as Q^2 times the power balance), so the dual
    # feasible lambda_dd ~ 1/Q^2 makes the Lagrangian numerically singular. The
    # derivative block is therefore scaled, x2 = omega Vsca'/s, t2 = omega I'/s,
    # i.e. omega -> omega/s in D, omega Zs' and omega V'. The constraint set and the
    # bound are unchanged, s = sqrt(max|eig Schur_dd| / min eig A_pp) balances them.
    O = np.zeros((N, N), dtype=complex)
    A1b = np.eye(N) + WH @ (np.conj(Zs) * L)
    A0b = -0.5 * Msym(WH @ R0 @ W)
    App = A0b + Msym(A1b @ W)  # lambda_pp = 1: W^H (R0/2 + Re[Zs] L) W
    sDer = os.environ.get("DERSCALE")
    if sDer is None:
        A2u = np.block([[W, O], [-W @ D @ W, W]])
        A1u = np.eye(2 * N) + A2u.conj().T @ np.block([[Zs * L, O], [0.5 * Zs * L, Zs * L]]).conj().T
        Add = Msym(A1u[:, N:] @ A2u[N:, :])  # P' = E_dd
        Sch = Add[:N, :N] - Add[:N, N:] @ np.linalg.solve(Add[N:, N:], Add[N:, :N])
        sDer = np.sqrt(np.max(np.abs(np.linalg.eigvalsh(Msym(Sch)))) / np.min(np.linalg.eigvalsh(App)))
    sDer = float(sDer)
    Ds = D / sDer
    dVs = dV / sDer
    zh = 0.5 * Zs / sDer
    print(f"derivative scaling s = {sDer:.4e}")

    # ---------------------------------------------------------------- rotated power pair
    # The global pair p = 1, -j is replaced by the rotated pair q = e^{j theta}, -j e^{j theta}
    # (VscatBoundRotation.ipynb), spanning the same constraints. The quadratic part of the
    # rotated real power, Herm(e^{j theta} Z) = cos(theta) Rt - sin(theta) Xt, stays positive
    # definite on a narrow interval of theta; THETA=opt (default) takes the angle maximizing
    # its smallest eigenvalue in the scattered-voltage space, W^H Herm(e^{j theta} Z) W,
    # THETA=<deg> a given angle. Both the power and the derivative blocks are rotated.
    Rt = R0 + Zs.real * L
    Xt = X0 + Zs.imag * L
    gXR = la.eigh(Xt, Rt, eigvals_only=True)
    thLo, thHi = np.arctan(1 / gXR.min()), np.arctan(1 / gXR.max())

    def rotReal(th):
        return np.cos(th) * Rt - np.sin(th) * Xt

    def lamMinV(th):
        return np.linalg.eigvalsh(Msym(WH @ rotReal(th) @ W))[0]

    thetaEnv = os.environ.get("THETA", "opt")
    if thetaEnv == "opt":
        width = thHi - thLo
        res = minimize_scalar(lambda th: -lamMinV(th) / lamMinV(0.0), method="bounded",
                              bounds=(thLo + 1e-6 * width, thHi - 1e-6 * width),
                              options=dict(xatol=1e-6 * width))
        theta = float(res.x)
    else:
        theta = np.radians(float(thetaEnv))
    qRot = [np.exp(1j * theta), -1j * np.exp(1j * theta)]
    lam0 = 1.05 * 0.5 * la.eigh(R0, rotReal(theta), eigvals_only=True)[-1]
    print(f"rotation: Herm(e^(j theta) Z) > 0 for theta in ({np.degrees(thLo):.4e}, {np.degrees(thHi):.4e}) deg, "
          f"theta = {np.degrees(theta):.5e} deg, lambda_min in Vsca {lamMinV(theta):.3e} (theta = 0: {lamMinV(0.0):.3e}), "
          f"start lambda_pp = {lam0:.4f}")

    def toX(I, dI):
        return np.concatenate([-Z0 @ I, -(Ds @ I + Z0 @ dI / sDer)])

    def fromX(x):
        """Current and (unscaled) omega I' of the lifted variable."""
        x1, x2 = x[:N], x[N:]
        I = -W @ x1
        dI = sDer * (W @ (Ds @ (W @ x1)) - W @ x2)
        return I, dI

    # ---------------------------------------------------------------- lifted QCQP
    A2 = np.block([[W, O], [-W @ Ds @ W, W]])
    Zt = np.block([[Zs * L, O], [zh * L, Zs * L]])
    A1 = np.eye(2 * N) + A2.conj().T @ Zt.conj().T
    b = np.concatenate([V, dVs])
    s1 = -b / 2
    A0 = np.zeros((2 * N, 2 * N), dtype=complex)
    A0[:N, :N] = -0.5 * Msym(WH @ R0 @ W)
    s0 = np.zeros(2 * N, dtype=complex)

    def localTests(x):
        """conj(t_a) * r_b for (a, b) = (1,1), (2,2), (1,2), (2,1), t = [I; omega I'/s]."""
        I, dI = fromX(x)
        dI = dI / sDer
        r1 = V + x[:N] - Zs * (L @ I)
        r2 = dVs + x[N:] - zh * (L @ I) - Zs * (L @ dI)
        return np.column_stack([I.conj() * r1, dI.conj() * r2, I.conj() * r2, dI.conj() * r1])

    # sparsity: per-RWG 2 x 2 blocks
    ii = np.arange(N)
    rows = np.concatenate([ii, ii + N, ii, ii + N])
    cols = np.concatenate([ii, ii + N, ii + N, ii])
    Pstruct = sp.csc_array((np.ones(4 * N, dtype=complex), (rows, cols)), shape=(2 * N, 2 * N))

    # global constraints: the block patterns pp, dd, pd, dp (each times I_N, times 1 and j)
    # in the Hadamard basis, so that every projector fills the whole Pstruct
    # (Projectors.get_Pdata_column_stack fails for projectors sparser than Pstruct)
    Had = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float)

    def hadP(h, q):
        vals = q * np.concatenate([h[0] * np.ones(N), h[1] * np.ones(N), h[2] * np.ones(N), h[3] * np.ones(N)])
        return sp.csc_array((vals.astype(complex), (rows, cols)), shape=(2 * N, 2 * N))

    PlistGlob = [hadP(h, np.conj(q)) for h in Had for q in qRot]

    def globalLags(c):
        """Multipliers of PlistGlob for the real block coefficients c = (pp, dd, pd, dp)."""
        lags = np.zeros(len(PlistGlob) + nGen)
        lags[0:len(PlistGlob):2] = Had @ np.asarray(c, dtype=float) / 4
        return lags

    def sharedForm(x, Pp):
        return np.real(-x.conj() @ A1 @ (Pp @ (A2 @ x)) + 2 * x.conj() @ (A2.conj().T @ (Pp.conj().T @ s1)))

    def directForm(x, Pp):
        I, dI = fromX(x)
        t = np.concatenate([I, dI / sDer])
        r = b + x - Zt @ t
        return np.real(t.conj() @ (Pp.conj().T @ r))

    # checks: shared form = direct form, residuals of physical structures vanish
    rng = np.random.default_rng(0)
    xr = rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)
    Pr = sp.csc_array((rng.standard_normal(4 * N) + 1j * rng.standard_normal(4 * N), (rows, cols)),
                      shape=(2 * N, 2 * N))
    errs = [abs(sharedForm(xr, Pp) - directForm(xr, Pp)) / max(abs(directForm(xr, Pp)), 1e-300)
            for Pp in PlistGlob + [Pr]]
    print(f"shared vs direct form, random x: max relative error {max(errs):.2e}")
    scale = np.max(np.abs(np.concatenate([Ifull, dIfull / sDer]))) * np.max(np.abs(b))
    xfull = toX(Ifull, dIfull)
    print(f"full plate: max |local test| / (max|t| max|b|) = {np.max(np.abs(localTests(xfull)))/scale:.2e}")
    Srand = np.sort(rng.choice(N, N // 2, replace=False))
    Ir, dIr = solveStructure(Srand)
    xrand = toX(Ir, dIr)
    print(f"random structure ({len(Srand)} BFs): max |local test| / scale = "
          f"{np.max(np.abs(localTests(xrand)))/scale:.2e}, "
          f"shared form (random P) {abs(sharedForm(xrand, Pr)):.2e} vs {abs(sharedForm(xr, Pr)):.2e} at random x")

    opt_params = OptimizationHyperparameters(opttol=opttol,
        gradConverge=True,
        min_inner_iter=10,
        max_restart=10,
        penalty_ratio=1e-2,
        penalty_reduction=0.1,
        break_iter_period=5,
        verbose=0)
    gcd_opt_params = dataclasses.replace(opt_params, opttol=gcdtol)

    # EQUIL=1: equilibrate the general constraints by their multipliers (off by default: it
    # makes the multipliers O(1), but GCD then stops earlier, 16.13 W instead of 15.89 W
    # after 40 iterations with DPZERO=1)
    equil = os.environ.get("EQUIL", "0") == "1"

    def equilibrate(Q, label=""):
        """Rescale the general constraints by their multipliers, B_j <- |l_j| B_j, l_j <- sign(l_j).

        The Lagrangian, dual value and x* are unchanged; the general multipliers become O(1),
        which keeps the Newton steps of the dual well scaled. The derivative constraints are
        normalized by their spectral norm, which their cross block Re[I^H X omega I'] (scaled
        by s) dominates, so their multipliers would otherwise reach ~1e6.
        """
        nP = Q.n_proj_constr
        if not equil or Q.n_gen_constr == 0 or Q.current_lags is None:
            return
        lg = Q.current_lags[nP:]
        f = np.where(np.abs(lg) > 1e-300, np.abs(lg), 1.0)
        for j in range(Q.n_gen_constr):
            Q.B_j[j] = Q.B_j[j] * f[j]
            Q.s_2j[j] = Q.s_2j[j] * f[j]
        Q.c_2j = Q.c_2j * f
        Q.current_lags = np.concatenate([Q.current_lags[:nP], lg / f])
        scale = getattr(Q, "genScale", np.ones(Q.n_gen_constr))
        Q.genScale = np.concatenate([scale, np.ones(Q.n_gen_constr - len(scale))]) * f
        Q.compute_precomputed_values()
        Q._assembly_map_built = False
        if label:
            print(f"  equilibrated general constraints ({label}): scales {np.array2string(f[:4], precision=3)}"
                  + (" ..." if len(f) > 4 else ""))

    def gcd(QCQP, nBase):
        gQCQP = copy.deepcopy(QCQP)
        np.random.seed(int(os.environ.get("SEED", 0)))  # random Pstruct coefficients of GCD
        gcd_params = GCDHyperparameters(max_proj_cstrt_num=nBase + 2 * Ngcd,
            max_gcd_iter_num=Ngcd,
            gcd_iter_period=Ngcd + 1,
            method="newton",
            opt_params=gcd_opt_params)
        t1 = time.time()
        gQCQP.run_gcd(gcd_params)
        equilibrate(gQCQP, "after GCD")
        lagsGCD, dualGCD = gQCQP.current_lags.copy(), gQCQP.current_dual
        try:
            gQCQP.solve_current_dual_problem(method="newton", init_lags=lagsGCD, opt_params=opt_params)
        except np.linalg.LinAlgError as err:
            # line search reached the PSD boundary where the feasibility test and the
            # Cholesky factorization disagree by round-off: keep the GCD solution
            print(f"final dual re-solve failed ({err}), keeping the GCD solution {dualGCD:.6e} W")
            gQCQP.current_lags = lagsGCD
            gQCQP.current_dual = gQCQP.get_dual(lagsGCD)[0]
            gQCQP.current_xstar = gQCQP._get_xstar(lagsGCD)[0]
        return gQCQP, time.time() - t1

    # ---------------------------------------------------------------- baseline: power balance only
    bQCQP = DenseSharedProjQCQP(A0[:N, :N], s0[:N], 0.0, A1b, -V / 2,
                                [sp.diags(np.conj(q) * np.ones(N, dtype=complex), format="csc") for q in qRot],
                                A2=W, verbose=0)
    t1 = time.time()
    bQCQP.solve_current_dual_problem(method="newton", init_lags=np.array([lam0, 0.0]), opt_params=opt_params)
    PlbBase = bQCQP.current_dual
    print(f"power only, global: P_ub = {PlbBase:.6e} W ({time.time()-t1:.1f}s)")
    bgQCQP, tg = gcd(bQCQP, 2)
    PlbBaseGCD = bgQCQP.current_dual
    print(f"power only, GCD: P_ub = {PlbBaseGCD:.6e} W ({tg:.1f}s, {bgQCQP.n_proj_constr} projectors)")

    # ---------------------------------------------------------------- lifted: power + derivative balance
    # optional design constraints (DPZERO=1): zero frequency derivative of the global
    # dissipated and reactive power,
    #   omega dPd/domega = 1/2 I^H Rd I + Re[I^H R omega I'] = 0,  R = R0 + Re[Zs] L, Rd = Re D + Re[Zs]/2 L,
    #   omega dQ/domega  = 1/2 I^H Xd I + Re[I^H X omega I'] = 0,  X = X0 + Im[Zs] L, Xd = Im D + Im[Zs]/2 L,
    # (1/2 I^H Z I with omega dZ/domega = D + Zs/2 L). They are not identities: they restrict
    # the bound to structures with stationary dissipated and reactive power at f0. With
    # t = [I; omega I'/s] = -A2 x they are x^H A2^H M A2 x, passed as general constraints
    # B_j = -M (normalized), s_2j = 0, c_2j = 0.
    dpZero = os.environ.get("DPZERO", "0") == "1"
    Rm, Xm = R0 + Zs.real * L, X0 + Zs.imag * L
    Rd, Xd = D.real + 0.5 * Zs.real * L, D.imag + 0.5 * Zs.imag * L

    def derForm(Md, M):
        return np.block([[0.5 * Md, 0.5 * sDer * M], [0.5 * sDer * M, O]]).astype(complex)

    def derPower(I, dI):
        """omega dPd/domega and omega dQ/domega evaluated directly."""
        return (0.5 * np.real(I.conj() @ Rd @ I) + np.real(I.conj() @ Rm @ dI),
                0.5 * np.real(I.conj() @ Xd @ I) + np.real(I.conj() @ Xm @ dI))

    Bgen, nGen, nrmGen = [], 0, []
    if dpZero:
        for M in (derForm(Rd, Rm), derForm(Xd, Xm)):
            nrmGen.append(np.linalg.norm(Msym(A2.conj().T @ M @ A2), 2))
            Bgen.append(-M / nrmGen[-1])
        nGen = len(Bgen)

    def genValues(x):
        """Values of the general constraints times their norms (= omega dPd/domega, omega dQ/domega)."""
        return np.array([np.real(-x.conj() @ Msym(A2.conj().T @ B @ A2) @ x) * n for B, n in zip(Bgen, nrmGen)])

    if dpZero:
        for name, (I_, dI_) in {"full plate": (Ifull, dIfull), "random structure": (Ir, dIr)}.items():
            dP_, dQ_ = derPower(I_, dI_)
            g_ = genValues(toX(I_, dI_))
            print(f"{name}: omega dPd/domega = {dP_:+.4e} W, omega dQ/domega = {dQ_:+.4e} W, Pd = "
                  f"{0.5*np.real(I_.conj() @ Rm @ I_):.4e} W, general form {g_[0]:+.4e}, {g_[1]:+.4e}")
    QCQP = DenseSharedProjQCQP(A0, s0, 0.0, A1, s1, PlistGlob, Pstruct=Pstruct, A2=A2,
                               B_j=Bgen if dpZero else None,
                               s_2j=[np.zeros(2 * N, dtype=complex)] * nGen if dpZero else None,
                               c_2j=np.zeros(nGen) if dpZero else None, verbose=0)
    # lambda_pp = 1 makes the x1 block positive definite; the derivative block enters
    # through W D W, so lambda_dd is reduced until the Schur complement stays positive
    lamDD = 1.0
    while not QCQP.is_dual_feasible(globalLags([lam0, lamDD, 0, 0])):
        lamDD /= 2
        if lamDD < 1e-16:
            raise RuntimeError("no dual feasible start found")
    init = globalLags([lam0, lamDD, 0, 0])
    print(f"lifted dual feasible start: lambda_pp = {lam0:.4f}, lambda_dd = {lamDD:.3e}")
    t1 = time.time()
    QCQP.solve_current_dual_problem(method="newton", init_lags=init, opt_params=opt_params)
    PlbGlobal = QCQP.current_dual
    if dpZero and equil:
        equilibrate(QCQP, "global")
        QCQP.solve_current_dual_problem(method="newton", init_lags=QCQP.current_lags, opt_params=opt_params)
        PlbGlobal = QCQP.current_dual
    lagsGlobal = QCQP.current_lags
    print(f"power + derivative, global: P_ub = {PlbGlobal:.6e} W ({time.time()-t1:.1f}s)")
    gQCQP, tg = gcd(QCQP, len(PlistGlob))
    PlbGCD = gQCQP.current_dual
    print(f"power + derivative, GCD: P_ub = {PlbGCD:.6e} W ({tg:.1f}s, {gQCQP.n_proj_constr} projectors)")

    # ---------------------------------------------------------------- local power-derivative GCD
    # Optional design constraints (NLOCGCD=n > 0): zero frequency derivative of the local
    # complex power of every basis function,
    #   q_i = omega d/domega [conj(i_i) (Z I)_i]
    #       = conj(i_i) ((D + Zs/2 L) I + Z omega I')_i + conj(omega i'_i) (Z I)_i = 0,
    # i.e. q = t^* (K t) with t = [I; omega I'/s], K = [[D + Zs/2 L, s Z], [s Z, 0]] (rows
    # of the two halves summed per basis function). Re/Im of sum_i q_i are the global
    # dissipated/reactive power derivatives (DPZERO). The constraints Re[sum_i p_i q_i] = 0
    # do not share A1 with the power-balance projectors, so they enter as general
    # constraints, B = -diag([p; p]) K. Each iteration adds the max-violation direction
    # p = conj(q(x*)) / |q(x*)| and re-solves the dual.
    nLoc = int(os.environ.get("NLOCGCD", 0))
    Kloc = np.block([[D + 0.5 * Zs * L, sDer * Z], [sDer * Z, O]])

    def localDer(x):
        I_, dI_ = fromX(x)
        return I_.conj() * ((D + 0.5 * Zs * L) @ I_ + Z @ dI_) + dI_.conj() * (Z @ I_)

    def addGeneral(Q, B):
        Q.B_j.append(np.asarray(B, dtype=complex))
        Q.s_2j.append(np.zeros(2 * N, dtype=complex))
        Q.c_2j = np.append(Q.c_2j, 0.0)
        Q.n_gen_constr = len(Q.B_j)
        Q.current_lags = np.append(Q.current_lags, 0.0)
        Q.compute_precomputed_values()
        Q._assembly_map_built = False

    locHist = []
    if nLoc > 0:
        qI, dqI = fromX(xrand)
        qr = localDer(xrand)
        print(f"local derivative check (random structure): Re/Im sum q / 2 = {np.sum(qr).real/2:+.4e}, "
              f"{np.sum(qr).imag/2:+.4e}, global {derPower(qI, dqI)[0]:+.4e}, {derPower(qI, dqI)[1]:+.4e} W")
        t1 = time.time()
        for it in range(nLoc):
            q = localDer(gQCQP.current_xstar)
            locHist.append([gQCQP.current_dual, np.max(np.abs(q)) / (2 * PlbGCD), np.linalg.norm(q) / (2 * PlbGCD)])
            pq = np.conj(q) / np.linalg.norm(q)
            B = -(np.concatenate([pq, pq])[:, None] * Kloc)
            B /= np.linalg.norm(Msym(A2.conj().T @ B @ A2), 2)
            # start at the largest scale reached by the general constraints so far
            sc = float(np.max(getattr(gQCQP, "genScale", [1.0])))
            addGeneral(gQCQP, sc * B)
            gQCQP.genScale = np.append(getattr(gQCQP, "genScale", np.ones(gQCQP.n_gen_constr - 1)), sc)
            lagsPrev, dualPrev = gQCQP.current_lags.copy(), gQCQP.current_dual
            try:
                gQCQP.solve_current_dual_problem(method="newton", init_lags=lagsPrev, opt_params=gcd_opt_params)
            except np.linalg.LinAlgError:
                gQCQP.current_lags = lagsPrev
                gQCQP.current_dual = gQCQP.get_dual(lagsPrev)[0]
                gQCQP.current_xstar = gQCQP._get_xstar(lagsPrev)[0]
                print(f"  local GCD {it + 1}: dual solve hit the PSD boundary, kept the previous multipliers")
            equilibrate(gQCQP)
            print(f"  local GCD {it + 1}: P_ub = {gQCQP.current_dual:.6e} W, max|q|/2P_ub before {locHist[-1][1]:.3e}")
        q = localDer(gQCQP.current_xstar)
        locHist.append([gQCQP.current_dual, np.max(np.abs(q)) / (2 * PlbGCD), np.linalg.norm(q) / (2 * PlbGCD)])
        PlbGCD = gQCQP.current_dual
        print(f"local derivative GCD: P_ub = {PlbGCD:.6e} W ({time.time()-t1:.1f}s, {nLoc} general constraints), "
              f"max|q|/2P_ub {locHist[0][1]:.3e} -> {locHist[-1][1]:.3e}")

    # ---------------------------------------------------------------- dual optimum
    # dual optimum used as is (no shift along the near-null space of Lstar)
    x = xstar = gQCQP.current_xstar
    Lstar = Msym(gQCQP._get_total_A(gQCQP.current_lags))
    eL = np.linalg.eigvalsh(Lstar)
    minEig = eL[0]
    Pd = gQCQP.Proj.get_Pdata_column_stack()
    k = int(np.sum(eL < 1e-7 * eL[-1]))
    I, dI = fromX(x)
    loc = localTests(x)
    print(f"L*: min/max eig {minEig/eL[-1]:.3e}, {k} eigenvalues below 1e-7 max eig")
    gs = loc.sum(axis=0)
    print(f"P(I*) = {Psca(I):.6e} W")
    dPstar, dQstar = derPower(I, dI)
    print(f"dual current: omega dPd/domega = {dPstar:+.4e} W, omega dQ/domega = {dQstar:+.4e} W, "
          f"Pd = {0.5*np.real(I.conj() @ Rm @ I):.4e} W"
          + (f", general multipliers {np.array2string(gQCQP.current_lags[gQCQP.n_proj_constr:][:4], precision=3)}"
             f" (max |l| {np.max(np.abs(gQCQP.current_lags[gQCQP.n_proj_constr:])):.3g})" if dpZero else ""))
    print(f"global residuals / 2P_ub (pp, dd, pd, dp): Re {gs.real/(2*PlbGCD)}, Im {gs.imag/(2*PlbGCD)}")
    print(f"max local |test| / 2P_ub (pp, dd, pd, dp): {np.max(np.abs(loc), axis=0)/(2*PlbGCD)}")

    # ---------------------------------------------------------------- direct inference
    Lam = Zs * np.diag(L)
    Zp = Z - np.diag(Lam)
    u = V - Zp @ I
    du = dV - (D + 0.5 * Zs * (L - np.diag(np.diag(L)))) @ I - Zp @ dI
    bP = u / Lam
    bD = du / Lam
    y = dI + I / 2
    clip = lambda v: np.clip(v, 0, 1)
    d = clip(np.real(bP.conj() * I + bD.conj() * y) / (np.abs(bP)**2 + np.abs(bD)**2))
    dPw = clip(np.real(bP.conj() * I) / np.abs(bP)**2)
    dDer = clip(np.real(bD.conj() * y) / np.abs(bD)**2)
    resP = np.linalg.norm(I - d * bP) / np.linalg.norm(I)
    resD = np.linalg.norm(y - d * bD) / np.linalg.norm(y)
    print(f"LSQ fit residual: current {resP:.3e}, derivative {resD:.3e}; "
          f"max |d_pw - d_der| = {np.max(np.abs(dPw - dDer)):.3f}, mean d = {d.mean():.3f}")

    def realizedP(dd):
        Ir = np.linalg.solve(np.diag(Lam) + dd[:, None] * Zp, dd * V)
        return Psca(Ir)

    Pd_, PdPw, PdDer = realizedP(d), realizedP(dPw), realizedP(dDer)
    print(f"realized relaxed density: LSQ {Pd_:.6e} W, current only {PdPw:.6e} W, derivative only {PdDer:.6e} W")

    def sweep(dd):
        order = np.argsort(-dd, kind="stable")
        P = np.zeros(N)
        for n in range(1, N + 1):
            Ir, _ = solveStructure(np.sort(order[:n]))
            P[n - 1] = Psca(Ir)
        nBest = int(np.argmax(P)) + 1
        S = np.sort(order[:nBest])
        return P, nBest, S

    sweepP, nBest, Sbest = sweep(d)
    sweepPpw, nBestPw, _ = sweep(dPw)
    Ibest, _ = solveStructure(Sbest)
    print(f"threshold sweep: LSQ best {sweepP[nBest-1]:.6e} W ({nBest} BFs), "
          f"current only best {sweepPpw[nBestPw-1]:.6e} W ({nBestPw} BFs)")
    print(f"summary: full plate {Pfull:.6e}, bounds power only {PlbBase:.6e} / {PlbBaseGCD:.6e}, "
          f"power + derivative {PlbGlobal:.6e} / {PlbGCD:.6e} W")

    Smask = np.zeros(N)
    Smask[Sbest] = 1
    col = lambda v: np.asarray(v).reshape(-1, 1)
    out = dict(PlbBase=PlbBase, PlbBaseGCD=PlbBaseGCD, PlbGlobal=PlbGlobal, PlbGCD=PlbGCD,
               PlbLocal=np.nan, Pfull=Pfull,
               xopt=col(x), xstar=col(xstar), Iopt=col(I), dIopt=col(dI), PIopt=Psca(I),
               d=col(d), dPw=col(dPw), dDer=col(dDer), Pd=Pd_, PdPw=PdPw, PdDer=PdDer,
               resP=resP, resD=resD,
               sweepN=col(np.arange(1, N + 1)), sweepP=col(sweepP), sweepPpw=col(sweepPpw),
               PsweepBest=sweepP[nBest - 1], PsweepBestPw=sweepPpw[nBestPw - 1],
               Sbest=col(Smask), Ibest=col(Ibest),
               Lstar=Lstar, lagsGlobal=col(lagsGlobal), lagsGCD=col(gQCQP.current_lags),
               lagsBaseGCD=col(bgQCQP.current_lags), sDer=sDer, theta=theta, dpZero=int(dpZero), nLocGCD=nLoc, locHist=np.array(locHist).reshape(-1, 3), dPdStar=dPstar, dQStar=dQstar, Pdata=Pd, minEigGCD=minEig, nNull=k,
               localTests=loc, Ngcd=Ngcd)
    sio.savemat(outFile, out)
    print(f"saved {outFile}")
    return out


if __name__ == "__main__":
    main(sys.argv)
