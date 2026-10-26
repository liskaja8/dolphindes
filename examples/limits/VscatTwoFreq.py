"""Maximum mean scattered power at two nearby frequencies, dual bound with block GCD.

RWG basis with Z = Z0 + Zs L (main.tex, Sec. "Frequency-Derivative Power Balance"),
two-frequency variant. The variable is the pair of scattered voltages at
omega+- = omega0 (1 +- h),

    x = [x+; x-] = [Vsca+; Vsca-],    I+- = -W+- x+-,   W+- = Z0+-^{-1},

so that omega I' = (I+ - I-)/(2h) is defined by construction. Every structure
(subset of enabled RWG functions) satisfies at every basis function i

    r+_i = (V+ - Z+ I+)_i = 0,   r-_i = (V- - Z- I-)_i = 0,
    rd_i = (Vw - Zd+ I+ + Zd- I-)_i = 0,   Zd+- = Z/(2h) +- Zw/2,

or i+_i = i-_i = 0; rd is the derivative of the state equation at omega0 with
center-frequency operators Z, Zw = omega dZ/domega = D + Zs/2 L, Vw = omega V'
(exact to O(h^2)). The tests are the power balance at each frequency (no cross
terms) and the derivative balance tested by the sum and difference currents,

    conj(t_k) r_k,   t = [I+; I-; I+ + I-; I+ - I-],   r = [r+; r-; rd; rd],

i.e. diagonal projectors in a 4N test space. In the Dolphindes shared-projector
form Re[t^H P^H r] = Re(-x^H A1 P A2 x + 2 x^H A2^H P^H s1) with the rectangular

    A2 = [[W+, 0], [0, W-], [W+, W-], [W+, -W-]]           (t = -A2 x),
    A1^H = [[1 + Zs+ L W+, 0], [0, 1 + Zs- L W-],
            [Zd+ W+, -Zd- W-], [Zd+ W+, -Zd- W-]],         (r = A1^H x - 2 s1),
    s1 = -[V+; V-; Vw; Vw]/2,

and the mean scattered power (P+ + P-)/2 has A0 = -1/4 blockdiag(W+^H R0+ W+, W-^H R0- W-).
DSCALE scales the sum/difference test rows (default auto = 2h, which brings the
derivative forms, ~ Z/(2h), to the norm of the power forms; same constraints).

Global constraints: the four blocks times the rotated pair q = e^{j theta}, -j e^{j theta}
(VscatBoundRotation.ipynb, theta at omega0). Block GCD: every iteration splits the
max-violation and min-eigenvalue projectors of dolphindes' run_gcd into their parts
on the + and - power blocks (GCD_BLOCKS=pm) and also on the sum and difference
derivative blocks (GCD_BLOCKS=all, default).

The dual optimum is shifted along the near-null space of L* (eig < NULLTOL max eig)
to minimize the projector residuals (maxGainGCD.py), and the density is inferred by
the least-squares fit of the admittance relations at both frequencies,
Lam+- i+-_i = d_i u+-_i, u+- = V+- - (Z+- - Lam+-) I+-, followed by a threshold sweep.

Usage
-----
    python VscatTwoFreq.py in.mat out.mat      # in.mat from VscatTwoFreq.m
    NGCD=20 NGCD_BASE=20 GCD_BLOCKS=all MAXPROJ=... DSCALE=auto BREAK_PERIOD=50 NULLTOL=1e-9 THETA=opt \
        OPTTOL=1e-9 GCD_OPTTOL=1e-6 python VscatTwoFreq.py ...

in.mat:  R0, X0, Lmat, D, V, dV, Zs (omega0), R0p, X0p, Vp, Zsp (omega+), R0m, X0m,
         Vm, Zsm (omega-), h, E0, Ngcd
out.mat: Pfull, PlbBasePlus, PlbBaseMinus, PlbGlobal, PlbGCD, xstar, xopt, Iplus, Iminus,
         dIfd, PIopt, dIratio, d, Pd, sweepN, sweepP, PsweepBest, Sbest, IbestPlus,
         IbestMinus, dIratioBest, Lstar, lags*, Pdata, minEigGCD, nNull, residuals
"""

import copy
import dataclasses
import os
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la
from scipy.optimize import least_squares, minimize_scalar

from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters


def Msym(M):
    return 0.5 * (M.conj().T + M)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    inFile = argv[1] if len(argv) >= 2 else os.path.join(here, "results", "VscatTwoFreq_patch21ka1_h1e-04_in.mat")
    outFile = argv[2] if len(argv) >= 3 else inFile.replace("_in.mat", "_out.mat")
    data = sio.loadmat(inFile, squeeze_me=True)
    E0 = float(data["E0"])
    h = float(data["h"])
    L = np.asarray(data["Lmat"], dtype=float)
    Z0c = np.asarray(data["R0"], dtype=float) + 1j * np.asarray(data["X0"], dtype=float)
    D = np.asarray(data["D"], dtype=complex)
    Zs = complex(data["Zs"])
    Vw = E0 * np.asarray(data["dV"], dtype=complex).ravel()
    R0 = {s: np.asarray(data["R0" + s], dtype=float) for s in "pm"}
    Z0 = {s: R0[s] + 1j * np.asarray(data["X0" + s], dtype=float) for s in "pm"}
    V = {s: E0 * np.asarray(data["V" + s], dtype=complex).ravel() for s in "pm"}
    Zsf = {s: complex(data["Zs" + s]) for s in "pm"}
    Ngcd = int(os.environ.get("NGCD", data["Ngcd"]))
    NgcdBase = int(os.environ.get("NGCD_BASE", 20))
    gcdBlocks = os.environ.get("GCD_BLOCKS", "all")
    # the derivative forms carry Z/(2h) and are ~1/(2h) larger than the power forms;
    # DSCALE = 2h (auto) makes them comparable, the constraint set is unchanged
    dScaleEnv = os.environ.get("DSCALE", "auto")
    dScale = 2 * h if dScaleEnv == "auto" else float(dScaleEnv)
    nullTol = float(os.environ.get("NULLTOL", 1e-9))
    opttol = float(os.environ.get("OPTTOL", 1e-9))
    gcdtol = float(os.environ.get("GCD_OPTTOL", 1e-6))
    N = L.shape[0]
    print(f"N = {N}, h = {h:.1e}, Zs = {Zs:.4e} Ohm, E0 = {E0} V/m, Ngcd = {Ngcd} (blocks {gcdBlocks}), "
          f"Ngcd base = {NgcdBase}, DSCALE = {dScale:.3e}, opttol = {opttol}, GCD opttol = {gcdtol}")

    Z = {s: Z0[s] + Zsf[s] * L for s in "pm"}
    W = {s: np.linalg.inv(Z0[s]) for s in "pm"}
    Zc = Z0c + Zs * L
    Zw = D + 0.5 * Zs * L
    Zd = {"p": Zc / (2 * h) + Zw / 2, "m": Zc / (2 * h) - Zw / 2}
    print(f"||Z+ - Z-|| / ||Z|| = {np.linalg.norm(Z['p'] - Z['m'])/np.linalg.norm(Zc):.3e}, "
          f"||(Z+ - Z-)/(2h) - Zw|| / ||Zw|| = {np.linalg.norm((Z['p'] - Z['m'])/(2*h) - Zw)/np.linalg.norm(Zw):.3e}")

    def Psca(I, s):
        return 0.5 * np.real(I.conj() @ R0[s] @ I)

    def Pmean(Ip, Im):
        return 0.5 * (Psca(Ip, "p") + Psca(Im, "m"))

    # ---------------------------------------------------------------- structures
    def solveStructure(S):
        """Currents at omega+ and omega- of the structure with the RWG functions S."""
        I = {}
        for s in "pm":
            I[s] = np.zeros(N, dtype=complex)
            I[s][S] = np.linalg.solve(Z[s][np.ix_(S, S)], V[s][S])
        return I["p"], I["m"]

    allBF = np.arange(N)
    Ifp, Ifm = solveStructure(allBF)
    Pfull = Pmean(Ifp, Ifm)
    print(f"full plate: P+ = {Psca(Ifp, 'p'):.6e}, P- = {Psca(Ifm, 'm'):.6e}, mean {Pfull:.6e} W")

    # ---------------------------------------------------------------- rotated power pair (omega0)
    Rt = np.real(Z0c) + Zs.real * L
    Xt = np.imag(Z0c) + Zs.imag * L
    Wc = np.linalg.inv(Z0c)
    gXR = la.eigh(Xt, Rt, eigvals_only=True)
    thLo, thHi = np.arctan(1 / gXR.min()), np.arctan(1 / gXR.max())

    def rotReal(th):
        return np.cos(th) * Rt - np.sin(th) * Xt

    def lamMinV(th):
        return np.linalg.eigvalsh(Msym(Wc.conj().T @ rotReal(th) @ Wc))[0]

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
    lam0 = 1.05 * 0.5 * max(la.eigh(R0[s], np.cos(theta) * np.real(Z[s]) - np.sin(theta) * np.imag(Z[s]),
                                    eigvals_only=True)[-1] for s in "pm")
    print(f"rotation: theta in ({np.degrees(thLo):.4e}, {np.degrees(thHi):.4e}) deg, theta = {np.degrees(theta):.5e} deg, "
          f"start lambda (single frequency) = {lam0:.4f}")

    # ---------------------------------------------------------------- two-frequency QCQP
    O = np.zeros((N, N), dtype=complex)
    Iu = np.eye(N)
    A2 = np.block([[W["p"], O], [O, W["m"]],
                   [dScale * W["p"], dScale * W["m"]], [dScale * W["p"], -dScale * W["m"]]])
    rowD = np.hstack([Zd["p"] @ W["p"], -Zd["m"] @ W["m"]])
    A1H = np.vstack([np.hstack([Iu + Zsf["p"] * L @ W["p"], O]),
                     np.hstack([O, Iu + Zsf["m"] * L @ W["m"]]),
                     rowD, rowD])
    A1 = A1H.conj().T
    bvec = np.concatenate([V["p"], V["m"], Vw, Vw])
    s1 = -bvec / 2
    A0 = np.zeros((2 * N, 2 * N), dtype=complex)
    A0[:N, :N] = -0.25 * Msym(W["p"].conj().T @ R0["p"] @ W["p"])
    A0[N:, N:] = -0.25 * Msym(W["m"].conj().T @ R0["m"] @ W["m"])
    s0 = np.zeros(2 * N, dtype=complex)
    m = 4 * N
    blockNames = ["+", "-", "sum", "diff"]

    def toX(Ip, Im):
        return np.concatenate([-Z0["p"] @ Ip, -Z0["m"] @ Im])

    def fromX(x):
        return -W["p"] @ x[:N], -W["m"] @ x[N:]

    def tests(x):
        """conj(t) * r per test row (4N), t = [I+; I-; s(I+ + I-); s(I+ - I-)]."""
        Ip, Im = fromX(x)
        t = np.concatenate([Ip, Im, dScale * (Ip + Im), dScale * (Ip - Im)])
        rp = V["p"] - Z["p"] @ Ip
        rm = V["m"] - Z["m"] @ Im
        rd = Vw - Zd["p"] @ Ip + Zd["m"] @ Im
        return t.conj() * np.concatenate([rp, rm, rd, rd])

    def sharedForm(x, p):
        return np.real(-x.conj() @ A1 @ (p * (A2 @ x)) + 2 * x.conj() @ (A2.conj().T @ (p.conj() * s1)))

    def blockDiag(b, q):
        p = np.zeros(m, dtype=complex)
        p[b * N:(b + 1) * N] = q
        return p

    rng = np.random.default_rng(0)
    xr = rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)
    pr = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    direct = np.real(np.conj(pr) @ tests(xr))
    print(f"shared vs direct form, random x and P: relative error {abs(sharedForm(xr, pr) - direct)/abs(direct):.2e}")
    print(f"objective: -x^H A0 x vs mean P at random x: {abs(-np.real(xr.conj() @ A0 @ xr) - Pmean(*fromX(xr)))/Pmean(*fromX(xr)):.2e}")

    def structureReport(name, Ip, Im):
        tt = tests(toX(Ip, Im)).reshape(4, N)
        scale = np.max(np.abs(np.concatenate([Ip, Im]))) * np.max(np.abs(bvec[:2 * N]))
        dI = (Ip - Im) / (2 * h)
        # lifted single-frequency derivative of the same structure, omega I' = Z^-1 (Vw - Zw I) on S
        S = np.flatnonzero(np.abs(Ip) > 0)
        Ic = np.linalg.solve(Zc[np.ix_(S, S)], (0.5 * (V["p"] + V["m"]))[S])
        dIl = np.zeros(N, dtype=complex)
        dIl[S] = np.linalg.solve(Zc[np.ix_(S, S)], (Vw - Zw[:, S] @ Ic)[S])
        print(f"{name}: max |test| / (max|I| max|V|) per block {np.max(np.abs(tt), axis=1)/scale}, "
              f"||(I+ - I-)/(2h)|| / ||I|| = {np.linalg.norm(dI)/np.linalg.norm(Ip):.3e}, "
              f"vs lifted omega I' {np.linalg.norm(dI - dIl)/np.linalg.norm(dIl):.2e}")

    structureReport("full plate", Ifp, Ifm)
    Srand = np.sort(rng.choice(N, N // 2, replace=False))
    structureReport(f"random structure ({len(Srand)} BFs)", *solveStructure(Srand))

    opt_params = OptimizationHyperparameters(opttol=opttol,
        gradConverge=True,
        min_inner_iter=10,
        max_restart=10,
        penalty_ratio=1e-2,
        penalty_reduction=0.1,
        break_iter_period=int(os.environ.get("BREAK_PERIOD", 50)),
        verbose=0)
    gcd_opt_params = dataclasses.replace(opt_params, opttol=gcdtol)

    def safeSolve(Q, lags, params):
        """Dual solve from lags; on a failed factorization at the PSD boundary keep lags."""
        try:
            Q.solve_current_dual_problem(method="newton", init_lags=lags, opt_params=params)
        except np.linalg.LinAlgError as err:
            print(f"  dual solve failed ({err}), keeping the previous multipliers")
            Q.current_lags = lags
            Q.current_dual = Q.get_dual(lags)[0]
            Q.current_xstar = Q._get_xstar(lags)[0]

    # ---------------------------------------------------------------- baseline: power balance at one frequency
    PlbBase, bQ = {}, {}
    for s in "pm":
        Q = DenseSharedProjQCQP(-0.5 * Msym(W[s].conj().T @ R0[s] @ W[s]), np.zeros(N, dtype=complex), 0.0,
                                Iu + W[s].conj().T @ (np.conj(Zsf[s]) * L), -V[s] / 2,
                                [np.conj(q) * np.eye(N, dtype=complex) for q in qRot], A2=W[s], verbose=0)
        safeSolve(Q, np.array([lam0, 0.0]), opt_params)
        P1 = Q.current_dual
        np.random.seed(int(os.environ.get("SEED", 0)))
        Q.run_gcd(GCDHyperparameters(max_proj_cstrt_num=2 + 2 * NgcdBase, max_gcd_iter_num=NgcdBase,
                                     gcd_iter_period=NgcdBase + 1, method="newton", opt_params=gcd_opt_params))
        safeSolve(Q, Q.current_lags.copy(), opt_params)
        PlbBase[s], bQ[s] = Q.current_dual, Q
        print(f"power only at omega{'+' if s == 'p' else '-'}: global {P1:.6e} W, GCD ({NgcdBase}) {Q.current_dual:.6e} W")

    # ---------------------------------------------------------------- two-frequency: global
    Pglob = [blockDiag(b, np.conj(q)) for b in range(4) for q in qRot]
    QCQP = DenseSharedProjQCQP(A0, s0, 0.0, A1, s1, [np.diag(p) for p in Pglob],
                               Pstruct=np.eye(m, dtype=complex), A2=A2, verbose=0)
    init = np.zeros(len(Pglob))
    init[0] = init[2] = lam0 / 2  # rotated real power at omega+ and omega-, mean objective
    while not QCQP.is_dual_feasible(init):
        init[0] *= 1.5
        init[2] *= 1.5
    print(f"two-frequency dual feasible start: lambda+ = lambda- = {init[0]:.4f}")
    t1 = time.time()
    safeSolve(QCQP, init, opt_params)
    PlbGlobal = QCQP.current_dual
    lagsGlobal = QCQP.current_lags.copy()
    print(f"two frequencies, global: P_ub = {PlbGlobal:.6e} W ({time.time()-t1:.1f}s), "
          f"multipliers {np.array2string(lagsGlobal, precision=4)}")

    # ---------------------------------------------------------------- two-frequency: block GCD
    blocks = [0, 1] if gcdBlocks == "pm" else [0, 1, 2, 3]
    maxProj = int(os.environ.get("MAXPROJ", len(Pglob) + 2 * len(blocks) * Ngcd))

    def orthonormalize(Q):
        """Euclidean orthonormalization of the projector data (as in run_gcd), multipliers transformed."""
        Pd = Q.Proj.get_Pdata_column_stack()
        Qr, Rr = la.qr(np.vstack([Pd.real, Pd.imag]), mode="economic")
        Q.Proj.set_Pdata_column_stack(Qr[:m] + 1j * Qr[m:])
        Q.current_lags = Rr @ Q.current_lags
        Q.compute_precomputed_values()
        Q._invalidate_factor_cache()

    gQ = copy.deepcopy(QCQP)
    t1 = time.time()
    gcdHist = []
    if Ngcd > 0:
        orthonormalize(gQ)
        for it in range(Ngcd):
            x = gQ.current_xstar
            viol = (2 * gQ.s1 - A1H @ x) * (A2 @ x).conj()
            v, w = gQ._get_PSD_penalty(gQ.current_lags)
            eg = (A1H @ v) * (A2 @ v).conj()
            eg = np.divide(eg, np.abs(eg), out=np.zeros_like(eg), where=np.abs(eg) > 0)
            new = []
            for vec in (viol, eg):
                for b in blocks:
                    pb = np.zeros(m, dtype=complex)
                    pb[b * N:(b + 1) * N] = vec[b * N:(b + 1) * N]
                    if np.linalg.norm(pb) > 1e-14 * max(np.linalg.norm(vec), 1e-300):
                        new.append(pb)
            gQ.add_constraints(new, orthonormalize=True, metric="euclidean")
            if len(gQ.Proj) > maxProj:
                gQ.merge_lead_constraints(merged_num=len(gQ.Proj) - maxProj + 1, metric="euclidean")
            safeSolve(gQ, gQ.current_lags.copy(), gcd_opt_params)
            gcdHist.append([gQ.current_dual, w])
            print(f"  block GCD {it + 1}: P_ub = {gQ.current_dual:.8e} W, {len(gQ.Proj)} projectors, "
                  f"min eig before {w:.3e}, max |violation| per block "
                  f"{np.array2string(np.max(np.abs(viol.reshape(4, N)), axis=1), precision=3)}")
        safeSolve(gQ, gQ.current_lags.copy(), opt_params)
    PlbGCD = gQ.current_dual
    print(f"two frequencies, block GCD: P_ub = {PlbGCD:.6e} W ({time.time()-t1:.1f}s, {len(gQ.Proj)} projectors)")

    # ---------------------------------------------------------------- dual optimum, null-space shift
    xstar = gQ.current_xstar
    Lstar = Msym(gQ._get_total_A(gQ.current_lags))
    eL, UL = np.linalg.eigh(Lstar)
    minEig = eL[0]
    Nn = UL[:, eL < nullTol * eL[-1]]
    k = Nn.shape[1]
    Pdata = gQ.Proj.get_Pdata_column_stack()

    def projResidual(x):
        return np.real(Pdata.conj().T @ tests(x)) / (2 * PlbGCD)

    if k > 0:
        fit = least_squares(lambda c: projResidual(xstar + Nn @ (c[:k] + 1j * c[k:])), np.zeros(2 * k),
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        x = xstar + Nn @ (fit.x[:k] + 1j * fit.x[k:])
    else:
        x = xstar
    Ip, Im = fromX(x)
    Is, _ = fromX(xstar)
    dIfd = (Ip - Im) / (2 * h)
    Imean = 0.5 * (Ip + Im)
    dIratio = np.linalg.norm(dIfd) / np.linalg.norm(Imean)
    PIopt = Pmean(Ip, Im)
    loc = tests(x).reshape(4, N)
    print(f"L*: min/max eig {minEig/eL[-1]:.3e}, {k} null vectors (tol {nullTol:.0e}); projector residual / 2P_ub "
          f"{np.max(np.abs(projResidual(xstar))):.3e} -> {np.max(np.abs(projResidual(x))):.3e}, "
          f"mean P {Pmean(*fromX(xstar)):.6e} -> {PIopt:.6e} W")
    print(f"dual current: P+ = {Psca(Ip, 'p'):.6e}, P- = {Psca(Im, 'm'):.6e} W, "
          f"||(I+ - I-)/(2h)|| / ||I|| = {dIratio:.3e}, ||I+ - I-|| / ||I|| = {np.linalg.norm(Ip - Im)/np.linalg.norm(Imean):.3e}")
    print(f"global residuals / 2P_ub per block (+, -, sum, diff): Re {loc.sum(axis=1).real/(2*PlbGCD)}, "
          f"Im {loc.sum(axis=1).imag/(2*PlbGCD)}")
    print(f"max local |test| / 2P_ub per block: {np.max(np.abs(loc), axis=1)/(2*PlbGCD)}")

    # ---------------------------------------------------------------- direct inference
    Lam = {s: Zsf[s] * np.diag(L) for s in "pm"}
    Zp = {s: Z[s] - np.diag(Lam[s]) for s in "pm"}
    Ix = {"p": Ip, "m": Im}
    bb = {s: (V[s] - Zp[s] @ Ix[s]) / Lam[s] for s in "pm"}
    d = np.clip(sum(np.real(bb[s].conj() * Ix[s]) for s in "pm") / sum(np.abs(bb[s])**2 for s in "pm"), 0, 1)
    res = {s: np.linalg.norm(Ix[s] - d * bb[s]) / np.linalg.norm(Ix[s]) for s in "pm"}

    def realizedP(dd):
        Ir = {s: np.linalg.solve(np.diag(Lam[s]) + dd[:, None] * Zp[s], dd * V[s]) for s in "pm"}
        return Pmean(Ir["p"], Ir["m"])

    Pd = realizedP(d)
    order = np.argsort(-d, kind="stable")
    sweepP = np.zeros(N)
    for n in range(1, N + 1):
        sweepP[n - 1] = Pmean(*solveStructure(np.sort(order[:n])))
    nBest = int(np.argmax(sweepP)) + 1
    Sbest = np.sort(order[:nBest])
    Ibp, Ibm = solveStructure(Sbest)
    dIratioBest = np.linalg.norm((Ibp - Ibm) / (2 * h)) / np.linalg.norm(0.5 * (Ibp + Ibm))
    print(f"LSQ fit residual: + {res['p']:.3e}, - {res['m']:.3e}, mean d = {d.mean():.3f}; realized relaxed {Pd:.6e} W")
    print(f"threshold sweep: best {sweepP[nBest-1]:.6e} W ({nBest} BFs), ||(I+ - I-)/(2h)|| / ||I|| = {dIratioBest:.3e}")
    print(f"summary: full plate {Pfull:.6e}, power only GCD +/- {PlbBase['p']:.6e} / {PlbBase['m']:.6e}, "
          f"two frequencies global / GCD {PlbGlobal:.6e} / {PlbGCD:.6e} W")

    Smask = np.zeros(N)
    Smask[Sbest] = 1
    col = lambda v: np.asarray(v).reshape(-1, 1)
    out = dict(Pfull=Pfull, PlbBasePlus=PlbBase["p"], PlbBaseMinus=PlbBase["m"], PlbGlobal=PlbGlobal,
               PlbGCD=PlbGCD, xstar=col(xstar), xopt=col(x), IstarPlus=col(Is), Iplus=col(Ip), Iminus=col(Im),
               dIfd=col(dIfd), PIopt=PIopt, dIratio=dIratio, d=col(d), Pd=Pd, resPlus=res["p"], resMinus=res["m"],
               sweepN=col(np.arange(1, N + 1)), sweepP=col(sweepP), PsweepBest=sweepP[nBest - 1],
               Sbest=col(Smask), IbestPlus=col(Ibp), IbestMinus=col(Ibm), dIratioBest=dIratioBest,
               Lstar=Lstar, lagsGlobal=col(lagsGlobal), lagsGCD=col(gQ.current_lags), Pdata=Pdata,
               minEigGCD=minEig, nNull=k, localTests=loc.T, gcdHist=np.array(gcdHist).reshape(-1, 2),
               theta=theta, h=h, dScale=dScale, Ngcd=Ngcd, NgcdBase=NgcdBase, gcdBlocks=gcdBlocks)
    sio.savemat(outFile, out)
    print(f"saved {outFile}")
    return out


if __name__ == "__main__":
    main(sys.argv)
