"""
Generalized Constraint Descent (GCD).

GCD tightens dual bounds for shared projection QCQPs by iteratively:
1. Adding new shared projection constraints likely to tighten the bound.
2. Merging older constraints to keep the total count small.

For usage examples see:
- examples/limits/LDOS.ipynb
- examples/verlan/LDOS_verlan.ipynb

Mathematical details: Appendix B of https://arxiv.org/abs/2504.10469
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.cvxopt.optimization import OptimizationHyperparameters
from dolphindes.types import ComplexArray
from dolphindes.util import CRdot, Sym


@dataclass(frozen=True)
class GCDHyperparameters:
    """Hyperparameters for GCD algorithm.

    Attributes
    ----------
    max_proj_cstrt_num : int
        Maximum number of projector constraints to keep during GCD.
    orthonormalize : bool
        Whether to keep projector constraints orthonormalized.
    opt_params : OptimizationHyperparameters | None
        Optimization hyperparameters used for the internal dual solve at each GCD
        iteration. If None, GCD uses defaults suitable for frequent re-solves
        (notably `max_restart=1`).
    max_gcd_iter_num : int
        Maximum number of GCD iterations.
    gcd_iter_period : int
        Period for checking GCD convergence.
    gcd_tol : float
        Relative tolerance for GCD convergence.
    """

    max_proj_cstrt_num: int = 10
    orthonormalize: bool = True
    opt_params: OptimizationHyperparameters | None = None
    max_gcd_iter_num: int = 50
    gcd_iter_period: int = 5
    gcd_tol: float = 1e-2


def merge_lead_constraints(QCQP: _SharedProjQCQP, merged_num: int = 2) -> None:
    """
    Merge the first m shared projection constraints of QCQP into a single one.

    Also, adjust the Lagrange multipliers so the dual value is the same.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; must be at least 2.

    Raises
    ------
    ValueError
        If merged_num < 2 or if there are insufficient constraints for merging.
    """
    proj_cstrt_num = len(QCQP.Proj)
    if merged_num < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if proj_cstrt_num < merged_num:
        raise ValueError("Number of constraints insufficient for size of merge.")

    if QCQP.current_lags is None:
        raise ValueError("Cannot merge constraints: QCQP.current_lags is None.")

    new_P = QCQP.Proj.Pstruct.astype(complex, copy=True)
    new_P.data[:] = 0.0
    for i in range(merged_num):
        # keep in mind the sharedProj multipliers come first in current_lags
        new_P += QCQP.current_lags[i] * QCQP.Proj[i]

    Pnorm = la.norm(new_P.data)
    new_P /= Pnorm

    QCQP.Proj[merged_num - 1] = new_P
    QCQP.Proj.erase_leading(merged_num - 1)

    # update QCQP
    if hasattr(QCQP, "precomputed_As"):
        # updated precomputed_As
        QCQP.precomputed_As[merged_num - 1] *= QCQP.current_lags[merged_num - 1]
        for i in range(merged_num - 1):
            QCQP.precomputed_As[merged_num - 1] += (
                QCQP.precomputed_As[i] * QCQP.current_lags[i]
            )
        QCQP.precomputed_As[merged_num - 1] /= Pnorm
        del QCQP.precomputed_As[: merged_num - 1]

    if hasattr(QCQP, "Fs"):
        QCQP.Fs = QCQP.Fs[:, merged_num - 1 :]
        
        # preconditioner update
        # QCQP.Fs[:, 0] = QCQP.A2.conj().T @ (new_P.conj().T @ QCQP.s1)
        # Calculate the 3 components of Fs for the merged projector
        Fs0_new = QCQP.A2.conj().T @ (new_P.conj().T @ QCQP.s1)
        FsL_new = (QCQP.A2.conj().T @ (new_P.conj().T @ (-QCQP.A1.conj().T @ QCQP.i1))) / 2
        FsR_new = (-QCQP.A1 @ (new_P @ (QCQP.A2 @ QCQP.i1))) / 2
        QCQP.Fs[:, 0] = Fs0_new + FsL_new + FsR_new
        # end of update

    QCQP.current_lags = QCQP.current_lags[merged_num - 1 :]
    QCQP.current_lags[0] = Pnorm
    QCQP.n_proj_constr = len(QCQP.Proj)

    QCQP.current_grad = QCQP.current_hess = None

# augmented lagrange orthonormalization update
def _get_Mi(QCQP, P_data) -> sp.spmatrix | np.ndarray:
    """
    Constructs the block matrix M_i for a given 1D projector data vector.
    Safely handles array dimensions for block assembly.
    """
    P = QCQP.Proj.Pstruct.astype(complex, copy=True)
    P.data = P_data
    
    # Calculate individual blocks
    B11 = -QCQP.A1 @ P @ QCQP.A2
    B12 = B11 @ QCQP.i1 + P.conj().T @ QCQP.s1
    B21 = QCQP.i1.conj().T @ B11 + QCQP.s1.conj().T @ P
    B22 = QCQP.i1.conj().T @ B12 
    
    # Force 2D shapes for block assembly
    # B12 must be a column vector (N, 1)
    B12_2d = np.reshape(B12, (-1, 1))
    # B21 must be a row vector (1, N)
    B21_2d = np.reshape(B21, (1, -1))
    # B22 must be a 1x1 2D array
    B22_2d = np.atleast_2d(B22)
    
    if sp.issparse(B11):
        # Explicit conversion to sparse to avoid mixed-type block errors
        B12_sp = sp.csr_matrix(B12_2d)
        B21_sp = sp.csr_matrix(B21_2d)
        B22_sp = sp.csr_matrix(B22_2d)
        return sp.bmat([[B11, B12_sp], [B21_sp, B22_sp]], format='csr')
    else:
        # Standard numpy block for dense arrays
        return np.block([[B11, B12_2d], [B21_2d, B22_2d]])

def _inner_product_Mi(QCQP, Pdata_A, Pdata_B, real_part=True) -> float | complex:
    """
    Calculates the inner product as a trace: trace(M_A^H @ M_B).
    """
    Ma = _get_Mi(QCQP, Pdata_A)
    Mb = _get_Mi(QCQP, Pdata_B)
    
    # Hermitian conjugate for norm stability in the Gram-Schmidt process
    Ma_H = Ma.conj().T
    
    if sp.issparse(Ma_H) and sp.issparse(Mb):
        # Efficient calculation of trace(A @ B) for sparse matrices: sum(A_ij * B_ji)
        trace_val = Ma_H.multiply(Mb.T).sum()
    else:
        trace_val = np.trace(Ma_H @ Mb)
        
    return np.real(trace_val) if real_part else trace_val
# end of update

def add_constraints(
    QCQP: _SharedProjQCQP,
    added_Pdata_list: list[ComplexArray],
    orthonormalize: bool = True,
) -> None:
    """
    Add new shared projection constraints into an existing QCQP.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which the new constraints are added in.
    added_Pdata_list : list
        List of 1d numpy arrays representing the sparse entries of
        the new constraints to be added in, with sparsity structure QCQP.Proj.Pstruct
    orthonormalize : bool, optional
        If true, assume that QCQP has orthonormal constraints and keeps it that way
    """
    x_size = QCQP.Proj.Pstruct.size
    proj_cstrt_num = QCQP.n_proj_constr
    added_Pdata_num = len(added_Pdata_list)

    if QCQP.current_lags is not None:
        new_lags = np.zeros(
            proj_cstrt_num + added_Pdata_num + QCQP.n_gen_constr, dtype=float
        )
        new_lags[:proj_cstrt_num] = QCQP.current_lags[:proj_cstrt_num]
        new_lags[proj_cstrt_num + added_Pdata_num :] = QCQP.current_lags[
            proj_cstrt_num:
        ]
        QCQP.current_lags = new_lags

    # update orthonormalization 
    # if orthonormalize:
    #     # in this case assume that existing Pdata is already orthonormalized
    #     new_Pdata = np.zeros((x_size, proj_cstrt_num + added_Pdata_num), dtype=complex)
    #     new_Pdata[:, :proj_cstrt_num] = QCQP.Proj.get_Pdata_column_stack()

    #     for m in range(added_Pdata_num):
    #         # do (modified) Gram-Schmidt orthogonalization for each added Pdata
    #         for j in range(proj_cstrt_num + m):
    #             added_Pdata_list[m] -= (
    #                 CRdot(new_Pdata[:, j], added_Pdata_list[m]) * new_Pdata[:, j]
    #             )
    #         added_Pdata_list[m] /= la.norm(added_Pdata_list[m])

    #         new_Pdata[:, proj_cstrt_num + m] = added_Pdata_list[m]
    
    if orthonormalize:
        new_Pdata = np.zeros((x_size, proj_cstrt_num + added_Pdata_num), dtype=complex)
        new_Pdata[:, :proj_cstrt_num] = QCQP.Proj.get_Pdata_column_stack()

        for m in range(added_Pdata_num):
            # Modified Gram-Schmidt with the new metric over M matrices
            for j in range(proj_cstrt_num + m):
                # Calculate the projection using the new metric
                proj_coef = _inner_product_Mi(QCQP, new_Pdata[:, j], added_Pdata_list[m])
                added_Pdata_list[m] -= proj_coef * new_Pdata[:, j]
            
            # Normalize with respect to the new metric
            norm_sq = _inner_product_Mi(QCQP, added_Pdata_list[m], added_Pdata_list[m])
            if norm_sq > 1e-14:
                added_Pdata_list[m] /= np.sqrt(norm_sq)

            new_Pdata[:, proj_cstrt_num + m] = added_Pdata_list[m]
    # end of update

    # update QCQP
    for m, added_Pdata in enumerate(added_Pdata_list):
        Pnew = QCQP.Proj.Pstruct.astype(complex, copy=True)
        Pnew.data = added_Pdata
        QCQP.Proj.append(Pnew)

        if hasattr(QCQP, "precomputed_As"):
            # updated precomputed_As
            QCQP.precomputed_As.insert(
                proj_cstrt_num + m, Sym(QCQP.A1 @ Pnew @ QCQP.A2)
            )

    if hasattr(QCQP, "Fs"):
        new_Fs = np.zeros(
            (QCQP.Fs.shape[0], len(QCQP.Proj) + QCQP.n_gen_constr), dtype=complex
        )
        
        # preconditioner  update
        # new_Fs[:, : len(QCQP.Proj)] = QCQP.A2.conj().T @ QCQP.Proj.allP_at_v(
        #     QCQP.s1, dagger=True
        # )
        # Calculate the 3 components of Fs for the entire projector set
        Pv = QCQP.Proj.allP_at_v(QCQP.s1, dagger=True)
        Fs0 = QCQP.A2.conj().T @ Pv
        Pv_precondL = QCQP.Proj.allP_at_v(-QCQP.A1.conj().T @ QCQP.i1, dagger=True)
        FsL = (QCQP.A2.conj().T @ Pv_precondL) / 2
        Pv_precondR = QCQP.Proj.allP_at_v(QCQP.A2 @ QCQP.i1, dagger=False)
        FsR = (-QCQP.A1 @ Pv_precondR) / 2
        new_Fs[:, : len(QCQP.Proj)] = Fs0 + FsL + FsR  
        # end of update      
        
        new_Fs[:, len(QCQP.Proj) :] = QCQP.Fs[:, proj_cstrt_num:]
        QCQP.Fs = new_Fs

    QCQP.n_proj_constr = len(QCQP.Proj)
    QCQP.current_grad = QCQP.current_hess = None


def run_gcd(
    QCQP: _SharedProjQCQP,
    gcd_params: GCDHyperparameters = GCDHyperparameters(),
) -> None:
    """
    Perform generalized constraint descent to gradually refine dual bound on QCQP.

    At each GCD iteration, add two new constraints:
    1.a constraint generated so the corresponding dual derivative is large,
    to hopefully tighten the dual bound
    2. a constraint generated so the corresponding derivative of the smallest
    Lagrangian quadratic form eigenvalue is large, to help the dual optimization
    navigate the semi-definite boundary

    If the total number of constraints is larger than max_gcd_proj_cstrt_num combine
    the earlier constraints to keep the total number of constraints fixed. Setting
    max_proj_cstrt_num large enough will eventually result in evaluating the dual bound
    with all possible constraints, which gives the tightest bound but may be extremely
    expensive. The goal of GCD is to approximate this tightest bound with greatly
    reduced computational cost.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        The SharedProjQCQP for which we compute and refine dual bounds.
    max_proj_cstrt_num : int, optional
        The maximum projection constraint number for QCQP. The default is 10.
    orthonormalize : bool, optional
        Whether or not to orthonormalize the constraint projectors. The default is True.
    opt_params : OptimizationHyperparameters, optional
        Optimization hyperparameters for the internal dual solve at every GCD
        iteration.
    max_gcd_iter_num : int, optional
        Maximum number of GCD iterations, by default 50.
    gcd_iter_period : int, optional
        Period for checking convergence, by default 5.
    gcd_tol : float, optional
        Tolerance for convergence, by default 1e-2.

    Notes
    -----
    TODO: formalize optimization and convergence parameters.
    """
    # Since GCD constantly changes constraints, there is typically little value in
    # running multiple outer penalty-reduction restarts for each intermediate solve.
    # Default to a single outer iteration (max_restart=1).
    if gcd_params.opt_params is None:
        opt_params = OptimizationHyperparameters(
            opttol=1e-2,
            gradConverge=False,
            min_inner_iter=5,
            max_restart=1,
            penalty_ratio=1e-2,
            penalty_reduction=0.1,
            break_iter_period=20,
            verbose=int(QCQP.verbose - 1),
        )
    else:
        opt_params = gcd_params.opt_params

    # get to feasible point
    # TODO: revamp find_feasible_lags
    QCQP.current_lags = QCQP.find_feasible_lags()
    assert QCQP.current_lags is not None

    orthonormalize = gcd_params.orthonormalize
    max_proj_cstrt_num = gcd_params.max_proj_cstrt_num
    max_gcd_iter_num = gcd_params.max_gcd_iter_num
    gcd_iter_period = gcd_params.gcd_iter_period
    gcd_tol = gcd_params.gcd_tol

    # update orthonormalization
    # if orthonormalize:
    #     # orthonormalize QCQP
    #     # informally checked for correctness
    #     x_size = QCQP.Proj.Pstruct.size
    #     proj_cstrt_num = QCQP.n_proj_constr
    #     Pdata = QCQP.Proj.get_Pdata_column_stack()
    #     realext_Pdata = np.zeros((2 * x_size, proj_cstrt_num), dtype=float)
    #     realext_Pdata[:x_size, :] = np.real(Pdata)
    #     realext_Pdata[x_size:, :] = np.imag(Pdata)
    #     realext_Pdata_Q, realext_Pdata_R = la.qr(realext_Pdata, mode="economic")

    #     QCQP.Proj.set_Pdata_column_stack(
    #         realext_Pdata_Q[:x_size, :] + 1j * realext_Pdata_Q[x_size:, :]
    #     )
    #     QCQP.current_lags[: QCQP.n_proj_constr] = (
    #         realext_Pdata_R @ QCQP.current_lags[: QCQP.n_proj_constr]
    #     )
    #     QCQP.compute_precomputed_values()
    if orthonormalize:
        # The original la.qr orthogonalization must be replaced
        # because the QR decomposition relies on the standard Euclidean metric.
        # We use Modified Gram-Schmidt and track the transformation matrix R
        # to correctly update the Lagrange multipliers and preserve the dual value.
        proj_cstrt_num = QCQP.n_proj_constr
        Pdata = QCQP.Proj.get_Pdata_column_stack()
        
        ortho_Pdata = np.zeros_like(Pdata)
        R = np.zeros((proj_cstrt_num, proj_cstrt_num), dtype=float)
        
        for m in range(proj_cstrt_num):
            current_vec = Pdata[:, m].copy()
            for j in range(m):
                # Compute projection coefficient using the custom metric
                proj_coef = _inner_product_Mi(QCQP, ortho_Pdata[:, j], current_vec)
                R[j, m] = proj_coef
                current_vec -= proj_coef * ortho_Pdata[:, j]
            
            # Normalize with respect to the custom metric
            norm_sq = _inner_product_Mi(QCQP, current_vec, current_vec)
            if norm_sq > 1e-14:
                R[m, m] = np.sqrt(norm_sq)
                ortho_Pdata[:, m] = current_vec / R[m, m]
            else:
                R[m, m] = 0.0
                ortho_Pdata[:, m] = current_vec

        QCQP.Proj.set_Pdata_column_stack(ortho_Pdata)
        
        # Update the Lagrange multipliers using the transition matrix R 
        # to keep the dual function value invariant
        QCQP.current_lags[: QCQP.n_proj_constr] = (
            R @ QCQP.current_lags[: QCQP.n_proj_constr]
        )
        
        QCQP.compute_precomputed_values()
    # end of update

    ## gcd loop
    gcd_iter_num = 0
    gcd_prev_dual: float = np.inf
    while True:
        gcd_iter_num += 1
        # solve current dual problem
        # print('at gcd iter num', gcd_iter_num)
        # print('QCQP.current_lags', QCQP.current_lags)
        # print('QCQP.Fs.shape', QCQP.Fs.shape)
        QCQP.solve_current_dual_problem(
            "newton", init_lags=QCQP.current_lags, opt_params=opt_params
        )
        assert QCQP.current_dual is not None
        assert QCQP.current_xstar is not None

        print(
            f"At GCD iteration #{gcd_iter_num}, best dual bound found is \
            {QCQP.current_dual}."
        )

        ## termination conditions
        if gcd_iter_num > max_gcd_iter_num:
            break
        if gcd_iter_num % gcd_iter_period == 0:
            if gcd_prev_dual - QCQP.current_dual < gcd_tol * abs(gcd_prev_dual):
                break
            gcd_prev_dual = QCQP.current_dual

        ## generate new constraints
        new_Pdata_list = []
        Pstruct_rows, Pstruct_cols = QCQP.Proj.Pstruct.nonzero()
        ## generate max dualgrad constraint
        
        # update preconditioner
        # maxViol_Pdiag = (2 * QCQP.s1 - (QCQP.A1.conj().T @ QCQP.current_xstar))[
        #     Pstruct_rows
        # ] * (QCQP.A2 @ QCQP.current_xstar).conj()[Pstruct_cols]
        u = QCQP.A1.conj().T @ QCQP.current_xstar
        y = QCQP.A2 @ QCQP.current_xstar
        u_precond = QCQP.A1.conj().T @ QCQP.i1
        y_precond = QCQP.A2 @ QCQP.i1
        left_vec = 2 * QCQP.s1 - u - u_precond
        right_vec = (y + y_precond).conj()
        maxViol_Pdiag = left_vec[Pstruct_rows] * right_vec[Pstruct_cols]
        # end of update

        if la.norm(maxViol_Pdiag) >= 1e-14:
            new_Pdata_list.append(maxViol_Pdiag)
            # skip this new constraint if maxViol_Pdiag is uniformly 0
            # can happen if there are no linear forms in objective and all constraints

        ## generate min A eig constraint
        minAeigv, minAeigw = QCQP._get_PSD_penalty(QCQP.current_lags)
        minAeig_Pdiag = (QCQP.A1.conj().T @ minAeigv)[Pstruct_rows] * (
            QCQP.A2 @ minAeigv
        ).conj()[Pstruct_cols]

        minAeig_Pdiag /= np.sqrt(np.real(minAeig_Pdiag.conj() * minAeig_Pdiag))
        # minAeig_Pdiag * np.sqrt(np.real(maxViol_Pdiag.conj() * maxViol_Pdiag))
        # use the same relative weights for minAeig_Pdiag as maxViol_Pdiag
        # informally checked that minAeigw increases when increasing multiplier of
        # minAeig_Pdiag
        # new_Pdata_list.append(minAeig_Pdiag)

        ## add new constraints
        QCQP.add_constraints(new_Pdata_list, orthonormalize=orthonormalize)
        # informally checked that new constraints are added in orthonormal fashion

        ## merge old constraints if necessary
        if len(QCQP.Proj) > max_proj_cstrt_num:
            QCQP.merge_lead_constraints(
                merged_num=len(QCQP.Proj) - max_proj_cstrt_num + 1
            )
