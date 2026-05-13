import numpy as np
import scipy.sparse as sp
import copy
import matplotlib.pyplot as plt
from dolphindes.cvxopt import DenseSharedProjQCQP, OptimizationHyperparameters
from dolphindes.cvxopt.gcd import GCDHyperparameters

# 1. Define the range of factors to sweep
# Sweeping from 0.01 to 1.0 (logarithmic scale is often better for conductivity)
factors = np.logspace(-2, 0, 10) 
min_eigenvalues = []

# Constants assumed from previous cells:
# omega, epsilon_0, c, Lmat, R0, X0, Vinc, Bmat, bVec, beta, eVec, Vzero, Pdiags, X0Norm

print(f"Starting sweep over {len(factors)} points...")

for i, factor in enumerate(factors):
    # --- A. Update Physics based on Factor ---
    copper_conductivity = factor * 5.96e7 # Adjusted conductivity
    copper_permittivity = -1j * copper_conductivity / (omega * epsilon_0)
    
    # Recalculate Surface Impedance (Zs)
    delta = -1.0 / np.imag(omega/c * np.sqrt(copper_permittivity))
    Zs = 1.0 / (copper_conductivity * delta)
    
    # --- B. Update Matrices ---
    # Dissipation matrix changes because Zs changes
    Rmat0 = R0 + Zs * Lmat 
    Rmat0Norm = np.linalg.norm(Rmat0, ord=2)
    
    # Impedance matrix Zmat and Umat
    Zmat = Rmat0 + 1j * X0
    Umat = 1j * Zmat.conj()
    
    # Re-normalize constraints
    norm_list = [X0Norm, Rmat0Norm]
    Plist = [sp.diags(Pdiags[:, k] / norm_list[k]) for k in range(Pdiags.shape[1])]

    # --- C. Setup QCQP ---
    # We create a new QCQP instance for this specific conductivity
    # verbose=0 to keep the output clean during the loop
    gcd_QCQP = DenseSharedProjQCQP(Bmat, bVec, beta,
                               Umat, eVec,
                               Plist, verbose=0)

    # --- D. Run GCD ---
    gcd_tol = 1e-2
    gcd_params = GCDHyperparameters(gcd_tol=gcd_tol)
    
    # Run the solver
    gcd_QCQP.run_gcd(gcd_params=gcd_params)
    
    # --- E. Calculate Minimal Eigenvalue ---
    lags = gcd_QCQP.current_lags
    totalA = gcd_QCQP._get_total_A(lags)
    
    # Calculate eigenvalues
    # Note: Using eigvalsh usually assumes Hermitian, which is generally true for the dual matrix
    # but we stick to np.linalg.eig as per your notebook for safety
    eigenvals, _ = np.linalg.eig(totalA)
    
    # We take the real part of the minimum eigenvalue
    min_eig = np.min(np.real(eigenvals))
    min_eigenvalues.append(min_eig)
    
    print(f"Iter {i+1}/{len(factors)}: Factor={factor:.2e}, Min Eig={min_eig:.2e}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.semilogx(factors, min_eigenvalues, 'b-o', linewidth=2)
plt.grid(True, which="both", ls="-")
plt.xlabel('Conductivity Reduction Factor')
plt.ylabel('Min Eigenvalue (Real Part)')
plt.title('GCD Minimal Eigenvalue vs Conductivity Reduction')
plt.axhline(0, color='r', linestyle='--', label='PSD Limit')
plt.legend()
plt.show()