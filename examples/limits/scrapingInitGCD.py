"""Initial GCD for the scraping strategies (scrapingGCD_*.ipynb), copper 2:1 patch, ka = 1.

Environment: NGCD (default 50), SEED (default 0), TAG, OPTTOL, GCD_OPTTOL (see scrapingCommon.py).
Writes results/scrapingInitGCD_<TAG>.npz with the projectors, multipliers, bound and dual current.
"""
import os
import time
import copy
import numpy as np
from scrapingCommon import *

NGCD = int(os.environ.get('NGCD', 50))
SEED = int(os.environ.get('SEED', 0))
TAG = os.environ.get('TAG', f"ngcd{NGCD}_seed{SEED}")
np.random.seed(SEED)
print(f"NGCD {NGCD}, SEED {SEED}, TAG {TAG}, OPTTOL {OPTTOL:g}, GCD_OPTTOL {GCD_OPTTOL:g}", flush=True)
print(f"full plate {Pfull:.6e} W; d_0 {measures(Vfull, Vinc)[0]:+.1e}, d_theta {measures(Vfull, Vinc)[1]:+.1e}, "
      f"c_im/r^2 {measures(Vfull, Vinc)[2]:+.1e}; rotation {np.degrees(thOpt):.4e} deg", flush=True)

QCQP = globalQCQP()
print(f"global constraints: bound {QCQP.current_dual:.6f} W, d_0 {measures(QCQP.current_xstar, Vinc)[0]:+.2e}", flush=True)

gQCQP = copy.deepcopy(QCQP)
t1 = time.time()
gQCQP.run_gcd(GCDHyperparameters(max_proj_cstrt_num=2 + 2*NGCD, max_gcd_iter_num=NGCD,
                                 gcd_iter_period=NGCD + 1, gcd_tol=GCD_OPTTOL, opt_params=opt_params))
d0, dth, cim = measures(gQCQP.current_xstar, Vinc)
A = Msym(gQCQP._get_total_A(gQCQP.current_lags))
ev = np.linalg.eigvalsh(A)
print(f"GCD {NGCD}: {time.time()-t1:.0f}s, bound {gQCQP.current_dual:.6f} W, {gQCQP.n_proj_constr} projectors, "
      f"d_0 {d0:+.3e}, d_theta {dth:+.3e}, c_im/r^2 {cim:+.3e}, viol {pointViolation(gQCQP.current_xstar, Vinc):.2e}, "
      f"min eig {ev[0]/ev[-1]:.2e}", flush=True)
saveInitialGCD(gQCQP, f"results/scrapingInitGCD_{TAG}.npz", ngcd=NGCD, seed=SEED)
print(f"saved results/scrapingInitGCD_{TAG}.npz")
