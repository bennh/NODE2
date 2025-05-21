import casadi as ca
import numpy as np

def solve_cnlls_gauss_newton(w: ca.MX,
                              F1: ca.MX,
                              w0: np.ndarray,
                              max_iter: int = 20,
                              tol: float = 1e-6) -> dict:
    """
    Solve a nonlinear least-squares problem via Gauss-Newton without constraints.

    At each iteration k:
      - Linearize residual F1 at w_k
      - Solve normal equations: J1^T J1 dw = -J1^T r
      - Optionally perform line search on the GN step
      - Update w_{k+1} = w_k + t*dw
    Terminates when ||dw|| < tol or max_iter reached.

    Returns:
        result dict containing 'x', 'iterations', 'converged'.
    """
    # Compile evaluation functions
    F1_fun = ca.Function('F1_fun', [w], [F1])
    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])

    xk = w0.copy()
    converged = False

    # Backtracking line-search parameters
    rho = 0.5      # step reduction factor
    c1  = 1e-4     # Armijo constant

    for k in range(max_iter):
        # 1. Evaluate residuals and Jacobian
        r = F1_fun(xk).full().flatten()    # residual vector
        J1 = J1_fun(xk).full()             # m x n Jacobian

        # 2. Solve Gauss-Newton step via normal equations
        H = J1.T @ J1                       # approximate Hessian
        g = J1.T @ r                        # gradient term
        # Solve H dw = -g
        dw_opt = np.linalg.solve(H, -g)

        # 3. Optional backtracking line search
        t = 1.0
        orig_cost = np.linalg.norm(r)**2
        descent   = g @ dw_opt             # directional derivative
        while True:
            x_trial = xk + t * dw_opt
            r_trial = F1_fun(x_trial).full().flatten()
            if np.linalg.norm(r_trial)**2 <= orig_cost + c1 * t * descent:
                break
            t *= rho

        # 4. Update iterate
        xk = xk + t * dw_opt

        # 5. Check convergence on step norm
        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {
        'x': xk,
        'iterations': k + 1,
        'converged': converged
    }

