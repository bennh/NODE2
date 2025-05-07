import casadi as ca

class CNLLSSolver:
    """
    A solver for constrained nonlinear least squares problems of the form:
        min_z 0.5*\|F1(z)\|^2
        s.t. F2(z) = 0, F3(z) >= 0
    Supports:
      - IPOPT via CasADi nlpsol
      - Generalized Gauss-Newton (GGN) + qpOASES via CasADi qpsol
    """
    def __init__(self, mode='ggn', opts=None):
        """
        mode: 'ipopt' or 'ggn'
        opts: dict of solver options
        """
        self.mode = mode
        self.opts = opts or {}

    def solve(self, z0, F1, F2, F3=None, J1_fun=None, J2_fun=None, J3_fun=None):
        """
        z0: initial guess (CasADi DM or SX)
        F1: CasADi function mapping z -> residual vector
        F2: CasADi function mapping z -> equality constraints vector
        F3: CasADi function mapping z -> inequality constraints (>=0)
        J1_fun, J2_fun, J3_fun: optional CasADi Jacobian functions
        """
        if self.mode == 'ipopt':
            return self._solve_ipopt(z0, F1, F2, F3)
        elif self.mode == 'ggn':
            return self._solve_ggn(z0, F1, F2, F3, J1_fun, J2_fun, J3_fun)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _solve_ipopt(self, z0, F1, F2, F3):
        # Pack problem into NLP form
        z = ca.MX.sym('z', z0.size1())
        obj = 0.5 * ca.dot(F1(z), F1(z))
        g = F2(z)
        lbg = ca.DM.zeros(g.size1())
        ubg = ca.DM.zeros(g.size1())
        if F3 is not None:
            h = F3(z)
            # inequality: h(z) >= 0 -> -h <= 0
            g = ca.vertcat(g, -h)
            lbg = ca.vertcat(lbg, ca.DM([-ca.inf] * h.size1()))
            ubg = ca.vertcat(ubg, ca.DM.zeros(h.size1()))

        nlp = {'x': z, 'f': obj, 'g': g}
        opts = {'ipopt.print_level': 5}
        opts.update(self.opts)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=z0, lbg=lbg, ubg=ubg)
        z_opt = sol['x']
        return z_opt, sol

    def _solve_ggn(self, z0, F1, F2, F3, J1_fun=None, J2_fun=None, J3_fun=None,
                   max_iter=20, tol=1e-6):
        # Initialize
        z = ca.DM(z0)
        if J1_fun is None:
            J1_fun = ca.Function('J1', [F1.sx_in()], [ca.jacobian(F1(F1.sx_in()), F1.sx_in())])
        if J2_fun is None:
            J2_fun = ca.Function('J2', [F2.sx_in()], [ca.jacobian(F2(F2.sx_in()), F2.sx_in())])
        if F3 is not None and J3_fun is None:
            J3_fun = ca.Function('J3', [F3.sx_in()], [ca.jacobian(F3(F3.sx_in()), F3.sx_in())])

        for k in range(max_iter):
            # Evaluate residuals and Jacobians
            r1 = F1(z)
            r2 = F2(z)
            J1 = J1_fun(z)
            J2 = J2_fun(z)
            if F3 is not None:
                r3 = F3(z)
                J3 = J3_fun(z)
            # Form QP: min 0.5*\|J1*dz + r1\|^2  s.t. J2*dz + r2 = 0, J3*dz + r3 >= 0
            # Equivalent QP: 0.5 dz^T (J1^T J1) dz + (J1^T r1)^T dz
            H = J1.T @ J1
            g = J1.T @ r1
            # Equality constraints
            A = J2
            lbA = -r2; ubA = -r2
            # Inequality constraints
            if F3 is not None:
                C = J3
                lbC = -r3; ubC = ca.DM([ca.inf] * r3.size1())
                A = ca.vertcat(A, C)
                lbA = ca.vertcat(lbA, lbC)
                ubA = ca.vertcat(ubA, ubC)
            # Build CasADi QP
            qp = {'h': H, 'g': g, 'a': A, 'lba': lbA, 'uba': ubA}
            qp_opts = {'printLevel': 'none'}
            qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts)
            sol = qp_solver()
            dz = sol['x']
            # Check convergence
            if ca.norm_2(dz) < tol:
                break
            # Line search
            t = 1.0
            phi0 = 0.5 * float(r1.T @ r1)
            while True:
                z_new = z + t * dz
                phi_new = 0.5 * float(F1(z_new).T @ F1(z_new))
                if phi_new <= phi0 + 1e-4 * t * float(g.T @ dz):
                    z = z_new
                    break
                t *= 0.5
            # Continue iterations
        return z, {'iterations': k+1}
