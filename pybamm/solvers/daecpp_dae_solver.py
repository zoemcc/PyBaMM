#
# Solver class using dae-cpp DAE solver
# (see `https://github.com/ikorotkin/dae-cpp`)
#
import pybamm
import pydae
import numpy as np

#import scipy.sparse as sparse


class DaecppDaeSolver(pybamm.DaeSolver):
    """Solve a discretised model, using dae-cpp.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    tolerance : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    """

    def __init__(self, method="BDF", tol=1e-8, root_method="lm", root_tol=1e-6):
        # TODO: check if dae-cpp is installed

        super().__init__(method, tol, root_method, root_tol)

    def integrate(
        self, residuals, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        """

        # dae-cpp RHS
        def fun_rhs(x, f, t):
            # solver works with ydot set to zero
            #y = np.array(x)
            #ydot = np.zeros_like(y)
            #f[:] = residuals(t, y, ydot)
            #f = pydae.state_type(residuals(t, x))  # this doesn't work
            for i, fi in enumerate(residuals(t, np.array(x))):
                f[i] = fi

        # dae-cpp Mass Matrix
        # TODO: Currently returns identity matrix of size y0.size.
        # In general it should convert mass_matrix or return identity matrix if
        # mass_matrix=None
        def fun_mass_matrix(M):
            size = y0.size
            M.A.resize(size)
            M.ja.resize(size)
            M.ia.resize(size + 1)
            for i in range(0, size):
                # Non-zero and/or diagonal elements
                M.A[i] = 1
                # Column index of each element given above
                M.ja[i] = i
                # Index of the first element for each row
                M.ia[i] = i
            # To close the matrix
            M.ia[size] = size

        #def eqsres(t, y, ydot, return_residuals):
        #    return_residuals[:] = residuals(t, y, ydot)

        #def rootfn(t, y, ydot, return_root):
        #    return_root[:] = [event(t, y) for event in events]

        #extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol}

        # dae-cpp solver option
        opt = pydae.SolverOptions()
        opt.atol = 1e-12 #self.tol
        opt.verbosity = 0
        #opt.dt_init = 0.0001
        opt.time_stepping = 1
        opt.bdf_order = 2
        #opt.dt_max = 0.0001

        dae_mass = pydae.MassMatrix(fun_mass_matrix)
        dae_rhs = pydae.RHS(fun_rhs)

        if jacobian:
            # TEMPORARY for testing
            dae_jacobian = pydae.NumericalJacobian(dae_rhs, 1e-10)
            #raise NotImplementedError
            
            #jac_y0_t0 = jacobian(t_eval[0], y0)
            #if sparse.issparse(jac_y0_t0):

            #    def jacfn(t, y, ydot, residuals, cj, J):
            #        jac_eval = jacobian(t, y) - cj * mass_matrix
            #        J[:][:] = jac_eval.toarray()

            #else:

            #    def jacfn(t, y, ydot, residuals, cj, J):
            #        jac_eval = jacobian(t, y) - cj * mass_matrix
            #        J[:][:] = jac_eval

            #extra_options.update({"jacfn": jacfn})
        else:
            dae_jacobian = pydae.NumericalJacobian(dae_rhs, 1e-10)

        #if events:
        #    extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        # solver works with ydot0 set to zero
        #ydot0 = np.zeros_like(y0)

        # set the solver up
        dae_solve = pydae.Solver(dae_rhs, dae_jacobian, dae_mass, opt)

        # initial condition
        x = pydae.state_type(y0)

        # solve
        first_pass = True
        for t1 in t_eval:
            # currently t1 cannot be 0 for dae-cpp. TODO: remove this restriction
            if(t1 == 0):
                y_sol = np.reshape(np.array(x), (-1, 1))
                t_sol = np.array([0])
                first_pass = False
                continue
            else:
                # solution for time t1
                status = dae_solve(x, t1)  # x will be overwritten
                if(status != 0):
                    break
                sol_t1 = np.reshape(np.array(x), (-1, 1))
                print("###### t1, x: " + str(np.exp(-0.1*t1)) + "  " + str(x) + " status = " + str(status))

            if(first_pass):
                y_sol = sol_t1
                t_sol = np.array([t1])
                first_pass = False
            else:
                y_sol = np.append(y_sol, sol_t1, axis=1)
                t_sol = np.append(t_sol, t1)

        #dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        #sol = dae_solver.solve(t_eval, y0, ydot0)

        # return solution
        if status == 0:
            # success
            termination = "final time"
            #return pybamm.Solution(
            #    t_sol, np.transpose(y_sol), termination
            #)
            return pybamm.Solution(t_sol, y_sol, termination)
        else:
            # error
            raise pybamm.SolverError("dae-cpp: Error during integration")

        # return solution, we need to tranpose y to match scipy's interface
        #if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
        #    if sol.flag == 0:
        #        termination = "final time"
            # 2 = found root(s)
        #    elif sol.flag == 2:
        #        termination = "event"
        #    return pybamm.Solution(
        #        sol.values.t, np.transpose(sol.values.y), termination
        #    )
        #else:
        #    raise pybamm.SolverError(sol.message)
