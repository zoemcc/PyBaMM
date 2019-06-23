#
# Tests for the DAE Solver class
#
import pybamm
import unittest
import pydae

class TestDaeCppSolver(unittest.TestCase):
    def test_wrapper(self):
        # Use Robertson example from dae-cpp

        # Solves Robertson Problem as Semi-Explicit Differential Algebraic Equations
        # (see https://www.mathworks.com/help/matlab/ref/ode15s.html):
        # x1' = -0.04*x1 + 1e4*x2*x3
        # x2' =  0.04*x1 - 1e4*x2*x3 - 3e7*x2^2
        # 0  =  x1 + x2 + x3 - 1
        #
        # Initial conditions are: x1 = 1, x2 = 0, x3 = 0.

        def fun_mass_matrix(M):
            # Non-zero and/or diagonal elements
            M.A[:] = [1, 1, 0]

            # Column index of each element given above
            M.ja[:] = [0, 1, 2]

            # Index of the first element for each row
            M.ia[:] = [0, 1, 2, 3]

        def fun_rhs(x, f, t):
            f[0] = -0.04 * x[0] + 1.0e4 * x[1] * x[2]
            f[1] = 0.04 * x[0] - 1.0e4 * x[1] * x[2] - 3.0e7 * x[1] * x[1]
            f[2] = x[0] + x[1] + x[2] - 1

        def fun_jacobian(J, x, t):
            # Initialize Jacobian in sparse format
            J.A.resize(9)
            J.ja.resize(9)
            J.ia.resize(4)

            # Non-zero elements
            J.A[0] = -0.04
            J.A[1] = 1.0e4 * x[2]
            J.A[2] = 1.0e4 * x[1]
            J.A[3] = 0.04
            J.A[4] = -1.0e4 * x[2] - 6.0e7 * x[1]
            J.A[5] = -1.0e4 * x[1]
            J.A[6] = 1.0
            J.A[7] = 1.0
            J.A[8] = 1.0

            # Column index of each element given above
            J.ja[0] = 0
            J.ja[1] = 1
            J.ja[2] = 2
            J.ja[3] = 0
            J.ja[4] = 1
            J.ja[5] = 2
            J.ja[6] = 0
            J.ja[7] = 1
            J.ja[8] = 2

            # Index of the first element for each row
            J.ia[0] = 0
            J.ia[1] = 3
            J.ia[2] = 6
            J.ia[3] = 9


        # Solution time 0 <= t <= t1
        t1 = 4.0e6

        # Define the state vector
        # Initial conditions.
        # We will use slightly inconsistent initial condition to test
        # initialization.
        x = pydae.state_type([1, 0, 1e-3])

        # Set up the RHS of the problem.
        # Class MyRHS inherits abstract RHS class from dae-cpp library.
        rhs = pydae.RHS(fun_rhs)

        # Set up the Mass Matrix of the problem.
        # MyMassMatrix inherits abstract MassMatrix class from dae-cpp library.
        mass = pydae.MassMatrix(fun_mass_matrix)

        # Create an instance of the solver options and update some of the solver
        # parameters defined in solver_options.h
        opt = pydae.SolverOptions()

        opt.dt_init               = 1.0e-6    # Change initial time step
        opt.dt_max                = t1 / 100  # Set maximum time step
        opt.time_stepping         = 1         # S-SATS works better here
        opt.dt_increase_threshold = 2         # Time step amplification threshold
        opt.atol                  = 1e-6      # Absolute tolerance
        opt.bdf_order             = 6         # Set BDF-6

        # We can override Jacobian class from dae-cpp library and provide
        # analytical Jacobian. We shall do this for single precision:
#ifdef DAE_SINGLE
        jac = pydae.AnalyticalJacobian(rhs, fun_jacobian)
        jac_numerical = pydae.NumericalJacobian(rhs, 1e-10)

        # Create an instance of the solver with particular RHS, Mass matrix,
        # Jacobian and solver options
        solve = pydae.Solver(rhs, jac, mass, opt)
        solve_numJ = pydae.Solver(rhs, jac_numerical, mass, opt)

        # Now we are ready to solve the set of DAEs
        status = solve(x, t1)




if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
