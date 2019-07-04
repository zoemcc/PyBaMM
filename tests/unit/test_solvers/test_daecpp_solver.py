#
# Tests for the DAE Solver class
#
import pybamm
import pydae
import numpy as np
import scipy.sparse as sparse
import unittest
import warnings
from tests import get_mesh_for_testing, get_discretisation_for_testing


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
            M.A.resize(3)   # Matrix size
            M.ja.resize(3)  # Matrix size
            M.ia.resize(4)  # Matrix size + 1

            # Non-zero and/or diagonal elements
            M.A[0] = 1
            M.A[1] = 1
            M.A[2] = 0

            # Column index of each element given above
            M.ja[0] = 0
            M.ja[1] = 1
            M.ja[2] = 2

            # Index of the first element for each row
            M.ia[0] = 0
            M.ia[1] = 1
            M.ia[2] = 2
            M.ia[3] = 3

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
        x0 = pydae.state_type([1, 0, 1e-3])

        # Set up the RHS of the problem.
        # Class MyRHS inherits abstract RHS class from dae-cpp library.
        rhs = pydae.RHS(fun_rhs)

        # Set up the Mass Matrix of the problem.
        # MyMassMatrix inherits abstract MassMatrix class from dae-cpp library.
        mass = pydae.MassMatrix(fun_mass_matrix)

        # We can override Jacobian class from dae-cpp library and provide
        # analytical Jacobian. We shall do this for single precision:
        jac = pydae.AnalyticalJacobian(rhs, fun_jacobian)
        jac_numerical = pydae.NumericalJacobian(rhs, 1e-10)

        for jacobian in [jac, jac_numerical]:
            # Create an instance of the solver options and update some of the solver
            # parameters defined in solver_options.h
            opt = pydae.SolverOptions()

            opt.dt_init = 1.0e-6    # Change initial time step
            opt.dt_max = t1 / 100  # Set maximum time step
            opt.time_stepping = 1         # S-SATS works better here
            opt.dt_increase_threshold = 2         # Time step amplification threshold
            opt.atol = 1e-6      # Absolute tolerance
            opt.bdf_order = 6         # Set BDF-6
            opt.verbosity = 0         # turn off output

            # Create an instance of the solver with particular RHS, Mass matrix,
            # Jacobian and solver options
            solve = pydae.Solver(rhs, jacobian, mass, opt)

            # Now we are ready to solve the set of DAEs
            x = pydae.state_type(x0)
            status = solve(x, t1)

            # Compare results with MATLAB ode15s solution
            x_ref = [0.00051675, 2.068e-9, 0.99948324]
            conservation = abs(x[0] + x[1] + x[2] - 1)

            # Find total relative deviation from the reference solution
            result = 0.0
            for xi, x_refi in zip(x, x_ref):
                result += abs(xi - x_refi) / x_refi * 100

            if jacobian is jac:
                print("Analytical Jacobian:")
            else:
                print("Numerical Jacobian:")
            print("\tTotal relative error: {} %".format(result))
            print("\tConservation law absolute deviation: {}".format(conservation))
            self.assertLessEqual(result, 1)
            self.assertLessEqual(conservation, 1e-10)
            self.assertEqual(status, 0)

    def test_ode_integrate(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth(t, y):
            return 0.5 * np.ones_like(y)

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_growth, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_decay(t, y):
            return -0.1 * y

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(exponential_decay, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-4)
        self.assertEqual(solution.termination, "final time")

    def test_ode_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        def sqrt_decay(t, y):
            return -np.sqrt(y)

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.DaecppDaeSolver()
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(sqrt_decay, y0, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_ode_integrate_with_event(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_decay(t, y):
            return -2 * np.ones_like(y)

        def y_equal_0(t, y):
            return y[0]

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(constant_decay, y0, t_eval, events=[y_equal_0])
        np.testing.assert_allclose(1 - 2 * solution.t, solution.y[0])
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_less(0, solution.y[0])
        np.testing.assert_array_less(solution.t, 0.5)
        self.assertEqual(solution.termination, "event")

        # Exponential growth
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_growth(t, y):
            return y

        def y_eq_9(t, y):
            return y - 9

        def ysq_eq_7(t, y):
            return y ** 2 - 7

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solution = solver.integrate(
            exponential_growth, y0, t_eval, events=[ysq_eq_7, y_eq_9]
        )
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-4)
        np.testing.assert_array_less(solution.y, 9)
        np.testing.assert_array_less(solution.y ** 2, 7)
        self.assertEqual(solution.termination, "event")

    def test_ode_integrate_with_jacobian(self):
        # Linear
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def linear_ode(t, y):
            return np.array([0.5, 2 - y[0]])

        J = np.array([[0.0, 0.0], [-1.0, 0.0]])
        sJ = sparse.csr_matrix(J)

        def jacobian(t, y):
            return J

        def sparse_jacobian(t, y):
            return sJ

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=5e-3
        )

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=5e-3
        )
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=5e-3
        )

        solution = solver.integrate(linear_ode, y0, t_eval, jacobian=sparse_jacobian)

        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            2.0 * solution.t - 0.25 * solution.t ** 2, solution.y[1], rtol=5e-3
        )
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Nonlinear exponential growth
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_growth(t, y):
            return np.array([y[0], (1.0 - y[0]) * y[1]])

        def jacobian(t, y):
            return np.array([[1.0, 0.0], [-y[1], 1 - y[0]]])

        def sparse_jacobian(t, y):
            return sparse.csr_matrix(jacobian(t, y))

        y0 = np.array([1.0, 1.0])
        t_eval = np.linspace(0, 1, 100)

        solution = solver.integrate(exponential_growth, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-2)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-2
        )

        solution = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-2)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-2
        )

        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        solution = solver.integrate(exponential_growth, y0, t_eval, jacobian=jacobian)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-2)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-2
        )

        solution = solver.integrate(
            exponential_growth, y0, t_eval, jacobian=sparse_jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(np.exp(solution.t), solution.y[0], rtol=1e-2)
        np.testing.assert_allclose(
            np.exp(1 + solution.t - np.exp(solution.t)), solution.y[1], rtol=1e-2
        )

    def test_dae_integrate(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        y0 = np.array([0, 0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Exponential decay
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            exponential_decay_dae, y0, t_eval, mass_matrix=mass_matrix
        )
        np.testing.assert_allclose(
            solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-4
        )
        np.testing.assert_allclose(
            solution.y[1], 2 * np.exp(-0.1 * solution.t), rtol=1e-4
        )
        self.assertEqual(solution.termination, "final time")

    def test_dae_integrate_failure(self):
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        y0 = np.array([0, 1])
        t_eval = np.linspace(0, 1, 100)
        with self.assertRaises(pybamm.SolverError):
            solver.integrate(
                constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix
            )

    def test_dae_integrate_bad_ics(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        def constant_growth_dae_rhs(t, y):
            return np.array([constant_growth_dae(t, y, [0])[0]])

        def constant_growth_dae_algebraic(t, y):
            return np.array([constant_growth_dae(t, y, [0])[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        y0_guess = np.array([0, 1])
        t_eval = np.linspace(0, 1, 100)
        y0 = solver.calculate_consistent_initial_conditions(
            constant_growth_dae_rhs, constant_growth_dae_algebraic, y0_guess
        )
        # check y0
        np.testing.assert_array_equal(y0, [0, 0])
        # check dae solutions
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Exponential decay
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return [-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]]

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            exponential_decay_dae, y0, t_eval, mass_matrix=mass_matrix
        )
        np.testing.assert_allclose(
            solution.y[0], np.exp(-0.1 * solution.t), rtol=1e-4
        )
        np.testing.assert_allclose(
            solution.y[1], 2 * np.exp(-0.1 * solution.t), rtol=1e-4
        )

    def test_dae_integrate_with_event(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return [0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] - y[1]]

        def y0_eq_2(t, y):
            return y[0] - 2

        def y1_eq_5(t, y):
            return y[1] - 5

        y0 = np.array([0, 0])
        t_eval = np.linspace(0, 7, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, events=[y0_eq_2, y1_eq_5]
        )
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])
        np.testing.assert_array_less(solution.y[0], 2)
        np.testing.assert_array_less(solution.y[1], 5)
        self.assertEqual(solution.termination, "event")

        # Exponential decay
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def exponential_decay_dae(t, y, ydot):
            return np.array([-0.1 * y[0] - ydot[0], 2 * y[0] - y[1]])

        def y0_eq_0pt9(t, y):
            return y[0] - 0.9

        def t_eq_0pt5(t, y):
            return t - 0.5

        y0 = np.array([1, 2])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            exponential_decay_dae, y0, t_eval, events=[y0_eq_0pt9, t_eq_0pt5]
        )

        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_allclose(solution.y[1], 2 * np.exp(-0.1 * solution.t))
        np.testing.assert_array_less(0.9, solution.y[0])
        np.testing.assert_array_less(solution.t, 0.5)
        self.assertEqual(solution.termination, "event")

    def test_dae_integrate_with_jacobian(self):
        # Constant
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def constant_growth_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2.0 * y[0] - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [2.0, -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            constant_growth_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(1.0 * solution.t, solution.y[1])

        # Nonlinear (tests when Jacobian a function of y)
        solver = pybamm.DaecppDaeSolver(tol=1e-8)

        def nonlinear_dae(t, y, ydot):
            return np.array([0.5 * np.ones_like(y[0]) - ydot[0], 2 * y[0] ** 2 - y[1]])

        mass_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])

        def jacobian(t, y):
            return np.array([[0.0, 0.0], [4.0 * y[0], -1.0]])

        y0 = np.array([0.0, 0.0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate(
            nonlinear_dae, y0, t_eval, mass_matrix=mass_matrix, jacobian=jacobian
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])
        np.testing.assert_allclose(0.5 * solution.t ** 2, solution.y[1])

    def test_model_solver_ode(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y[0], np.exp(0.1 * solution.t), rtol=1e-4
        )

        # Test time
        self.assertGreater(
            solution.total_time, solution.solve_time + solution.set_up_time
        )

    def test_model_solver_ode_events(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=whole_cell)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = {
            "2 * var = 2.5": pybamm.min(2 * var - 2.5),
            "var = 1.5": pybamm.min(var - 1.5),
        }
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-9)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[0], 1.25)

    def test_model_solver_ode_jacobian(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh[0].npts

        # construct jacobian in order of model.rhs
        J = []
        for var in model.rhs.keys():
            if var.id == var1.id:
                J.append([np.eye(N), np.zeros((N, N))])
            else:
                J.append([-1.0 * np.eye(N), np.zeros((N, N))])

        J = np.block(J)

        def jacobian(t, y):
            return J

        model.jacobian = jacobian

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)

        T, Y = solution.t, solution.y
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(T, Y),
            np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(T, Y),
            np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
        )

    def test_model_solver_dae(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.use_jacobian = False
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

        # Test time
        self.assertGreater(
            solution.total_time, solution.solve_time + solution.set_up_time
        )

    def test_model_solver_dae_bad_ics(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 3}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_events(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = {
            "var1 = 1.5": pybamm.min(var1 - 1.5),
            "var2 = 2.5": pybamm.min(var2 - 2.5),
        }
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0], 1.5)
        np.testing.assert_array_less(solution.y[-1], 2.5)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_model_solver_dae_with_jacobian(self):
        # Create simple test model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1.0, var2: 2.0}
        model.initial_conditions_ydot = {var1: 0.1, var2: 0.2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        mesh = get_mesh_for_testing()
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh[0].npts

        def jacobian(t, y):
            return np.block(
                [
                    [0.1 * np.eye(N), np.zeros((N, N))],
                    [2.0 * np.eye(N), -1.0 * np.eye(N)],
                ]
            )

        model.jacobian = jacobian

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

    def test_solve_ode_model_with_dae_solver(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.DaecppDaeSolver(tol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y[0], np.exp(0.1 * solution.t), rtol=1e-4
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
