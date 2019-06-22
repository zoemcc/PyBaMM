#include <functional>

#include "solver.h"

class PybammMassMatrix : public daecpp::MassMatrix {
  public:
  using function_type = std::function<void(daecpp::sparse_matrix_holder&)>;

  PybammMassMatrix(const function_type& f)
      : m_f(f)
  {
  }

  void operator()(daecpp::sparse_matrix_holder& M) { m_f(M); }

  private:
  function_type m_f;
};

class PybammRHS : public daecpp::RHS {
  public:
  using function_type = std::function<void(
      const daecpp::state_type&, daecpp::state_type&, const double)>;
  PybammRHS(const function_type& f)
      : m_f(f)
  {
  }
  void operator()(const daecpp::state_type& x, daecpp::state_type& f, const double t)
  {
    m_f(x, f, t);
  }

  private:
  function_type m_f;
};

class PybammJacobian : public daecpp::Jacobian {
  public:
  using function_type = std::function<void(
      daecpp::sparse_matrix_holder&, const daecpp::state_type&, const double)>;

  PybammJacobian(PybammRHS& rhs, const function_type& f)
      : Jacobian(rhs)
      , m_f(f)
  {
  }

  void operator()(
      daecpp::sparse_matrix_holder& J, const daecpp::state_type& x, const double t)
  {
    m_f(J, x, t);
  }

  private:
  function_type m_f;
};

class PybammSolver : public daecpp::Solver {
  public:
  PybammSolver(daecpp::RHS& rhs, daecpp::Jacobian& jac, daecpp::MassMatrix& mass,
      daecpp::SolverOptions& opt)
      : Solver(rhs, jac, mass, opt)
  {
  }

  void observer(daecpp::state_type& x, const double t)
  {
    m_x_axis.push_back(t);
    m_x.push_back(x);
  }

  daecpp::state_type_matrix m_x;
  daecpp::state_type m_x_axis;
};

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(daecpp::state_type);
PYBIND11_MAKE_OPAQUE(daecpp::vector_type_int);
PYBIND11_MAKE_OPAQUE(daecpp::state_type_matrix);

PYBIND11_MODULE(pydae, m)
{
  py::bind_vector<daecpp::state_type>(m, "state_type");
  py::bind_vector<daecpp::vector_type_int>(m, "vector_type_int");
  py::bind_vector<daecpp::state_type_matrix>(m, "state_type_matrix");

  py::class_<daecpp::sparse_matrix_holder>(m, "sparse_matrix_holder");
  //.def(py::init<const daecpp::state_type&, const daecpp::vector_type_int&,
  //    const daecpp::state_type_matrix&>());

  py::class_<daecpp::SolverOptions>(m, "SolverOptions").def(py::init<>())
    .def_readwrite("fact_every_iter", daecpp::SolverOptions::fact_every_iter)
    .def_readwrite("time_stepping", daecpp::SolverOptions::time_stepping)
    .def_readwrite("max_Newton_iter", daecpp::SolverOptions::max_Newton_iter)
    .def_readwrite("atol", daecpp::SolverOptions::atol)
    .def_readwrite("dt_eps_m", daecpp::SolverOptions::dt_eps_m)
    .def_readwrite("value_max", daecpp::SolverOptions::value_max)
    .def_readwrite("dt_init", daecpp::SolverOptions::dt_init)

double t0 = 0.0;
    double dt_min = dt_eps_m;
    double dt_max = 100.0;
    int verbosity = 1;
    int dt_increase_threshold = 4;    // Time step amplification threshold
    int dt_decrease_threshold = 8;    // Time step reduction threshold
    double dt_increase_factor = 2.0;  // Time step amplification factor
    double dt_decrease_factor = 2.0;  // Time step reduction factor
    double dt_eta_min = 0.05;  // Monitor function lower threshold (A-SATS only)
    double dt_eta_max = 0.5;  // Monitor function higher threshold (A-SATS only)
    MKL_INT preconditioned_CGS = 0;
    MKL_INT refinement_steps = 2;
    MKL_INT parallel_fact_control = 0;
    MKL_INT parallel_solve_control = 0;



  py::class_<PybammMassMatrix>(m, "MassMatrix")
      .def(py::init<const PybammMassMatrix::function_type&>());
  py::class_<PybammRHS>(m, "RHS").def(py::init<const PybammRHS::function_type&>());
  py::class_<PybammJacobian>(m, "AnalyticalJacobian")
      .def(py::init<PybammRHS&, const PybammJacobian::function_type&>());
  py::class_<daecpp::Jacobian>(m, "NumericalJacobian").def(py::init<PybammRHS&>());
  py::class_<PybammSolver>(m, "Solver")
      .def(py::init<daecpp::RHS&, daecpp::Jacobian&, daecpp::MassMatrix&,
          daecpp::SolverOptions&>());
}
