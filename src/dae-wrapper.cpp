#include <functional>

#include "solver.h"

#include <iostream>

class PybammMassMatrix : public daecpp::MassMatrix {
  public:
  using function_type = std::function<void(daecpp::sparse_matrix_holder*)>;

  PybammMassMatrix(const function_type& f)
      : m_f(f)
  {
  }

  void operator()(daecpp::sparse_matrix_holder& M) { m_f(&M); }

  private:
  function_type m_f;
};

class PybammRHS : public daecpp::RHS {
  public:
  using function_type = std::function<void(
      const daecpp::state_type*, daecpp::state_type*, const double)>;
  using stop_type = std::function<bool(const daecpp::state_type*, const double)>;

  PybammRHS(const function_type& f)
      : m_f(f)
  {
  }

  void set_stop_condition(const stop_type& f) { m_stop_f = f; }

  void operator()(const daecpp::state_type& x, daecpp::state_type& f, const double t)
  {
    m_f(&x, &f, t);
  }

  bool stop_condition(const daecpp::state_type& x, const double t)
  {
    if (m_stop_f) {
      return m_stop_f(&x, t);
    } else {
      return false;
    }
  }

  private:
  function_type m_f;
  stop_type m_stop_f;
};

class PybammJacobian : public daecpp::Jacobian {
  public:
  using function_type = std::function<void(
      daecpp::sparse_matrix_holder*, const daecpp::state_type*, const double)>;

  PybammJacobian(PybammRHS& rhs, const function_type& f)
      : Jacobian(rhs)
      , m_f(f)
  {
  }

  void operator()(
      daecpp::sparse_matrix_holder& J, const daecpp::state_type& x, const double t)
  {
    m_f(&J, &x, t);
  }

  private:
  function_type m_f;
};

class PybammSolver : public daecpp::Solver {
  public:
  PybammSolver(daecpp::RHS& rhs, daecpp::Jacobian& jac, daecpp::MassMatrix& mass,
      daecpp::SolverOptions& opt)
      : Solver(rhs, jac, mass, opt)
      , m_observe(false)
  {
  }

  void observer(daecpp::state_type& x, const double t)
  {
    if (m_observe) {
      m_x_axis.push_back(t);
      m_x.push_back(x);
    }
  }

  daecpp::state_type_matrix m_x;
  daecpp::state_type m_x_axis;
  bool m_observe;
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
  py::bind_vector<daecpp::state_type>(m, "state_type")
      .def("resize",
          (void (daecpp::state_type::*)(size_t)) & daecpp::state_type::resize);
  py::bind_vector<daecpp::vector_type_int>(m, "vector_type_int")
      .def("resize",
          (void (daecpp::vector_type_int::*)(size_t))
              & daecpp::vector_type_int::resize);
  py::bind_vector<daecpp::state_type_matrix>(m, "state_type_matrix");

  py::class_<daecpp::sparse_matrix_holder>(m, "sparse_matrix_holder")
      .def(py::init<>())
      .def_readwrite("A", &daecpp::sparse_matrix_holder::A)
      .def_readwrite("ja", &daecpp::sparse_matrix_holder::ja)
      .def_readwrite("ia", &daecpp::sparse_matrix_holder::ia);
  //.def(py::init<const daecpp::state_type&, const daecpp::vector_type_int&,
  //    const daecpp::state_type_matrix&>());

  py::class_<daecpp::SolverOptions>(m, "SolverOptions")
      .def(py::init<>())
      .def_readwrite("fact_every_iter", &daecpp::SolverOptions::fact_every_iter)
      .def_readwrite("bdf_order", &daecpp::SolverOptions::bdf_order)
      .def_readwrite("time_stepping", &daecpp::SolverOptions::time_stepping)
      .def_readwrite("max_Newton_iter", &daecpp::SolverOptions::max_Newton_iter)
      .def_readwrite("atol", &daecpp::SolverOptions::atol)
      .def_readwrite("dt_eps_m", &daecpp::SolverOptions::dt_eps_m)
      .def_readwrite("value_max", &daecpp::SolverOptions::value_max)
      .def_readwrite("dt_init", &daecpp::SolverOptions::dt_init)
      .def_readwrite("t0", &daecpp::SolverOptions::t0)
      .def_readwrite("dt_min", &daecpp::SolverOptions::dt_min)
      .def_readwrite("dt_max", &daecpp::SolverOptions::dt_max)
      .def_readwrite("verbosity", &daecpp::SolverOptions::verbosity)
      .def_readwrite(
          "dt_increase_threshold", &daecpp::SolverOptions::dt_increase_threshold)
      .def_readwrite(
          "dt_decrease_threshold", &daecpp::SolverOptions::dt_decrease_threshold)
      .def_readwrite("dt_increase_factor", &daecpp::SolverOptions::dt_increase_factor)
      .def_readwrite("dt_decrease_factor", &daecpp::SolverOptions::dt_decrease_factor)
      .def_readwrite("dt_eta_min", &daecpp::SolverOptions::dt_eta_min)
      .def_readwrite("dt_eta_max", &daecpp::SolverOptions::dt_eta_max)
      .def_readwrite("preconditioned_CGS", &daecpp::SolverOptions::preconditioned_CGS)
      .def_readwrite("refinement_steps", &daecpp::SolverOptions::refinement_steps)
      .def_readwrite(
          "parallel_fact_control", &daecpp::SolverOptions::parallel_fact_control)
      .def_readwrite(
          "parallel_solve_control", &daecpp::SolverOptions::parallel_solve_control);

  py::class_<daecpp::MassMatrix>(m, "BaseMassMatrix");
  py::class_<PybammMassMatrix, daecpp::MassMatrix>(m, "MassMatrix")
      .def(py::init<const PybammMassMatrix::function_type&>())
      .def("__call__", &PybammMassMatrix::operator());
  py::class_<daecpp::RHS>(m, "BaseRHS");
  py::class_<PybammRHS, daecpp::RHS>(m, "RHS")
      .def(py::init<const PybammRHS::function_type&>())
      .def("set_stop_condition", &PybammRHS::set_stop_condition)
      .def("__call__", &PybammRHS::operator());
  py::class_<daecpp::Jacobian>(m, "NumericalJacobian")
      .def(py::init<PybammRHS&>())
      .def(py::init<PybammRHS&, const double>())
      .def("__call__", &daecpp::Jacobian::operator());
  py::class_<PybammJacobian, daecpp::Jacobian>(m, "AnalyticalJacobian")
      .def(py::init<PybammRHS&, const PybammJacobian::function_type&>())
      .def("__call__", &PybammJacobian::operator());
  py::class_<daecpp::Solver>(m, "BaseSolver");
  py::class_<PybammSolver, daecpp::Solver>(m, "Solver")
      .def(py::init<daecpp::RHS&, daecpp::Jacobian&, PybammMassMatrix&,
          daecpp::SolverOptions&>())
      .def("__call__", &PybammSolver::operator());
}
