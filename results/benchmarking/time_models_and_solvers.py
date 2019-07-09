#
# Time models with each solver
#
import pybamm
import numpy as np


def time_model_solvers(model, solvers, all_npts):
    # load parameter values and geometry
    geometry = model.default_geometry
    param = model.default_parameter_values

    # Process parameters
    param.process_model(model)
    param.process_geometry(geometry)

    # Calculate time for each solver and each number of grid points
    var = pybamm.standard_spatial_vars
    times = {solver: {} for solver in solvers}
    t_eval = np.linspace(0, 1, 100)
    for npts in all_npts:
        # discretise
        pybamm.logger.info("Setting number of grid points to {}".format(npts))
        var_pts = {
            spatial_var: npts
            for spatial_var in [var.x_n, var.x_s, var.x_p, var.r_n, var.r_p]
        }
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve
        for solver in solvers:
            solution = solver.solve(model, t_eval)
            times[solver][npts] = solution.solve_time

    return times


def print_model_times_to_markdown_table(times):
    pass


def print_all_times_to_readme(all_times):
    pass


if __name__ == "__main__":
    li_ion_models = [pybamm.lead_acid.LOQS()]
    solvers = [
        pybamm.ScipySolver(),
        pybamm.ScikitsOdeSolver(),
        # pybamm.ScikitsDaeSolver(),
    ]
    all_npts = [10, 20, 40]

    all_times = {
        model: time_model_solvers(model, solvers, all_npts) for model in li_ion_models
    }
