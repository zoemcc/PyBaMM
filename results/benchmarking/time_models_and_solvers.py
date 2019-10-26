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
    times = {npts: {} for npts in all_npts}
    t_eval = np.linspace(0, 0.6, 100)
    for npts in all_npts:
        # discretise
        pybamm.logger.info("Setting number of grid points to {}".format(npts))
        var_pts = {
            spatial_var: npts
            for spatial_var in [var.x_n, var.x_s, var.x_p, var.r_n, var.r_p]
        }
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        model_disc = disc.process_model(model, inplace=False)

        # solve
        for solver in solvers:
            # don't use ODE solver on DAE models
            if not (len(model.algebraic) > 0 and isinstance(solver, pybamm.OdeSolver)):
                solution = solver.solve(model_disc, t_eval)
                times[npts][solver.name] = solution.solve_time

    return times


def model_times_to_markdown_table(times):
    timer = pybamm.Timer()
    num_solvers = len(list(times.values())[0])
    table = ""
    table += (" {} |" * (num_solvers + 1) + "\n").format(
        "Grid points", *list(times.values())[0].keys()
    )
    table += "---|" * (num_solvers + 1) + "\n"
    for npts, solver_times in times.items():
        table += (" {} |" * (num_solvers + 1) + "\n").format(
            npts, *[timer.format(time) for time in solver_times.values()]
        )
    table += "\n"
    return table


def print_all_times_to_readme(li_ion_times, lead_acid_times):
    filename = "results/benchmarking/README.md"
    with open(filename, "w") as f:
        f.write("# Benchmarking\n\n")
        f.write(
            "Comparison of solvers (integrators) for different models with varying"
            + " mesh sizes. In all cases, each subdomain has the same number of"
            + " grid points.\n\n"
        )
        f.write("## Lithium-ion models\n\n")
        for model, times in li_ion_times.items():
            print("Solving {}".format(model.name))
            f.write("### {}\n\n".format(model.name))
            table = model_times_to_markdown_table(times)
            f.write(table)
        f.write("## Lead-acid models\n\n")
        for model, times in lead_acid_times.items():
            print("Solving {}".format(model.name))
            f.write("### {}\n\n".format(model.name))
            table = model_times_to_markdown_table(times)
            f.write(table)


if __name__ == "__main__":
    # pybamm.set_logging_level("DEBUG")
    li_ion_models = {pybamm.lithium_ion.SPM(): [100, 200, 400]}
    lead_acid_models = {
        pybamm.lead_acid.LOQS(): [1000, 2000, 4000],
        # pybamm.lead_acid.Full(): [10, 20, 40],
    }
    solvers = [
        pybamm.ScipySolver(),
        pybamm.ScikitsOdeSolver(),
        pybamm.ScikitsDaeSolver(),
    ]

    li_ion_times = {
        model: time_model_solvers(model, solvers, npts)
        for model, npts in li_ion_models.items()
    }
    lead_acid_times = {
        model: time_model_solvers(model, solvers, npts)
        for model, npts in lead_acid_models.items()
    }
    print_all_times_to_readme(li_ion_times, lead_acid_times)
