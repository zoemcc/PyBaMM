import pybamm
import numpy as np
import pickle
import matplotlib.pyplot as plt


pybamm.set_logging_level("INFO")


run = False
# filename = "compare_electrolyte.p"
filename = "compare_particles.p"

if run is False:
    sol = pickle.load(open(filename, "rb"))

    for i, pts in enumerate(sol["pts"]):
        print("The error at ", pts, "pts is ", sol["errors"][i])

    # plt.loglog(sol["pts"], sol["errors"])
    plt.plot(sol["pts"], sol["errors"])
    # plt.loglog(
    #     sol["pts"],
    #     (sol["pts"][0] ** 2 / sol["errors"][0]) / sol["pts"] ** 2,
    #     linestyle="--",
    # )
    # plt.loglog(sol["pts"], 1 / sol["pts"] ** 2, linestyle=":")
    plt.show()


def pts(n):
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 200,
        var.x_s: 200,
        var.x_p: 200,
        var.r_n: n,
        var.r_p: n,
    }
    return var_pts


def pts_all(n):
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: n,
        var.x_s: n,
        var.x_p: n,
        var.r_n: n,
        var.r_p: n,
    }
    return var_pts


if run is True:
    t_eval = np.linspace(0, 3300, 100)
    solver = pybamm.CasadiSolver(mode="fast")

    model_truth = pybamm.lithium_ion.DFN()
    sim_truth = pybamm.Simulation(model_truth, var_pts=pts_all(200), solver=solver)
    sim_truth.solve()

    t_max = sim_truth.solution.t[-1]
    t = np.linspace(0, t_max, 100)
    truth_voltage = sim_truth.solution["Terminal voltage [V]"](t)

    # other_pts = np.array([3, 5, 10, 30, 40])
    other_pts = np.arange(3, 70, 5)
    solutions = np.zeros_like(other_pts)
    errors = np.linspace(0, 1, len(other_pts))

    for i, p in enumerate(other_pts):

        model = pybamm.lithium_ion.DFN()
        solver = pybamm.CasadiSolver(mode="fast")
        sim = pybamm.Simulation(model, var_pts=pts(p), solver=solver)
        sim.solve()
        v = sim.solution["Terminal voltage [V]"](t)
        errors[i] = np.sum((v - truth_voltage) ** 2) ** (1 / 2)

    out = {"pts": other_pts, "errors": errors}
    pickle.dump(out, open(filename, "wb"))

    plt.loglog(other_pts, errors)
    plt.show()

