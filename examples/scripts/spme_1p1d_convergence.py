import pybamm
import numpy as np
import pickle
import matplotlib.pyplot as plt


def pts(n):
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: n,
        var.x_s: n,
        var.x_p: n,
        var.r_n: n,
        var.r_p: n,
        var.y: n,
        var.z: n,
    }
    return var_pts


t_eval = np.linspace(0, 3300, 100)
solver = pybamm.CasadiSolver(mode="fast")

options = {"current collector": "potential pair", "dimensionality": 1}

model_truth = pybamm.lithium_ion.DFN(options)
sim_truth = pybamm.Simulation(model_truth, var_pts=pts(3), solver=solver)
sim_truth.solve()

t_max = sim_truth.solution.t[-1]
t = np.linspace(0, t_max, 100)
truth_voltage = sim_truth.solution["Terminal voltage [V]"](t)

other_pts = np.array([3, 5, 10, 20, 30])
solutions = np.zeros_like(other_pts)
errors = np.linspace(0, 1, len(other_pts))

for i, p in enumerate(other_pts):

    model = pybamm.lithium_ion.DFN(options)
    solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(model, var_pts=pts(p), solver=solver)
    sim.solve()
    v = sim.solution["Terminal voltage [V]"](t)

    model = pybamm.lithium_ion.SPMe(options)
    solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(model, var_pts=pts(p), solver=solver)
    sim.solve()
    v_spm = sim.solution["Terminal voltage [V]"](t)

    errors[i] = np.sum((v - v_spm) ** 2) ** (1 / 2)

plt.plot(other_pts, errors)
plt.show()

out = {"pts": other_pts, "errors": errors}
pickle.dump(out, open("convergence_spme.p", "wb"))
