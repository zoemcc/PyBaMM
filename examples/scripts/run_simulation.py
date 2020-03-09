import pybamm

model = pybamm.BaseBatteryModel()
v = pybamm.Variable("v")
model.rhs = {v: -1}
model.initial_conditions = {v: 1}
model.variables = {
    "v": v,
    "2v": 2 * v,
    "3v": 3 * v,
    "4v": 4 * v,
    "5v": 5 * v,
    "6v": 6 * v,
    "7v": 7 * v,
    "8v": 8 * v,
}

sim = pybamm.Simulation(model)
sim.interactive(
    {"Current function [A]": 0.5},
    quick_plot_vars=["v", "2v", "3v", "4v", "5v", "6v", "7v", "8v"],
)
# sim.solve()
# sim.plot()
