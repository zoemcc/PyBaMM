import pybamm as pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pybamm.set_logging_level("INFO")
# load data
voltage_data_1C = pd.read_csv("ddliion/voltage_1C.dat", sep="\t")
voltage_data_2_5C = pd.read_csv("ddliion/voltage_2_5C.dat", sep="\t")
voltage_data_5C = pd.read_csv("ddliion/voltage_5C.dat", sep="\t")
voltage_data_7_5C = pd.read_csv("ddliion/voltage_7_5C.dat", sep="\t")

# set up models
models = {
    "SPM": pybamm.lithium_ion.SPM(),
    "SPMe": pybamm.lithium_ion.SPMe(),
    "DFN": pybamm.lithium_ion.DFN(),
}

# pick parameters, keeping C-rate as an input to be changed for each solve
chemistry = pybamm.parameter_sets.Ecker2015
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
parameter_values.update({"C-rate": "[input]"})

# set up number of points for discretisation
var = pybamm.standard_spatial_vars
# var_pts = {var.x_n: 101, var.x_s: 101, var.x_p: 101, var.r_n: 101, var.r_p: 101}
var_pts = {
    var.x_n: int(parameter_values.evaluate(pybamm.geometric_parameters.L_n / 1e-6)),
    var.x_s: int(parameter_values.evaluate(pybamm.geometric_parameters.L_s / 1e-6)),
    var.x_p: int(parameter_values.evaluate(pybamm.geometric_parameters.L_p / 1e-6)),
    # var.r_n: int(parameter_values.evaluate(pybamm.geometric_parameters.R_n / 1e-7)),
    # var.r_p: int(parameter_values.evaluate(pybamm.geometric_parameters.R_p / 1e-7)),
    var.r_n: 250,
    var.r_p: 250,
}

# set up simulations
sims = {}
for name, model in models.items():
    sims[name] = pybamm.Simulation(
        model, parameter_values=parameter_values, var_pts=var_pts
    )

# pick C_rates and times to integrate over (using casasi fast mode, so want to
# stop before e.g. surface concentration goes negative)
C_rates = [1, 2.5, 5, 7.5]
t_evals = [
    np.linspace(0, 3830, 1000),
    np.linspace(0, 1520, 1000),
    np.linspace(0, 730, 1000),
    np.linspace(0, 450, 1000),
]
# t_evals = [
#    np.array(voltage_data_1C["t(s)"]),
#    np.array(voltage_data_2_5C["t(s)"]),
#    np.array(voltage_data_5C["t(s)"]),
#    np.array(voltage_data_7_5C["t(s)"]),
# ]

# loop over C-rates
solutions = {
    "SPM": [None] * len(C_rates),
    "SPMe": [None] * len(C_rates),
    "DFN": [None] * len(C_rates),
}
for i, C_rate in enumerate(C_rates):
    print("C-rate = {}".format(C_rate))

    # solve models
    t_eval = t_evals[i]

    for name, sim in sims.items():
        print("Solving {}...".format(name))
        sim.solve(
            t_eval=t_eval,
            solver=pybamm.CasadiSolver(mode="fast"),
            inputs={"C-rate": C_rate},
        )
        solutions[name] = sim.solution
print("Finished")

# plot
fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85, wspace=0.3, hspace=0.5)
# 1C
for key in solutions.keys():
    t = solutions[key][0]["Time [s]"](solutions[key][0].t)
    V = solutions[key][0]["Terminal voltage [V]"](solutions[key][0].t)
    ax[0, 0].plot(t, V, label=key + "(PyBaMM)")
ax[0, 0].plot(
    voltage_data_1C["t(s)"], voltage_data_1C["Voltage(V)"], label="DFN (Dandeliion)"
)
ax[0, 0].set_xlabel("Time [s]")
ax[0, 0].set_ylabel("Voltage [V]")
ax[0, 0].set_xlim([0, 4000])
ax[0, 0].set_ylim([2.4, 4.2])
ax[0, 0].grid()
ax[0, 0].legend()
ax[0, 0].title.set_text("1C")
# 2.5C
for key in solutions.keys():
    t = solutions[key][1]["Time [s]"](solutions[key][1].t)
    V = solutions[key][1]["Terminal voltage [V]"](solutions[key][1].t)
    ax[0, 1].plot(t, V, label=key + "(PyBaMM)")
ax[0, 1].plot(
    voltage_data_2_5C["t(s)"], voltage_data_2_5C["Voltage(V)"], label="DFN (Dandeliion)"
)
ax[0, 1].set_xlabel("Time [s]")
ax[0, 1].set_ylabel("Voltage [V]")
ax[0, 1].set_xlim([0, 1600])
ax[0, 1].set_ylim([2.4, 4.2])
ax[0, 1].grid()
ax[0, 1].legend()
ax[0, 1].title.set_text("2.5C")
# 5C
for key in solutions.keys():
    t = solutions[key][2]["Time [s]"](solutions[key][2].t)
    V = solutions[key][2]["Terminal voltage [V]"](solutions[key][2].t)
    ax[1, 0].plot(t, V, label=key + "(PyBaMM)")
ax[1, 0].plot(
    voltage_data_5C["t(s)"], voltage_data_5C["Voltage(V)"], label="DFN (Dandeliion)"
)
ax[1, 0].set_xlabel("Time [s]")
ax[1, 0].set_ylabel("Voltage [V]")
ax[1, 0].set_xlim([0, 800])
ax[1, 0].set_ylim([2.4, 4.2])
ax[1, 0].grid()
ax[1, 0].legend()
ax[1, 0].title.set_text("5C")
# 7.5C
for key in solutions.keys():
    t = solutions[key][3]["Time [s]"](solutions[key][3].t)
    V = solutions[key][3]["Terminal voltage [V]"](solutions[key][3].t)
    ax[1, 1].plot(t, V, label=key + "(PyBaMM)")
ax[1, 1].plot(
    voltage_data_7_5C["t(s)"], voltage_data_7_5C["Voltage(V)"], label="DFN (Dandeliion)"
)
ax[1, 1].set_xlabel("Time [s]")
ax[1, 1].set_ylabel("Voltage [V]")
ax[1, 1].set_xlim([0, 500])
ax[1, 1].set_ylim([2.4, 4.2])
ax[1, 1].grid()
ax[1, 1].legend()
ax[1, 1].title.set_text("7.5C")

plt.show()
