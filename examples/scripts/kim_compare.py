import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# options = {"thermal": "x-full"}
data_1C = pd.read_csv("kim_1C.csv", header=None)
data_2C = pd.read_csv("kim_2C.csv", header=None)
data_4C = pd.read_csv("kim_4C.csv", header=None)
data = [data_1C, data_2C, data_4C]

colors = ["r", "g", "b"]

solutions = [None] * 3

dc = [None] * 3
voltage = [None] * 3

t_final = 1e6

L_y = 0.14
L_z = 0.2
A_cc = L_y * L_z


kim_set = {
    "Cell capacity [A.h]": 20,
    "Typical current [A]": 1,
    "Electrode height [m]": 0.2,
    "Electrode width [m]": 0.14,
    "Positive tab width [m]": 0.044000000000000004,
    "Negative tab width [m]": 0.044000000000000004,
    "Negative tab centre y-coordinate [m]": 0.013000000000000001,
    "Negative tab centre z-coordinate [m]": 0.2,
    "Positive tab centre y-coordinate [m]": 0.13699999999999998,
    "Positive tab centre z-coordinate [m]": 0.2,
    "Upper voltage cut-off [V]": 4.5,
    "Lower voltage cut-off [V]": 2.5,
    "Heat transfer coefficient [W.m-2.K-1]": 0,  # 0.260,
    "Cation transference number": 0.5,  # makes no difference to heat gen
}

chemistry = pybamm.parameter_sets.NCA_Kim2011
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
# parameter_values.update(kim_set)

c_rates = [1, 2, 4]
c_rates = [1]

for i, c_rate in enumerate(c_rates):
    model = pybamm.lithium_ion.SPMe()
    sim = pybamm.Simulation(model, parameter_values=parameter_values, C_rate=c_rate)

    sim.solve()

    solutions[i] = sim.solution

    dc[i] = sim.solution["Discharge capacity [A.h]"]
    voltage[i] = sim.solution["Terminal voltage [V]"]

    t = np.linspace(0, solutions[i].t[-1], 100)
    plt.plot(dc[i](t), voltage[i](t), color=colors[i])
    plt.plot(data[i][0] * 48 * A_cc, data[i][1], color="k", linestyle=":")


plt.legend(["1C", "", "2C", "", "3C"])
plt.xlabel("Discharge capacity [A.h]")
plt.ylabel("Terminal voltage [V]")
plt.show()

sols = [sol for sol in solutions if sol is not None]
labels = [
    label for i, label in enumerate(["1C", "2C", "4C"]) if solutions[i] is not None
]
linestyles = [
    label for i, label in enumerate(["-", "--", ":"]) if solutions[i] is not None
]

plot = pybamm.QuickPlot(
    sols,
    output_variables=[
        "Electrolyte concentration [mol.m-3]",
        "Electrolyte potential [V]",
        "Terminal voltage [V]",
    ],
    labels=labels,
    linestyles=linestyles,
)

plot.dynamic_plot()
