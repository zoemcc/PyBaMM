import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pb.set_logging_level("INFO")

# pick model and parameters
model = pb.lithium_ion.DFN()
model.convert_to_format = "python"
chemistry = pb.parameter_sets.Ecker2015
# chemistry = pb.parameter_sets.Chen2020
# chemistry = pb.parameter_sets.Marquis2019
parameter_values = pb.ParameterValues(chemistry=chemistry)
parameter_values.update({"C-rate": 7.5})

# set up simulation
var = pb.standard_spatial_vars
var_pts = {var.x_n: 75, var.x_s: 21, var.x_p: 55, var.r_n: 51, var.r_p: 51}
sim = pb.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)

# solve
t_eval = np.linspace(0, 450, 100)
sim.solve(t_eval=t_eval, solver=pb.CasadiSolver(mode="fast"))

# load in dandeliion data
voltage_data = pd.read_csv("ddliion/voltage.dat", sep="\t")
# t_eval = np.array(voltage_data["t(s)"])

# plot
V = sim.solution["Terminal voltage [V]"]
dc = sim.solution["Discharge capacity [A.h]"]
time = sim.solution["Time [s]"]
plt.figure()
plt.plot(dc(sim.solution.t), V(sim.solution.t), label="PyBaMM")
plt.plot(voltage_data["Capacity(Ah)"], voltage_data["Voltage(V)"], label="Dandeliion")
plt.xlabel("Dicharge capacity [A.h]")
plt.ylabel("Voltage [V]")
plt.ylim([2.4, 4.2])
plt.legend()
# plt.show()

plt.figure()
plt.plot(time(sim.solution.t), V(sim.solution.t), label="PyBaMM")
plt.plot(
    voltage_data["Capacity(Ah)"] * 3600 / parameter_values["Current function [A]"],
    voltage_data["Voltage(V)"],
    label="Dandeliion",
)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.ylim([2.4, 4.2])
plt.legend()
plt.show()
