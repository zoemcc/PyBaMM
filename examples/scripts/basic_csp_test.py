import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")
# pybamm.settings.debug_mode = True

C_rate = 5

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)

cSP = pybamm.lithium_ion.BasicSPMe(linear_diffusion=False, use_log=True)
solver = pybamm.CasadiSolver(mode="fast")
sim_cSP = pybamm.Simulation(
    cSP, parameter_values=parameter_values, C_rate=C_rate, solver=solver
)
sim_cSP.solve()

spme = pybamm.lithium_ion.SPMe()
solver = pybamm.CasadiSolver(mode="fast")
sim_spme = pybamm.Simulation(
    spme, parameter_values=parameter_values, C_rate=C_rate, solver=solver
)
sim_spme.solve()

# dfn = pybamm.lithium_ion.DFN()
# sim_DFN = pybamm.Simulation(dfn, parameter_values=parameter_values, C_rate=C_rate)
# sim_DFN.solve()

dc = sim_spme.solution["Discharge capacity [A.h]"](sim_spme.solution.t)

spme_voltage = sim_spme.solution["Terminal voltage [V]"](sim_spme.solution.t)
basic_spme_voltage = sim_cSP.solution["Terminal voltage [V]"](sim_spme.solution.t)

spme_potential = sim_spme.solution["Electrolyte potential [V]"](
    sim_spme.solution.t, x=np.linspace(0, 1, 100)
)
basic_spme_potential = sim_cSP.solution["Electrolyte potential [V]"](
    sim_spme.solution.t, x=np.linspace(0, 1, 100)
)

rmse = pybamm.rmse(spme_voltage, basic_spme_voltage)
print(rmse * 1e3)

error = np.abs(spme_voltage - basic_spme_voltage)

potential_error = np.max(np.abs(spme_potential - basic_spme_potential), axis=1)

plt.semilogy(dc, potential_error)
plt.title("SPMe -- Basic SPMe electrolyte potential difference")
plt.xlabel("Discharge capacity [A.h]")
plt.ylabel("Error [V]")
plt.show()

plt.semilogy(dc, error)
plt.title("SPMe -- Basic SPMe voltage difference")
plt.xlabel("Discharge capacity [A.h]")
plt.ylabel("Error [V]")
plt.show()


variables = [
    "Electrolyte concentration [mol.m-3]",
    # "Electrolyte current density",
    "Terminal voltage [V]",
    "Real terminal voltage [V]",
    "Electrolyte potential [V]",
    # "Negative electrode potential [V]",
    # "Positive electrode potential [V]",
]

# plot = pybamm.QuickPlot(
#     # [sim_DFN.solution, sim_cSP.solution, sim_spme.solution],
#     [sim_cSP.solution, sim_spme.solution],
#     output_variables=variables,
#     labels=["DFN", "cSP", "SPMe"],
#     linestyles=["-", "--", ":"],
#     colors=["k", "r", "b"],
# )
plot = pybamm.QuickPlot(
    # [sim_DFN.solution, sim_cSP.solution, sim_spme.solution],
    [sim_spme.solution, sim_cSP.solution],
    output_variables=variables,
    labels=["SPMe", "Basic SPMe"],
    linestyles=["-", ":"],
    colors=["k", "r"],
)
plot.dynamic_plot()
