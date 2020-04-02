import pybamm

pybamm.set_logging_level("INFO")
# pybamm.settings.debug_mode = True

C_rate = 5

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)

cSP = pybamm.lithium_ion.BasicCSP()
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

dfn = pybamm.lithium_ion.DFN()
sim_DFN = pybamm.Simulation(dfn, parameter_values=parameter_values, C_rate=C_rate)
sim_DFN.solve()

variables = [
    "Electrolyte concentration",
    "Electrolyte current density",
    "Terminal voltage [V]",
    "Electrolyte potential [V]",
    "Negative electrode potential [V]",
    "Positive electrode potential [V]",
]

plot = pybamm.QuickPlot(
    [sim_DFN.solution, sim_cSP.solution, sim_spme.solution],
    output_variables=variables,
    labels=["DFN", "cSP", "SPMe"],
    linestyles=["-", "--", ":"],
    colors=["k", "r", "b"],
)
plot.dynamic_plot()
