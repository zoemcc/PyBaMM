import pybamm

pybamm.set_logging_level("INFO")
# pybamm.settings.debug_mode = True

cSP = pybamm.lithium_ion.BasicCSP()
sim_cSP = pybamm.Simulation(cSP)
sim_cSP.solve()

spme = pybamm.lithium_ion.SPMe()
sim_spme = pybamm.Simulation(spme)
sim_spme.solve()

dfn = pybamm.lithium_ion.DFN()
sim_DFN = pybamm.Simulation(dfn)
sim_DFN.solve()

variables = [
    "Electrolyte concentration",
    "Electrolyte current density",
    "Terminal voltage [V]",
    "Electrolyte potential [V]",
    # "Negative electrode potential [V]",
]

plot = pybamm.QuickPlot(
    [sim_DFN.solution, sim_cSP.solution, sim_spme.solution],
    output_variables=variables,
    labels=["DFN", "cSP", "SPMe"],
    linestyles=["-", "--", ":"],
)
plot.dynamic_plot()
