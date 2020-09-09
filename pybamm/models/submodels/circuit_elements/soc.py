#
# Circuit element to track the SoC
#

import pybamm


class SoC(pybamm.BaseSubModel):
    """A class to track the SoC of the cell.

    Parameters
    ----------
    param : parameter class
        The parameters to use in this circuit element.

    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        SoC = pybamm.Variable("SoC", domain="current collector", bounds=(0, 1))
        variables = {"SoC": SoC, "Av SoC": SoC - pybamm.z_average(SoC)}
        return variables

    def set_rhs(self, variables):
        # d SoC / dt = I_through_cell [A.m-2]
        #               * Coulombic efficiency
        #               / Nominal capacity per unit area of cell [A.h.m-2]

        SoC = variables["SoC"]
        i_through_cell = variables["Current collector current density"]
        eta = self.param.eta  # Coulombic efficiency
        C_nom = self.param.C_nom  # Nominal capacity per unit area (dimensionless)

        self.rhs[SoC] = -i_through_cell * eta / C_nom * 0.001

        return variables

    def set_initial_conditions(self, variables):
        SoC = variables["SoC"]
        self.initial_conditions[SoC] = self.param.SoC_init
        return variables

    def set_events(self, variables):
        SoC = variables["SoC"]
        tol = 1e-4
        self.events.append(
            pybamm.Event(
                "Minumum SoC",
                pybamm.min(SoC) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum SoC",
                (1 + tol) - pybamm.max(SoC),
                pybamm.EventType.TERMINATION,
            )
        )
