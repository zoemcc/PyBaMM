#
# Circuit element for a linear resistor
#

import pybamm
from .shared import get_standard_current_collector_potential_variables


class LinearResistor(pybamm.BaseSubModel):
    """A class to represent a linear resistor element.

    Parameters
    ----------
    param : parameter class
        The parameters to use in this circuit element.

    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        i_through_cell = variables["Current collector current density"]

        R_0 = self.param.R_0

        phi_s_cp += -R_0 * i_through_cell

        variables.update(
            get_standard_current_collector_potential_variables(
                self.param, phi_s_cn, phi_s_cp
            )
        )
        # Note: "Positive current collector potential" will be updated by other
        # circuit elements
        return variables
