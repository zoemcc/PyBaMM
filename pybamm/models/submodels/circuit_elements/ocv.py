#
# Circuit element for the OCV
#

import pybamm
from .shared import get_standard_current_collector_potential_variables


class OCV(pybamm.BaseSubModel):
    """A class to represent the OCV circuit element.

    Parameters
    ----------
    param : parameter class
        The parameters to use in this circuit element.

    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):

        pot_scale = self.param.potential_scale

        T = self.param.T_ref

        SoC = variables["SoC"]
        OCV = self.param.OCV(SoC, T)
        OCV_dim = self.param.OCV_ref + OCV * pot_scale

        # For coupling to current collectors
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = phi_s_cn + OCV

        variables.update(
            {
                "Through-cell open-circuit voltage": OCV,
                "Through-cell open-circuit voltage [V]": OCV_dim,
            }
        )

        variables.update(
            get_standard_current_collector_potential_variables(
                self.param, phi_s_cn, phi_s_cp
            )
        )
        # Note: "Positive current collector potential" will be updated by other
        # circuit elements
        return variables
