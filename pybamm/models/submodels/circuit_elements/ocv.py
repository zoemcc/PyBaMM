#
# Circuit element for the OCV
#

import pybamm


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

        SoC = variables["SoC"]
        OCV = self.param.OCV(SoC)
        OCV_dim = self.param.OCV_ref + OCV * pot_scale

        # For coupling to current collectors
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = phi_s_cn + OCV

        phi_s_cp_dim = self.param.OCV_ref + phi_s_cp * pot_scale

        variables.update(
            {
                "Through-cell open-circuit voltage": OCV,
                "Through-cell open-circuit voltage [V]": OCV_dim,
                "Positive current collector potential": phi_s_cp,
                "Positive current collector potential [V]": phi_s_cp_dim,
            }
        )
        # Note: "Positive current collector potential" will be updated by other
        # circuit elements
        return variables
