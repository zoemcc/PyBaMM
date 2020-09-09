import pybamm


def get_standard_current_collector_potential_variables(param, phi_s_cn, phi_s_cp):
    """
    A function to obtain the standard variables which
    can be derived from the potentials in the current collector.

    Parameters
    ----------
    phi_cc : :class:`pybamm.Symbol`
        The potential in the current collector.

    Returns
    -------
    variables : dict
        The variables which can be derived from the potential in the
        current collector.
    """

    pot_scale = param.potential_scale
    phi_s_cp_dim = param.OCV_ref + phi_s_cp * pot_scale

    # Local potential difference
    V_cc = phi_s_cp - phi_s_cn

    # Terminal voltage
    # Note phi_s_cn is always zero at the negative tab
    V = pybamm.boundary_value(phi_s_cp, "positive tab")
    V_dim = pybamm.boundary_value(phi_s_cp_dim, "positive tab")

    # Voltage is local current collector potential difference at the tabs, in 1D
    # this will be equal to the local current collector potential difference

    variables = {
        "Negative current collector potential": phi_s_cn,
        "Negative current collector potential [V]": phi_s_cn * pot_scale,
        "Positive current collector potential": phi_s_cp,
        "Positive current collector potential [V]": phi_s_cp_dim,
        "Local voltage": V_cc,
        "Local voltage [V]": param.OCV_ref + V_cc * pot_scale,
        "Terminal voltage": V,
        "Terminal voltage [V]": V_dim,
    }

    return variables
