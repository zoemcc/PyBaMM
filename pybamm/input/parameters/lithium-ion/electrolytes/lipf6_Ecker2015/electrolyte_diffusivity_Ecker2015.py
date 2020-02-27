from pybamm import exp


def electrolyte_diffusivity_Ecker2015(c_e, T, T_inf, E_D_e, R_g):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
l

    Parameters
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_D_e: double
        Electrolyte diffusion activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Solid diffusivity
    """

    cm = 1e-3 * c_e

    sigma_e = 0.2667 * cm ** 3 - 1.2983 * cm ** 2 + 1.7919 * cm + 0.1726

    C = T_inf * exp(17100 / R_g * (1 / T_inf))

    sigma = C * sigma_e * exp(-17100 / R_g * (1 / T)) / T

    k_b = 1.38 * 1e-23
    F = 96487
    q_e = 1.602 * 1e-19

    D_c_e = k_b / (F * q_e) * sigma * T / c_e

    return D_c_e
