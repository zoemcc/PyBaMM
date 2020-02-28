from pybamm import exp, tanh


def graphite_ocp_Ecker2015_function(sto):
    """
    Graphite OCP as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
       ----------
       .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
       """

    # Graphite Anode from Ecker, Kabitz, Laresgoiti et al.
    # Analytical fit (WebPlotDigitizer + gnuplot)
    a = 0.716502
    b = 369.028
    c = 0.12193
    d = 35.6478
    e = 0.0530947
    g = 0.0169644
    h = 27.1365
    i = 0.312832
    j = 0.0199313
    k = 28.5697
    m = 0.614221
    n = 0.931153
    o = 36.328
    p = 1.10743
    q = 0.140031
    r = 0.0189193
    s = 21.1967
    t = 0.196176

    u_eq = (
        a * exp(-b * sto)
        + c * exp(-d * (sto - e))
        - r * tanh(s * (sto - t))
        - g * tanh(h * (sto - i))
        - j * tanh(k * (sto - m))
        - n * exp(o * (sto - p))
        + q
    )

    return u_eq
