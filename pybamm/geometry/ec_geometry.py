#
# Function to create geometries for ec models
#
import pybamm


def ec_geometry(current_collector_dimension=0):
    """
    A convenience function to create geometries for ec models.

    Parameters
    ----------
    current_collector_dimensions : int, default
        The dimensions of the current collector. Should be 0 (default), 1 or 2

    Returns
    -------
    :class:`pybamm.Geometry`
        A geometry class for the battery

    """
    var = pybamm.standard_spatial_vars
    geo = pybamm.GeometricParameters()
    l_n = geo.l_n
    l_s = geo.l_s

    geometry = {}

    if current_collector_dimension == 0:
        geometry["current collector"] = {var.z: {"position": 1}}
    elif current_collector_dimension == 1:
        geometry["current collector"] = {
            var.z: {"min": 0, "max": 1},
            "tabs": {
                "negative": {"z_centre": geo.centre_z_tab_n},
                "positive": {"z_centre": geo.centre_z_tab_p},
            },
        }
    elif current_collector_dimension == 2:
        geometry["current collector"] = {
            var.y: {"min": 0, "max": geo.l_y},
            var.z: {"min": 0, "max": geo.l_z},
            "tabs": {
                "negative": {
                    "y_centre": geo.centre_y_tab_n,
                    "z_centre": geo.centre_z_tab_n,
                    "width": geo.l_tab_n,
                },
                "positive": {
                    "y_centre": geo.centre_y_tab_p,
                    "z_centre": geo.centre_z_tab_p,
                    "width": geo.l_tab_p,
                },
            },
        }
    else:
        raise pybamm.GeometryError(
            "Invalid current collector dimension '{}' (should be 0, 1 or 2)".format(
                current_collector_dimension
            )
        )

    return pybamm.Geometry(geometry)
