import numpy as np

# --- Sin/cos Julian day ----------------------------------------------------------------
def compute_julian_day(dates, type, points=None):
    julian_day = dates.dayofyear.values
    theta = 2 * np.pi * julian_day / 365  # angle in radians

    if type == "cos":
        out = np.cos(theta)
    elif type == "sin":
        out = np.sin(theta)

    if points is not None:
        out = out[:, np.newaxis, np.newaxis]  # shape (time, 1, 1)
        out = np.tile(out, (1, 1, points))  # shape (time, 1, points)
    return out


# --- Sin/cos lat/lon ----------------------------------------------------------------
def compute_sincos_coords(coords, type, samples=1):
    if type == "cos":
        out = np.cos(coords)
    elif type == "sin":
        out = np.sin(coords)
    out = out[np.newaxis, np.newaxis, :]  # shape (1, 1, points)
    out = np.tile(out, (samples, 1, 1))  # shape (time, 1, points)
    return out



# --- Top of the atmosphere solar radiation ----------------------------------------------------------------
def compute_toa_solar_radiation(dates, lats):
    """
    Compute daily mean TOA solar radiation for each date and latitude.

    Parameters
    ----------
    dates : array-like of datetime64 or pandas timestamps
    lats : array-like, degrees

    Returns
    -------
    dict : {day_of_year: [toa_at_lat0, toa_at_lat1, ...], ...}
    """

    # Solar constant
    S0 = 1361.0  # W/m^2

    # Day of the year
    day_of_year = dates.dayofyear.values

    # 1. Earth–sun distance factor
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)

    # 2. Solar declination
    delta = 0.409 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Prepare output array: (time, lat)
    toa = np.zeros((len(day_of_year), len(lats)), dtype=np.float32)

    # Loop over latitudes
    for j, lat in enumerate(lats):
        phi = np.deg2rad(lat)

        # Sunset hour angle
        cosH0 = -np.tan(phi) * np.tan(delta)
        cosH0 = np.clip(cosH0, -1, 1)  # numerical safety
        H0 = np.arccos(cosH0)

        # TOA daily mean insolation
        toa[:, j] = (
            S0 / np.pi * dr *
            (H0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(H0))
        )

    # Expand dimensions: (time, 1, lat)
    toa = toa[:, np.newaxis, :]

    # Return
    return toa