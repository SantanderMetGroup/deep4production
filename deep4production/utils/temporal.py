import numpy as np
from deep4production.utils.log import get_logger

log = get_logger("temporal")


# -------------------------------------------------------------------------
def get_pairs(dates, freq, num_lagged_x):
    dates_set = set(dates)
    pairs = {}
    for _, date in enumerate(dates):
        # Dates (instant and lagged)
        if freq == "1D":
            dates_pair = [
                str(np.datetime64(date) - l * np.timedelta64(1, "D"))[:10]
                for l in reversed(range(num_lagged_x + 1))
            ]
        else:
            raise ValueError(
                "❌ Temporal frequency different from daily not implemented."
            )
        # Do dates_pair exist in dates
        if all(dp in dates_set for dp in dates_pair):
            pairs[date] = dates_pair
    # Return
    return pairs


# -------------------------------------------------------------------------
def get_sample_map(dates_yaml, data_zarrs):
    """
    Map each requested YAML date (YYYY-MM-DD string) to its location
    [zarr_file_idx, time_idx] in the provided zarr stores.

    First match wins: if the same date is present in multiple zarr files,
    the lowest-indexed file is used.
    """
    # One O(1) lookup table per zarr — {YYYY-MM-DD: time_idx}
    luts = [
        {d: t for t, d in enumerate(z["dates"][:].astype("datetime64[D]").astype(str))}
        for z in data_zarrs
    ]

    sample_map = {}
    found_dates = []
    for date_yaml in dates_yaml:
        for i, lut in enumerate(luts):
            if date_yaml in lut:
                sample_map[date_yaml] = [i, lut[date_yaml]]
                found_dates.append(date_yaml)
                break
        else:
            log.warning("Date %s not found in any input data; skipping.", date_yaml)
    return sample_map, found_dates


# -------------------------------------------------------------------------
def get_dates_from_yaml(years_yaml, freq):
    dates_yaml = []
    for year in years_yaml:
        # Dates in YAML
        if freq == "1D":
            time_delta = np.timedelta64(1, "D")
            start = np.datetime64(f"{year}-01-01")
            end = np.datetime64(f"{year}-12-31")
        else:
            raise ValueError(
                "❌ Temporal frequency different from daily not implemented."
            )
        dates_yaml_year = np.arange(start, end + time_delta, time_delta)
        dates_yaml.append(dates_yaml_year)
    # Concatenate all years
    dates_yaml = np.concatenate(dates_yaml)
    # Return
    return [str(d)[:10] for d in dates_yaml]
