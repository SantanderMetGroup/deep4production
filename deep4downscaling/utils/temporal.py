import numpy as np
# -------------------------------------------------------------------------
def get_pairs(dates, freq, num_lagged_x):
    dates_set = set(dates)
    pairs = {}
    for _, date in enumerate(dates):
        # Dates (instant and lagged)
        if freq == "1D":
            dates_pair = [
                str(np.datetime64(date) - l * np.timedelta64(1, 'D'))[:10]
                for l in reversed(range(num_lagged_x + 1))
            ]
        else:
            raise ValueError("❌ Temporal frequency different from daily not implemented.")
        # Do dates_pair exist in dates
        if all(dp in dates_set for dp in dates_pair):
            pairs[date] = dates_pair
    # Return
    return pairs


# -------------------------------------------------------------------------
def get_sample_map(dates_yaml, data_zarrs):
    sample_map = {}
    found_dates = []
    # Pre-convert zarr dates to YYYY-MM-DD strings
    zarr_dates = [
        [str(d)[:10] for d in z.attrs["dates"]]
        for z in data_zarrs
    ]
    # For each date in YAML, find its location in the zarr files
    for date_yaml in dates_yaml:
        found = False
        for i, dates_i in enumerate(zarr_dates):
            if date_yaml in dates_i:
                j = dates_i.index(date_yaml)
                sample_map[date_yaml] = [i, j]
                found_dates.append(date_yaml)
                found = True
        # Was the date found in any zarr file?
        if not found:
            print(f"⚠️ Date {date_yaml} not found in any input data. Skipping...")
    # Return
    return sample_map, found_dates

# -------------------------------------------------------------------------
def get_dates_from_yaml(years_yaml, freq):
    dates_yaml = []
    for year in years_yaml:
        # Dates in YAML
        if freq == "1D": 
            time_delta = np.timedelta64(1, 'D')
            start = np.datetime64(f"{year}-01-01")
            end = np.datetime64(f"{year}-12-31")
        else:
            raise ValueError("❌ Temporal frequency different from daily not implemented.")
        dates_yaml_year = np.arange(start, end + time_delta, time_delta)
        dates_yaml.append(dates_yaml_year)
    # Concatenate all years
    dates_yaml = np.concatenate(dates_yaml)
    # Return
    return [str(d)[:10] for d in dates_yaml]