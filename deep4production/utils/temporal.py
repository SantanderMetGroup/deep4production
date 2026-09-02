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
                str(np.datetime64(date) - lag * np.timedelta64(1, "D"))[:10]
                for lag in reversed(range(num_lagged_x + 1))
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

    The YAML carries no time of day, so matching is done on the calendar day
    and the store's own timestamp is kept aside: stores may stamp a daily mean
    at midnight or at midday (12:00 is the CORDEX convention), and predictions
    have to be written back on the stamp their inputs came with, not on a
    midnight rebuilt from the date string.

    Returns
    -------
    sample_map : dict
        {YYYY-MM-DD: [zarr_file_idx, time_idx]}
    found_dates : list of str
        The requested days that were found, in request order.
    stamps : dict
        {YYYY-MM-DD: np.datetime64} — the store's full timestamp for that day.
    """
    # One O(1) lookup table per zarr — {YYYY-MM-DD: (time_idx, stamp)}
    luts = []
    for z in data_zarrs:
        raw = z["dates"][:]
        days = raw.astype("datetime64[D]").astype(str)
        # Dates the store declares missing occupy a slot on the fixed-frequency
        # axis but hold no data (anemoi writes them all-NaN), so they must never
        # become samples.
        missing = set(z.attrs.get("missing_dates", ()))
        kept = [(t, d) for t, d in enumerate(days) if d not in missing]
        lut = {d: (t, raw[t]) for t, d in kept}
        if missing:
            log.warning(
                "Store declares %d missing date(s) (%s%s); excluded from sampling.",
                len(missing),
                ", ".join(sorted(missing)[:3]),
                ", ..." if len(missing) > 3 else "",
            )
        if len(lut) != len(kept):
            log.warning(
                "Store has %d usable time steps but only %d distinct days; the "
                "last sample of each day is used.",
                len(kept),
                len(lut),
            )
        luts.append(lut)

    sample_map = {}
    found_dates = []
    stamps = {}
    # Misses are summarized, not logged one by one: a store that legitimately
    # covers less than the requested period (e.g. an ERA5-driven run inside a
    # multi-source recipe whose training_period spans the full GCM range) would
    # otherwise emit tens of thousands of lines per pydataset, per rank.
    missing_dates = []
    for date_yaml in dates_yaml:
        for i, lut in enumerate(luts):
            if date_yaml in lut:
                time_idx, stamp = lut[date_yaml]
                sample_map[date_yaml] = [i, time_idx]
                stamps[date_yaml] = stamp
                found_dates.append(date_yaml)
                break
        else:
            missing_dates.append(date_yaml)
    if missing_dates:
        log.warning(
            "%d of %d requested date(s) not found in any input data (%s ... %s); "
            "skipped.",
            len(missing_dates),
            len(dates_yaml),
            missing_dates[0],
            missing_dates[-1],
        )
    return sample_map, found_dates, stamps


# -------------------------------------------------------------------------
# CORDEX temporal chunking
# -------------------------------------------------------------------------
# The CORDEX archive splits a run's time series over several files on a fixed,
# archive-wide grid rather than at arbitrary offsets (CORDEX Archive Design
# §5.4): daily data goes in files of 5 years or less that START on a year
# ending in '1' or '6' and END on a year ending in '5' or '0'; monthly uses 10
# years ('1'→'0'); 3/6-hourly uses one year. Ragged chunks appear only at the
# ends of an experiment, e.g. a run starting in 1950 gives 1950, 1951-1955,
# 1956-1960, ...
#
# Both the span cap and the start/end alignment collapse to a single bucket
# key. For N years per file the valid end years are exactly those with
# year % N == 0, so grouping by
#
#     (year - 1) // N
#
# reproduces the grid for every frequency with no special-casing, and the
# ragged ends fall out for free because only the requested years land in a
# bucket.

# Frequency token used in output file names, keyed by the store's `frequency`
# attribute. CORDEX spells these 'day', 'mon', '3hr', '6hr', 'fx'.
_FREQ_TOKENS = {"1D": "day"}


def freq_token(freq):
    """
    CORDEX file-name token for a store frequency (``'1D'`` → ``'day'``).

    Raises on an unknown frequency rather than guessing a token, since a wrong
    one would silently mislabel published files.
    """
    try:
        return _FREQ_TOKENS[freq]
    except KeyError:
        raise ValueError(
            f"❌ No CORDEX frequency token known for '{freq}'. "
            f"Known frequencies: {sorted(_FREQ_TOKENS)}."
        ) from None


def cordex_year_chunks(dates, years_per_file=5):
    """
    Split an ordered date list onto the CORDEX file grid.

    Parameters
    ----------
    dates : list of str
        Chronologically ordered ``'YYYY-MM-DD'`` dates.
    years_per_file : int
        Chunk length N: 5 for daily, 10 for monthly, 1 for sub-daily.

    Returns
    -------
    list of (int, int)
        ``(i0, i1)`` half-open index slices into ``dates``, in chronological
        order, covering every date exactly once.
    """
    if years_per_file < 1:
        raise ValueError(f"❌ years_per_file must be >= 1, got {years_per_file}.")
    if len(dates) == 0:
        return []

    chunks = []
    i0 = 0
    key = (int(dates[0][:4]) - 1) // years_per_file
    for i, date in enumerate(dates):
        k = (int(date[:4]) - 1) // years_per_file
        if k != key:
            chunks.append((i0, i))
            i0 = i
            key = k
    chunks.append((i0, len(dates)))
    return chunks


def chunk_time_label(stamps):
    """
    ``StartTime-EndTime`` token for a chunk, as ``YYYYMMDD-YYYYMMDD``.

    Built from the first and last stamp of the chunk — the spec allows using
    the first/last record values for averaged data — so the label matches the
    time axis actually written, including stores that date daily means at 12:00.
    """
    first = np.datetime64(stamps[0], "D").astype(str).replace("-", "")
    last = np.datetime64(stamps[-1], "D").astype(str).replace("-", "")
    return f"{first}-{last}"


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
