import numpy as np


class d4dunits:
    """
    Per-variable physical unit conversions applied at dataset-creation time
    (NetCDF -> Zarr), before per-channel statistics are computed.

    Each conversion is a named method that takes the raw variable array and
    returns ``(converted_array, target_units)``. The returned units string is
    written back into the store's ``units`` attribute so the metadata stays
    consistent with the stored values.

    Conversions are referenced from the create-config by name, mirroring the
    ``imputer`` pattern, e.g.::

        data:
          unit_conversion:
            pr:
              name: flux_to_mm_day
            tasmax:
              name: kelvin_to_celsius

    Note on precipitation: ``kg m-2 s-1 -> mm/day`` is *not* a pure dimensional
    conversion (mass/area vs length). It is only valid under the water-density
    assumption 1 kg m-2 == 1 mm of water. That assumption is baked explicitly
    into ``flux_to_mm_day`` rather than delegated to a generic units library,
    which would (correctly) refuse the conversion as dimensionally inconsistent.

    Parameters
    ----------
    data : np.ndarray
        Raw variable array as read from the source NetCDF.
    """

    def __init__(self, data):
        self.data = data

    # ADD CUSTOM UNIT CONVERSIONS BELOW
    # -----------------------------------
    def affine(self, scale=1.0, offset=0.0, units="N/A"):
        """Generic linear conversion: ``out = data * scale + offset``.

        Use for one-off conversions not covered by a named method. ``units``
        is the target-unit string to record in the store metadata.
        """
        return self.data * scale + offset, units

    # -----------------------------------
    def kelvin_to_celsius(self):
        """Temperature: K -> degC."""
        return self.data - 273.15, "degC"

    # -----------------------------------
    def celsius_to_kelvin(self):
        """Temperature: degC -> K."""
        return self.data + 273.15, "K"

    # -----------------------------------
    def flux_to_mm_day(self):
        """Precipitation flux -> daily accumulation: kg m-2 s-1 -> mm/day.

        Valid under the water-density assumption (1 kg m-2 == 1 mm); the
        86400 factor converts per-second to per-day.
        """
        return self.data * 86400.0, "mm/day"

    # -----------------------------------
    def mm_day_to_flux(self):
        """Precipitation daily accumulation -> flux: mm/day -> kg m-2 s-1."""
        return self.data / 86400.0, "kg m-2 s-1"

    # -----------------------------------
    def pa_to_hpa(self):
        """Pressure: Pa -> hPa."""
        return self.data / 100.0, "hPa"

    # -----------------------------------
    def hpa_to_pa(self):
        """Pressure: hPa -> Pa."""
        return self.data * 100.0, "Pa"

    # -----------------------------------
    def m_to_mm(self):
        """Length / accumulation: m -> mm (e.g. precip accumulation)."""
        return self.data * 1000.0, "mm"
