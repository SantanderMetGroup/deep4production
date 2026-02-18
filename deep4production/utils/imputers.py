import numpy as np
from deep4production.utils.general import get_func_from_string

class d4dimputers():
    def __init__(self, data, lat_gp, lon_gp, lats_ref, lons_ref):
        # Store reference and target coordinates
        self.lat_gp = lat_gp
        self.lon_gp = lon_gp
        self.lats_ref = lats_ref
        self.lons_ref = lons_ref
        
        # Get idx of gridpoint
        idx_lat_gp = np.where(np.array(lats_ref)==lat_gp)[0]
        idx_lon_gp = np.where(np.array(lons_ref)==lon_gp)[0]
        self.idx = np.intersect1d(idx_lat_gp, idx_lon_gp)[0]

        # Store dataset
        self.data = data # Shape: (G)

    # ADD CUSTOM IMPUTERS BELOW
    # -----------------------------------
    def constant(self, value):
        # Optionally assign value 
        return value

    # -----------------------------------
    def nearest_spatial(self, num_nearest_neighbours, aggr_function="mean"):
        """
        Find nearest spatial neighbors to (lat_gp, lon_gp) using great-circle distance.

        Parameters
        ----------
        num_nearest_neighbours : int
            Number of nearest neighbors to include (excluding the point itself).
        aggr_function : str
            Numpy function used to aggregate values (e.g., mean, median, sum).

        Returns
        -------
        aggregated_value : float
            Aggregated value over the nearest neighbors.
        """
        # Convert degrees to radians
        lat1 = np.radians(self.lat_gp)
        lon1 = np.radians(self.lon_gp)
        lat2 = np.radians(self.lats_ref)
        lon2 = np.radians(self.lons_ref)

        # Vectorized haversine distance (great-circle)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2 # Harvesine formula. Since sin is periodic, it handles the cases where points are e.g., -178, 178 degrees of latitude (or equivalent in radians)
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # Earth radius in km
        distances = R * c

        # Sort and select nearest neighbors (excluding the point itself)
        sorted_idx = np.argsort(distances)
        valid_sorted_idx = sorted_idx[~np.isnan(self.data[sorted_idx])] # True: non-NaN, False: NaN
        nearest_idx = valid_sorted_idx[1:num_nearest_neighbours + 1]  # skip self (idx 0)
        # print(f"Grid point:({self.lat_gp},{self.lon_gp})")
        # print(f"Grid point (1-closest):({self.lats_ref[nearest_idx[0]]},{self.lons_ref[nearest_idx[0]]})")
        # print(f"Grid point (2-closest):({self.lats_ref[nearest_idx[1]]},{self.lons_ref[nearest_idx[1]]})")
        # print(f"Grid point (3-closest):({self.lats_ref[nearest_idx[2]]},{self.lons_ref[nearest_idx[2]]})")
        # print(f"Grid point (4-closest):({self.lats_ref[nearest_idx[3]]},{self.lons_ref[nearest_idx[3]]})")

        # Aggregate over selected neighbors
        aggr_function = get_func_from_string("numpy", aggr_function)
        nearest_values = self.data[nearest_idx]
        aggregated_value = aggr_function(nearest_values)
    
        return aggregated_value
        
        

