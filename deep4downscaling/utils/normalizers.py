import pandas as pd
import zarr
import xarray as xr
import numpy as np

class d4dnormalizers():
    def __init__(self, mean, std, min, max, ref1=None, ref2=None):
        # Store statistics
        self.mean_i = mean
        self.std_i = std
        self.min_i = min
        self.max_i = max
        self.ref1 = ref1
        self.ref2 = ref2
    # ADD CUSTOM NORMALIZERS BELOW
    # -----------------------------------
    def none(self, array, denormalize=False):
        return array

    def mean_std(self, array, denormalize=False):
        print("Performing mean_std normalization NOT in build...")
        print(f"Normalization mean and std: {self.mean_i} - {self.std_i}")
        print(f"Normalization array: {array.mean()}")

        
        return (array - self.mean_i) / self.std_i if not denormalize else array * self.std_i + self.mean_i

    def std(self, array, denormalize=False):
        print("Performing std normalization NOT in build...")
        return array / self.std_i if not denormalize else array * self.std_i

    def max(self, array, denormalize=False):
        print("Performing max normalization NOT in build...")
        return array / self.max_i if not denormalize else array * self.max_i

    def bias_adjust(self, array, date=None, variable=None, denormalize=False):
        """Perform bias adjustment by aligning the monthly means of the input array to those of the reference datasets, being ref1: Driving GCM, ref2: U-RCM."""
        print("Performing bias adjustment NOT in build...")
        print(f"Reference dataset 1: {self.ref1}")
        print(f"Array: {len(array)}")
        ds_ref1 = xr.open_dataset(f"{self.ref1}")
        ds_ref2 = xr.open_dataset(f"{self.ref2}")
        # if check_variable_order(ds_ref1, array.attrs):

        #     zarr_order = sorted(
        #         array.attrs["variables"],
        #         key=lambda k: array.attrs["variables"][k]
        #     )

        #     ds_ref1 = ds_ref1[zarr_order]
        #     ds_ref2 = ds_ref2[zarr_order]

        # ds_ref1_T = ds_ref1.T      # (month, var)
        # ds_ref2_T = ds_ref2.T      # (month, var)
        time = pd.to_datetime(date)
        month = time.month
        # m_idx = months -1   # (batch_size,)
        # Select the monthly means for each variable and month
        # ref1_monthly_means = ds_ref1_T[m_idx]
        # ref2_monthly_means = ds_ref2_T[m_idx]
        # ref1_monthly_means = ref1_monthly_means[:, :, None] # (batch_size, var, 1)
        # ref2_monthly_means = ref2_monthly_means[:, :, None] # (batch_size, var, 1)
        ref1_monthly_means = ds_ref1[variable].sel(month=month).values  # (var,)
        ref2_monthly_means = ds_ref2[variable].sel(month=month).values  # (var,)
        if not denormalize:
            corrected = (
                array
                - ref1_monthly_means
                + ref2_monthly_means
            )
        else:
            corrected = (
                array
                + ref1_monthly_means
                - ref2_monthly_means
            )
        return corrected
    

# def check_variable_order(ds, zarr_attrs):
#     for var in ds.data_vars:
#         if var not in zarr_attrs["variables"]:
#             print(f"Variable {var} not found in Zarr attributes")
#             return False
#     sorted_ds_vars = list(ds.data_vars)
#     sorted_zarr_vars = sorted(zarr_attrs["variables"].keys(), key=lambda k: zarr_attrs["variables"][k])
#     if sorted_ds_vars != sorted_zarr_vars:
#         print("Variable order in dataset does not match Zarr variable order")
#         print("Dataset variables:", sorted_ds_vars)
#         print("Zarr variables:", sorted_zarr_vars)
#         return False
#     print("Variable order matches Zarr variable order")
#     return True