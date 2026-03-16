class d4dnormalizers():
    """
    Normalization class for climate data.
    Purpose: Provides methods for normalizing and denormalizing arrays using mean, std, min, max.
    Parameters:
        mean: Mean value.
        std: Standard deviation.
        min: Minimum value.
        max: Maximum value.
    """
    def __init__(self, mean, std, min, max):
        # Store statistics
        self.mean_i = mean
        self.std_i = std
        self.min_i = min
        self.max_i = max

    # ADD CUSTOM NORMALIZERS BELOW
    # -----------------------------------
    def none(self, array, denormalize=False):
        """
        No normalization. Returns array unchanged.
        Parameters:
            array: Input array.
            denormalize (bool): If True, returns unchanged.
        Returns:
            array: Unchanged array.
        """
        return array

    def mean_std(self, array, denormalize=False):
        """
        Normalizes array using mean and std, or denormalizes if specified.
        Parameters:
            array: Input array.
            denormalize (bool): If True, denormalizes.
        Returns:
            array: Normalized or denormalized array.
        """
        return (array - self.mean_i) / self.std_i if not denormalize else array * self.std_i + self.mean_i

    def std(self, array, denormalize=False):
        """
        Normalizes array using std, or denormalizes if specified.
        Parameters:
            array: Input array.
            denormalize (bool): If True, denormalizes.
        Returns:
            array: Normalized or denormalized array.
        """
        return array / self.std_i if not denormalize else array * self.std_i

    def max(self, array, denormalize=False):
        """
        Applies max normalization or denormalization.
        Parameters:
            array: Input array.
            denormalize (bool): If True, denormalizes.
        Returns:
            array: Normalized or denormalized array.
        """
        return array / self.max_i if not denormalize else array * self.max_i

