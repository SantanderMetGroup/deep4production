class d4dnormalizers():
    def __init__(self, mean, std, min, max):
        # Store statistics
        self.mean_i = mean
        self.std_i = std
        self.min_i = min
        self.max_i = max

    # ADD CUSTOM NORMALIZERS BELOW
    # -----------------------------------
    def none(self, array, denormalize=False):
        return array

    def mean_std(self, array, denormalize=False):
        return (array - self.mean_i) / self.std_i if not denormalize else array * self.std_i + self.mean_i

    def std(self, array, denormalize=False):
        return array / self.std_i if not denormalize else array * self.std_i

    def max(self, array, denormalize=False):
        return array / self.max_i if not denormalize else array * self.max_i

