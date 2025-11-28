def mean_std(array, mean, std, denormalize=False):
    return (array - mean) / std if not denormalize else array * std + mean

def std(array, mean, std, denormalize=False):
    return array / std if not denormalize else array * std

