def cordex_ml(ds, attributes_dict):

    """
    Set CORDEX ESD-specific global attributes on an xarray Dataset according to
    the CORDEX experiment design for statistical downscaling of CMIP6.
    This function removes all existing global attributes and replaces them with
    only the CORDEX ESD-compliant metadata.

    https://cordex.org/wp-content/uploads/2024/04/Second-order-draft-CORDEX-experiment-design-for-statistical-downscaling-of-CMIP6.pdf

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to modify
    attributes_dict : dict
        Dictionary containing the attributes to set. Required keys:
        - 'project_id': Should be 'CORDEX'
        - 'activity_id': Should be 'ML-Benchmark'
        - 'source_type': Should be 'ESD-Combined'
        - 'esd_method_id': Short identifier of the ESD method
        - 'esd_method_version': Version of the ESD method
        - 'esd_method_description': Free text describing the ESD method
        - 'training_methodology': Free text describing training methodology and data
        - 'training_methodology_id': Short identifier for training configuration (no dashes)
        - 'training_experiment': What training experiment was used to train the model
        - 'evaluation_experiment': What evaluation experiment was used to evaluate the model
        - 'probabilistic_output': 'yes' if probabilistic method
        - 'realization_generation_id': 
            For a single realization: 'rM' (where M is the realization number, e.g., 'r1')
            For an ensemble of realizations: 'ensemble'
        - 'institution_id': Institution identifier
        - 'further_info_url': URL for further information (e.g. publication or repository)
        - 'source_id': Full source identifier (esd_method_id-esd_version-training_methodology_id-realization_generation_id)

    Returns:
    --------
    xarray.Dataset
        Dataset with updated attributes
    """

    # Fix the boolean
    if isinstance(attributes_dict.get("probabilistic_output"), bool):
        attributes_dict["probabilistic_output"] = str(attributes_dict["probabilistic_output"])
        
    # Create a copy to avoid modifying the original
    ds_copy = ds.copy()

    # Clear all existing global attributes
    ds_copy.attrs.clear()

    # Set all provided attributes
    for key, value in attributes_dict.items():
        ds_copy.attrs[key] = value

    # Generate the source_id
    if 'source_id' not in attributes_dict and 'esd_method_id' in attributes_dict and 'training_methodology_id' in attributes_dict and 'esd_version' in attributes_dict:
        ds_copy.attrs['source_id'] = f"{attributes_dict['esd_method_id']}-{attributes_dict['esd_version']}-{attributes_dict['training_methodology_id']}-{attributes_dict['realization_generation_id']}"

    return ds_copy