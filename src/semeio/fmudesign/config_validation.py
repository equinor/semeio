"""
Module for validation of config (typically read from Excel).
"""
import copy
from semeio.fmudesign._excel_to_dict import resolve_path, _read_background
import numbers


def validate_configuration(config:dict, input_filename) -> dict:
    """Main function for config validation.
    
    This function is responsible for:
        - Checking that required keys exist
        - Checking that values are set to valid types
        - Setting default values if keys are not set
    
    """
    config = copy.deepcopy(config)
    

    # Validate the general input sheet
    config["general_input"] = validate_general_input(config["general_input"],
                                                     input_filename)
    
    
    return config



def validate_general_input(config:dict, input_filename) -> dict:
    """Validate all config specified in the general input sheet."""
    config_out = copy.deepcopy(config)
    
    # 
    key = "repeats"
    if key not in config.keys():
        raise LookupError(f"Key {key!r} must be set in general input sheet.")
           
    key = "seeds"
    if key in config.keys():
        raise ValueError(
            "The 'seeds' parameter has been deprecated and is no longer supported. "
            "Use 'rms_seeds' instead"
        )
    
    
    key = "correlation_iterations"
    if key not in config.keys():
        print(f"Key {key!r} not set in general input sheet. Setting it to default value 0.")
        config_out[key] = 0
    if not isinstance(config_out[key], numbers.Integral):
        value = config_out[key]
        raise ValueError(f"Key {key!r} in general input sheet must be integer, got: {value}")
                         
                         
    key = "seeds"
    try:
        config_out[key] = resolve_path(input_filename, str(config_out["rms_seeds"]))
    except KeyError:
        config_out[key] = None
    except (ValueError, TypeError):
        config_out[key] = config_out["rms_seeds"]  # Validatetion done later
        
        
    key = "background"
    config_out[key] = {}
    try:
        value = str(config[key])
        if value.endswith(("csv", "xlsx")):
            config_out[key]["extern"] = resolve_path(input_filename, value)
        else:
            config_out[key] = _read_background(input_filename, value)
    except KeyError:
        config_out[key] = None
    except ValueError:
        config_out[key] = config[key]  # Validation should raise
        
        
    key = "distribution_seed"
    try:
        config_out[key] = int(config[key])
    except KeyError as err:
        raise ValueError(
            "You did not specify a value for 'distribution_seed', which is used to seed "
            "the random number generator that draws from distributions in Monte Carlo "
            "sensitivities.\n"
            "- Specify a number (e.g. a 6 digit integer) to seed the random number "
            "generator and obtain reproducible results.\n"
            "- Specify None if you do not want to seed the random number generator. "
            "Your analysis will not be reproducible."
        ) from err
        # If key does not exsist, raise an error and ask user to input key.
    except (ValueError, TypeError):
        config_out[key] = config[key]
        
    value = config_out[key]
        
    if not (isinstance(value, numbers.Integral) or value is None):
        raise ValueError(f"Key {key!r} in general input sheet must be integer or 'None', got: {value}")
        
    return config_out




