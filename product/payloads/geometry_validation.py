"""
Geometry validation rules for SCYTHE payloads.
Strict structural validation only. No physics.
"""

def validate_geometry(payload_type, dimensions):
    """
    Validates that the dimensions dictionary contains the required keys
    for the given payload_type.
    
    Args:
        payload_type (str): one of 'sphere', 'cylinder', 'box', 'capsule', 'blunt_cone'
        dimensions (dict): dictionary of dimension values (float)
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: if payload_type is unknown or dimensions are missing/invalid
    """
    if not isinstance(dimensions, dict):
        raise ValueError("Dimensions must be a dictionary.")

    # Check for non-negative values
    for k, v in dimensions.items():
        if not isinstance(v, (int, float)):
             raise ValueError(f"Dimension '{k}' must be a number.")
        if v < 0:
            raise ValueError(f"Dimension '{k}' cannot be negative.")

    if payload_type == "sphere":
        _require_keys(dimensions, ["diameter_m"])
        # model implied: axisymmetric

    elif payload_type == "cylinder":
        _require_keys(dimensions, ["length_m", "diameter_m"])
        # model implied: axisymmetric

    elif payload_type == "box":
        _require_keys(dimensions, ["length_m", "width_m", "height_m"])
        # model implied: 3D

    elif payload_type == "capsule":
        _require_keys(dimensions, ["length_m", "diameter_m"])
        # model implied: axisymmetric

    elif payload_type == "blunt_cone":
        _require_keys(dimensions, ["length_m", "base_diameter_m"])
        # model implied: axisymmetric

    else:
        raise ValueError(f"Unknown payload geometry type: {payload_type}")

    return True

def validate_aerodynamics(payload_type, cd, cd_uncertainty=None):
    """
    Validates that the Drag Coefficient (Cd) is within the allowable
    physically realistic range for the given geometry type.

    Ranges (orientation-averaged, unpowered):
      - sphere:      [0.45, 0.60]
      - capsule:     [0.50, 0.90]
      - cylinder:    [0.80, 1.20]
      - blunt_cone:  [0.60, 1.00]
      - box:         [1.00, 1.50]

    Args:
        payload_type (str): Geometry type.
        cd (float): Declared drag coefficient.
        cd_uncertainty (float, optional): Declared uncertainty.

    Raises:
        ValueError: If Cd is outside the allowed range.
    """
    # Allowed ranges mapping
    allowed_cd = {
        "sphere": (0.45, 0.60),
        "capsule": (0.50, 0.90),
        "cylinder": (0.80, 1.20),
        "blunt_cone": (0.60, 1.00),
        "box": (1.00, 1.50),
    }

    if payload_type not in allowed_cd:
        # Unknown type? defaulting to loose check or error?
        # User defined strictly these 5 categories.
        raise ValueError(f"Unknown payload geometry type for aerodynamics: {payload_type}")

    min_cd, max_cd = allowed_cd[payload_type]
    if not (min_cd <= cd <= max_cd):
        raise ValueError(f"Cd {cd} for '{payload_type}' is out of bounds [{min_cd}, {max_cd}].")

    # Optional uncertainty check? User said "Optionally include... consistent with geometry".
    # We won't strictly validate uncertainty *value* unless requested, but we could default it.
    return True

def _require_keys(data, required_keys):
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required dimensions: {missing}")
