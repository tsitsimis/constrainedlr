"""
Validation of Constrained Lineage Regression parameters and inputs
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def validate_constraint_features_all_strings_or_all_int(constraints: dict) -> None:
    """
    Validates the keys of the constraints dictionary

    Validates if they are all integers (column indices) or all strings (feature names)

    Args:
        constraints: Constraints dictionary (signs or range)

    Raises:
        ValueError: if dict keys are mixed integers and strings
    """
    features = list(constraints.keys())
    if not (
        all(isinstance(feature, int) for feature in features) or all(isinstance(feature, str) for feature in features)
    ):
        raise ValueError("Constraints must be all of type int or type str")


def get_clean_feature_names_from_pipeline(feature_names: list[str]) -> list[str]:
    """Removes feature name prefix added within a pipeline

    Args:
        feature_names: Feature names

    Returns:
        Formatted feature names
    """
    clean_feature_names = [feature_name.rsplit("__", 1)[-1] for feature_name in feature_names]
    return clean_feature_names


def validate_feature_names_in_constraints(constraints: dict, feature_names: list[str]) -> None:
    """Validates if features names in constraints exist as columns in input DataFrame

    Args:
        constraints: Constraints dictionary (signs or range)
        feature_names: Feature names
    """
    invalid_features = list(set(constraints.keys()) - set(feature_names))
    if len(invalid_features) > 0:
        raise ValueError(f"Features {invalid_features} are not in input")


def convert_feature_names_to_indices(constraints: dict, feature_names_in_: np.ndarray[str]) -> dict:
    """Converts constraints with feature names to feature indices

    Example:
        Input: `{"age": "positive"}`
        Output: `{0: "positive"}`  # "age" is the first column

    Args:
        constraints: Constraints dictionary (signs or range)
        feature_names_in_: Input feature names

    Returns:
        dict: _description_
    """
    constraints_feature_names = list(constraints.keys())
    clean_feature_names_in_ = get_clean_feature_names_from_pipeline(feature_names_in_)
    constraints_feature_indices = [
        clean_feature_names_in_.index(feature_name) for feature_name in constraints_feature_names
    ]
    feature_names_to_indices_map = dict(zip(constraints_feature_names, constraints_feature_indices))

    formatted_constraints = {
        feature_names_to_indices_map[feature_name]: constraint for feature_name, constraint in constraints.items()
    }
    return formatted_constraints


def validate_coefficients_sign_constraints(
    coefficients_sign_constraints: Optional[dict],
    X: Union[np.ndarray, pd.DataFrame],  # noqa: N803,
    feature_names_in_: Optional[np.ndarray[str]],
) -> dict:
    """
    Validates and formats coefficient sign constraints

    Args:
        coefficients_sign_constraints: _description_
        X: Input data
        feature_names_in_: Feature names in case of DataFrame input

    Returns:
        Formatted coefficient sign constraints
    """
    if coefficients_sign_constraints is None:
        return {}
    if not isinstance(coefficients_sign_constraints, dict):
        raise ValueError(
            "coefficients_sign_constraints must be of type dict,"
            f"now it is of type {type(coefficients_sign_constraints)}"
        )
    if len(coefficients_sign_constraints) == 0:
        return coefficients_sign_constraints

    validate_constraint_features_all_strings_or_all_int(coefficients_sign_constraints)

    if (feature_names_in_ is not None) and all(
        isinstance(feature, str) for feature in list(coefficients_sign_constraints.keys())
    ):
        coefficients_sign_constraints = convert_feature_names_to_indices(
            coefficients_sign_constraints, feature_names_in_
        )

    coef_indices = list(coefficients_sign_constraints.keys())
    if any(not isinstance(ci, int) for ci in coef_indices):
        raise ValueError("Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])")
    if (min(coef_indices) < 0) or (max(coef_indices) >= X.shape[1]):
        raise ValueError("Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])")

    if len(set(coefficients_sign_constraints.values()) - {-1, 0, 1, "positive", "negative"}) > 0:
        raise ValueError(
            "Values of coefficients_sign_constraints must be 0 (no sign constraint), "
            "'positive' or 1 (positive sign constraint), 'negative' or -1 (negative sign constraint)"
        )

    # Replace "positive" with 1, "negative" with -1 for compatibility with the optimizer
    coefficients_sign_constraints = {
        k: 1 if v == "positive" else -1 if v == "negative" else v for k, v in coefficients_sign_constraints.items()
    }
    return coefficients_sign_constraints


def validate_intercept_sign_constraint(intercept_sign_constraint: Union[int, str]) -> int:
    """
    Validates and formats Intercept sign constraint

    Args:
        intercept_sign_constraint: Sign constraint for intercept. It can be "positive" (or 1), "negative" (or -1),
        or 0 to indicate no constraint
    """
    if intercept_sign_constraint not in [-1, 0, 1, "positive", "negative"]:
        raise ValueError(
            "intercept_sign_constraint must be 0 (no sign constraint), 'positive' or 1 (positive sign constraint), "
            "'negative' or -1 (negative sign constraint)"
        )
    formatted_intercept_sign_constraint = (
        1
        if intercept_sign_constraint == "positive"
        else -1
        if intercept_sign_constraint == "negative"
        else intercept_sign_constraint
    )
    return formatted_intercept_sign_constraint  # type: ignore[return-value]


def validate_coefficients_range_constraints(
    coefficients_range_constraints: Optional[dict],
    X: Union[np.ndarray, pd.DataFrame],  # noqa: N803
    feature_names_in_: Optional[np.ndarray[str]],
) -> dict:
    """
    Validates and formats coefficient range constraints

    Args:
        coefficients_range_constraints: _description_
        X: Input data
        feature_names_in_: Input feature names

    Returns:
        Formatted coefficient range constraints
    """
    if coefficients_range_constraints is None:
        return {}
    if not isinstance(coefficients_range_constraints, dict):
        raise ValueError("coefficients_range_constraints must be of type dict")
    if len(coefficients_range_constraints) == 0:
        return coefficients_range_constraints

    validate_constraint_features_all_strings_or_all_int(coefficients_range_constraints)

    if (feature_names_in_ is not None) and all(
        isinstance(feature, str) for feature in list(coefficients_range_constraints.keys())
    ):
        coefficients_range_constraints = convert_feature_names_to_indices(
            coefficients_range_constraints, feature_names_in_
        )
        print(coefficients_range_constraints)
        print("hi")

    coef_indices = list(coefficients_range_constraints.keys())
    if any(not isinstance(ci, int) for ci in coef_indices):
        raise ValueError("Keys of coefficients_range_constraints must be integers within the interval [0, X.shape[1])")
    if (min(coef_indices) < 0) or (max(coef_indices) >= X.shape[1]):
        raise ValueError("Keys of coefficients_range_constraints must be integers within the interval [0, X.shape[1])")

    # Check that the nested dictionaries are of the form {"lower": <value>, "upper": <value>}
    range_dicts = list(coefficients_range_constraints.values())
    if any(len(set(range_dict.keys()) - {"lower", "upper"}) > 0 for range_dict in range_dicts):
        raise ValueError(
            "Values of coefficients_range_constraints must be dictionaries with keys 'lower' and/or 'upper'"
        )

    # Check that provided lower bound is always smaller than upper bound
    if any(
        ("lower" in range_dict and "upper" in range_dict) and (range_dict["lower"] > range_dict["upper"])
        for range_dict in range_dicts
    ):
        raise ValueError("Lower bound must always be smaller than the upper bound")

    return coefficients_range_constraints
