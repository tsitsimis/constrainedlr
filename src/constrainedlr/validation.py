from typing import Union


def validate_coefficients_sign_constraints(coefficients_sign_constraints: dict, X) -> dict:
    if type(coefficients_sign_constraints) != dict:
        raise ValueError("coefficients_sign_constraints must be of type dict")

    if len(coefficients_sign_constraints) > 0:
        coef_indices = list(coefficients_sign_constraints.keys())
        if any([type(ci) != int for ci in coef_indices]):
            raise ValueError(
                "Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])"
            )
        if (min(coef_indices) < 0) or (max(coef_indices) >= X.shape[1]):
            raise ValueError(
                "Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])"
            )

        if len(set(coefficients_sign_constraints.values()) - {-1, 0, 1, "positive", "negative"}) > 0:
            raise ValueError(
                "Values of coefficients_sign_constraints must be 0 (no sign constraint), 'positive' or 1 (positive sign constraint), "
                "'negative' or -1 (negative sign constraint)"
            )

        # Replace "positive" with 1, "negative" with -1 for compatibility with the optimizer
        coefficients_sign_constraints = {
            k: 1 if v == "positive" else -1 if v == "negative" else v for k, v in coefficients_sign_constraints.items()
        }
    return coefficients_sign_constraints


def validate_intercept_sign_constraint(intercept_sign_constraint: Union[int, str]) -> int:
    if intercept_sign_constraint not in [-1, 0, 1, "positive", "negative"]:
        raise ValueError(
            "intercept_sign_constraint must be 0 (no sign constraint), 'positive' or 1 (positive sign constraint), "
            "'negative' or -1 (negative sign constraint)"
        )
    intercept_sign_constraint = (
        1
        if intercept_sign_constraint == "positive"
        else -1
        if intercept_sign_constraint == "negative"
        else intercept_sign_constraint
    )
    return intercept_sign_constraint


def validate_coefficients_range_constraints(coefficients_range_constraints: dict, X) -> None:
    if type(coefficients_range_constraints) != dict:
        raise ValueError("coefficients_range_constraints must be of type dict")

    if len(coefficients_range_constraints) > 0:
        coef_indices = list(coefficients_range_constraints.keys())
        if any([type(ci) != int for ci in coef_indices]):
            raise ValueError(
                "Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])"
            )
        if (min(coef_indices) < 0) or (max(coef_indices) >= X.shape[1]):
            raise ValueError(
                "Keys of coefficients_sign_constraints must be integers within the interval [0, X.shape[1])"
            )

        # Check that the nested dictionaries are of the form {"lower": <value>, "upper": <value>}
        range_dicts = list(coefficients_range_constraints.values())
        if any([len(set(range_dict.keys()) - {"lower", "upper"}) > 0 for range_dict in range_dicts]):
            raise ValueError(
                "Values of coefficients_range_constraints must be dictionaries with keys 'lower' and/or 'upper'"
            )

        # Check that provided lower bound is always smaller than upper bound
        if any(
            [
                ("lower" in range_dict and "upper" in range_dict) and (range_dict["lower"] > range_dict["upper"])
                for range_dict in range_dicts
            ]
        ):
            raise ValueError("Lower bound must always be smaller than the upper bound")
