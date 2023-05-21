def validate_coefficients_sign_constraints(coefficients_sign_constraints: dict, X) -> None:
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
