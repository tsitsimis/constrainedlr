# ConstrainedLR
constrainedlr is a drop-in replacement of sklearn's `linear_model.LinearRegression`, `linear_model.RidgeRegression`, `linear_model.Elasticnet`, `linear_model.Lasso`, with the additional ability to add coefficient constraints

**Source Code:** [https://github.com/tsitsimis/constrainedlr](https://github.com/tsitsimis/constrainedlr).

## Installation

<!-- termynal -->

```
$ pip install constrainedlr
```

## Getting Started

### Sign constraints
Apply constraints on the signs of one or more coefficients of the model.

```python
from sklearn.datasets import load_diabetes
from constrainedlr.model import ConstrainedLinearRegression

# Load dataset
dataset = load_diabetes()
X = dataset["data"]
y = dataset["target"]

# Instantiate Constrained Linear Regression model
model = ConstrainedLinearRegression(fit_intercept=True)

# Fit model and constraint the sign of the 1st and 3rd coefficient
# Coefficients are selected based on their index (zero-based) in the dataset
sign_constraints = {
    0: "positive",  # Coefficient of 1st feature must be positive
    2: "negative":,  # Coefficient of 3rd feature must be negative
    # The remaining coefficients are not specified and by default have no sign constraints
}
model.fit(X, y, coefficients_sign_constraints=sign_constraints)
print(model.coef_)
```

You can also impose constraint on the sign of the intercept:
```python
model.fit(X, y, intercept_sign_constraint="positive")
print(model.intercept_)
```

### Range constraints
Apply constraints on the value of one or more coefficients. It enables to define a lower and/or upper bound of each coeficient.

```python
model = ConstrainedLinearRegression(fit_intercept=True)

# Fit model and constraint the sign of the 1st and 3rd coefficient
# Coefficients are selected based on their index (zero-based) in the dataset
range_constraints = {
    0: {"lower": 2},  # Coefficient of 1st feature must be 2 or higher
    2: {"upper": 10},  # Coefficient of 3rd feature can not be larger than 10
    3: {"lower": 3, "upper": 4},  # Coefficient of 4th feature must have a value between 3 and 4
    # The remaining coefficients are not specified and by default have no range constraints
}
model.fit(X, y, coefficients_range_constraints=range_constraints)
print(model.coef_)
```
