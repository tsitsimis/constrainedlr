# Constrained Linear Regression
<a href="https://pypi.org/project/constrainedlr" target="_blank">
    <img src="https://img.shields.io/pypi/v/constrainedlr?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/constrainedlr" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/constrainedlr.svg?color=%2334D058" alt="Supported Python versions">
</a>

constrainedlr is a drop-in replacement for `scikit-learn`'s `linear_model.LinearRegression` with the extended capability to apply constraints on the model's coefficients, such as signs and lower/upper bounds.

## Installation
```bash
pip install constrainedlr
```

## Example Usage

### Coefficients sign constraints
```python
from constrainedlr import ConstrainedLinearRegression

model = ConstrainedLinearRegression()

model.fit(
    X_train,
    y_train,
    coefficients_sign_constraints={0: "positive", 2: "negative"},
    intercept_sign_constraint="positive",
)

y_pred = model.predict(X_test)

print(model.coef_, model.intercept_)
```

### Coefficients range constraints
```python
from constrainedlr import ConstrainedLinearRegression

model = ConstrainedLinearRegression()

model.fit(
    X_train,
    y_train,
    coefficients_range_constraints={
        0: {"lower": 2},  # 1st coefficient must be 2 or higher
        2: {"upper": 10},  # 3rd coefficient must be smaller than 10
        3: {"lower": 0.1, "upper": 0.5},  # 4th coefficient must be between 0.1 and 0.5
    },
)

y_pred = model.predict(X_test)

print(model.coef_)
```

See more in the [documentation](https://tsitsimis.github.io/constrainedlr/)


### Licence
MIT
