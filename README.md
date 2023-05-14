# Constrained Linear Regression
constrained-lr is a drop-in replacement for `scikit-learn`'s `linear_model.LinearRegression` with the additional flexibility to define more complex (but linear) constraints on the model's coefficients.

### Use-cases
#### SHAP
The Kernel SHAP algorithm includes the training of a constrainted linear regression model where the sum of its coefficients is equal to the model's prediction

#### Marketing Mix Modeling
In Marketing Mix Modeling (MMM), the attribution of sales to various marketing channels can be informed by business sense or prior knowledge, by enforcing the contribution of channel variables to be positive or negative.

### Installation
```bash
pip install constrained-lr
```

### Example Usage
```python
from constrained_lr import ConstrainedLinearRegression
from sklearn.metrics import mean_squared_error

model = ConstrainedLinearRegression()
model.fit(X_train, y_train, features_sign_constraints={"s3": 1, "s4": -1})

y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(dict(zip(X.columns, model.coef_)))
```

See full example in the [notebook](./notebooks/demo.ipynb)


### Licence
MIT
