# Regression
Pure Numpy implementations of linear and logistic regression.

## Install

```python
pip install -r requirements.txt
# or
conda install -c conda-forge numpy pandas scipy matplotlib scikit-learn
```

## Usage

```python
from models import LinearRegression
...
model = LinearRegression(alpha=1e-3, n_iters=1000)
model.fit(X, y)
y_pred = model.predict(X_test)
```

## Tests

```shell
python main.py --linear-test
python main.py --logistic-test
```