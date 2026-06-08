# 🚗 KM/H to MPH Regression

A simple PyTorch linear regression model that learns the relationship between kilometers per hour (km/h) and miles per hour (mph).

## Formula

```text
mph = km/h × 0.621371
```

## Tech Stack

- Python
- PyTorch
- Scikit-Learn

## Concepts

- Linear Regression
- Feature Scaling (Normalization)
- SGD Optimization
- MSE Loss

## Run

```bash
pip install -r requirements.txt
python src/train.py
```

## Example Output

```text
30 km/h  -> 18.6411 mph
60 km/h  -> 37.2823 mph
120 km/h -> 74.5645 mph
```

## Note

This project is also my first experiment with data normalization using `StandardScaler`.

---

Built while learning PyTorch and machine learning fundamentals.
