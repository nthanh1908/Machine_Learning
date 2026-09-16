import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# =========================
# 1. Tạo dữ liệu
# =========================

np.random.seed(42)

X = np.linspace(-3, 3, 30).reshape(-1, 1)

y = 2 * X[:, 0]**2 + 3 * X[:, 0] + 5
y = y + np.random.normal(0, 3, size=len(X))


# =========================
# 2. Chia Train / Test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 3. Thử các bậc Polynomial
# =========================

degrees = [1, 2, 5, 10, 15]

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    # Train
    model.fit(X_train, y_train)

    # Điểm trên Train
    train_score = model.score(X_train, y_train)

    # Điểm trên Test
    test_score = model.score(X_test, y_test)

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()

    print(f"\nPolynomial degree = {degree}")
    print(f"Train R2 = {train_score:.3f}")
    print(f"Test R2  = {test_score:.3f}")
    print(f"CV R2    = {cv_mean:.3f}")