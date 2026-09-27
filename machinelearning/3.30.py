# Bài 3.30 - Perceptron và đánh giá mô hình

# ==========================================
# 1. Xây dựng lớp Perceptron
# ==========================================

class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):

        # Số lượng đặc trưng
        n_features = len(X[0])

        # Khởi tạo trọng số
        self.weights = [0] * n_features

        # Huấn luyện
        for epoch in range(self.epochs):

            for i in range(len(X)):

                x = X[i]
                target = y[i]

                # Tính w^T x + b
                z = self.bias

                for j in range(len(x)):
                    z = z + self.weights[j] * x[j]

                # Dự đoán
                if z >= 0:
                    prediction = 1
                else:
                    prediction = -1

                # Nếu dự đoán sai
                if prediction != target:

                    # Cập nhật trọng số
                    for j in range(len(x)):
                        self.weights[j] = (
                            self.weights[j]
                            + self.learning_rate * target * x[j]
                        )

                    # Cập nhật bias
                    self.bias = (
                        self.bias
                        + self.learning_rate * target
                    )

    def predict(self, X):

        predictions = []

        for x in X:

            # Tính w^T x + b
            z = self.bias

            for j in range(len(x)):
                z = z + self.weights[j] * x[j]

            # Dự đoán
            if z >= 0:
                prediction = 1
            else:
                prediction = -1

            predictions.append(prediction)

        return predictions


# ==========================================
# 2. Tạo dữ liệu
# ==========================================

X_train = [
    [2, 3],
    [3, 4],
    [4, 5],
    [-1, -2],
    [-2, -3],
    [-3, -4]
]

y_train = [
    1,
    1,
    1,
    -1,
    -1,
    -1
]


# ==========================================
# 3. Dữ liệu kiểm tra
# ==========================================

X_test = [
    [5, 6],
    [1, 2],
    [-4, -5],
    [-1, -3]
]

y_test = [
    1,
    1,
    -1,
    -1
]


# ==========================================
# 4. Tạo mô hình
# ==========================================

model = Perceptron(
    learning_rate=0.1,
    epochs=10
)


# ==========================================
# 5. Huấn luyện
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 6. Dự đoán
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. In kết quả
# ==========================================

print("Trọng số:", model.weights)
print("Bias:", model.bias)

print("Nhãn thực tế:", y_test)
print("Nhãn dự đoán:", y_pred)


# ==========================================
# 8. Tính TP, TN, FP, FN
# ==========================================

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(y_test)):

    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1

    elif y_test[i] == -1 and y_pred[i] == -1:
        TN += 1

    elif y_test[i] == -1 and y_pred[i] == 1:
        FP += 1

    elif y_test[i] == 1 and y_pred[i] == -1:
        FN += 1


# ==========================================
# 9. Accuracy
# ==========================================

accuracy = (TP + TN) / len(y_test)


# ==========================================
# 10. Precision
# ==========================================

if TP + FP != 0:
    precision = TP / (TP + FP)
else:
    precision = 0


# ==========================================
# 11. Recall
# ==========================================

if TP + FN != 0:
    recall = TP / (TP + FN)
else:
    recall = 0


# ==========================================
# 12. F1-score
# ==========================================

if precision + recall != 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0


# ==========================================
# 13. In kết quả đánh giá
# ==========================================

print("\nKết quả đánh giá:")
print("TP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

print("Accuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1-score =", f1)