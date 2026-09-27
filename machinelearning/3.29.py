# Bài 3.29 - Xây dựng lớp Perceptron

class Perceptron:

    # Hàm khởi tạo
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    # Hàm huấn luyện
    def fit(self, X, y):

        # Số lượng đặc trưng
        n_features = len(X[0])

        # Khởi tạo tất cả trọng số bằng 0
        self.weights = [0] * n_features

        # Lặp qua số epoch
        for epoch in range(self.epochs):

            # Duyệt từng mẫu dữ liệu
            for i in range(len(X)):

                # Lấy mẫu dữ liệu thứ i
                x = X[i]

                # Lấy nhãn thực tế
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

                # Nếu dự đoán sai thì cập nhật
                if prediction != target:

                    for j in range(len(x)):
                        self.weights[j] = (
                            self.weights[j]
                            + self.learning_rate * target * x[j]
                        )

                    self.bias = (
                        self.bias
                        + self.learning_rate * target
                    )

    # Hàm dự đoán
    def predict(self, X):

        predictions = []

        # Duyệt từng mẫu
        for x in X:

            # Tính w^T x + b
            z = self.bias

            for j in range(len(x)):
                z = z + self.weights[j] * x[j]

            # Xác định nhãn
            if z >= 0:
                prediction = 1
            else:
                prediction = -1

            # Thêm kết quả vào danh sách
            predictions.append(prediction)

        return predictions


# ==========================================
# DỮ LIỆU HUẤN LUYỆN
# ==========================================

X = [
    [2, 3],
    [1, 1],
    [-1, -2],
    [-2, -3]
]

y = [
    1,
    1,
    -1,
    -1
]


# ==========================================
# TẠO MÔ HÌNH
# ==========================================

model = Perceptron(
    learning_rate=0.1,
    epochs=10
)


# ==========================================
# HUẤN LUYỆN
# ==========================================

model.fit(X, y)


# ==========================================
# DỰ ĐOÁN
# ==========================================

predictions = model.predict(X)


# ==========================================
# IN KẾT QUẢ
# ==========================================

print("Trọng số sau khi huấn luyện:", model.weights)
print("Bias:", model.bias)
print("Nhãn thực tế:", y)
print("Nhãn dự đoán:", predictions)