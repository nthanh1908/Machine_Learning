# Bài 3.27 - Perceptron

# Trọng số w
w = [1, 2, -10]

# Điểm dữ liệu x
# x đã có bias
x = [3, 4, 1]

# Nhãn thực tế
y = -1


# ==========================================
# 1. Tính w^T x
# ==========================================

wx = 0

for i in range(len(w)):
    wx = wx + w[i] * x[i]


print("w^T x =", wx)


# ==========================================
# 2. Xác định nhãn dự đoán
# ==========================================

if wx >= 0:
    y_pred = 1
else:
    y_pred = -1


print("Nhãn dự đoán =", y_pred)


# ==========================================
# 3. Kiểm tra phân lớp đúng hay sai
# ==========================================

print("Nhãn thực tế =", y)

if y_pred == y:
    print("Điểm dữ liệu được phân lớp đúng.")
else:
    print("Điểm dữ liệu bị phân lớp sai.")