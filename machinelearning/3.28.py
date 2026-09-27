# Bài 3.28 - Perceptron Update

# Trọng số ban đầu
w = [-2, 1, 0]

# Điểm dữ liệu x
# x đã có bias
x = [2, 3, 1]

# Nhãn thực tế
y = 1

# Learning rate
eta = 1


# ==========================================
# 1. Tính w^T x
# ==========================================

wx = 0

for i in range(len(w)):
    wx = wx + w[i] * x[i]

print("w^T x ban đầu =", wx)


# ==========================================
# 2. Xác định nhãn dự đoán
# ==========================================

if wx >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)
print("Nhãn thực tế =", y)


# ==========================================
# 3. Kiểm tra mẫu có bị phân lớp sai không
# ==========================================

if y_pred != y:

    print("Mẫu bị phân lớp sai.")

    # ======================================
    # 4. Cập nhật trọng số
    # ======================================

    for i in range(len(w)):
        w[i] = w[i] + eta * y * x[i]

    print("Trọng số sau cập nhật =", w)

else:

    print("Mẫu được phân lớp đúng.")


# ==========================================
# 5. Tính lại w^T x sau cập nhật
# ==========================================

wx_new = 0

for i in range(len(w)):
    wx_new = wx_new + w[i] * x[i]

print("w^T x sau cập nhật =", wx_new)


# ==========================================
# 6. Xác định lại nhãn dự đoán
# ==========================================

if wx_new >= 0:
    y_pred_new = 1
else:
    y_pred_new = -1

print("Nhãn dự đoán sau cập nhật =", y_pred_new)