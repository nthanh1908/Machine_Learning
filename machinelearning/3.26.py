# Bài 3.26 - Gradient Descent

# Hàm số f(x) = x^2 - 4x + 5
def f(x):
    return x**2 - 4*x + 5


# Đạo hàm f'(x) = 2x - 4
def grad(x):
    return 2*x - 4


# Điểm khởi tạo
x = 5

# Learning rate
eta = 0.2


# In thông tin ban đầu
print("Hàm số: f(x) = x^2 - 4x + 5")
print("Đạo hàm: f'(x) = 2x - 4")
print("Điểm khởi tạo x(0) =", x)
print("Learning rate =", eta)

print("\nKết quả Gradient Descent:")
print("----------------------------------------")


# Thực hiện 4 bước cập nhật
for i in range(4):

    # Tính đạo hàm tại x hiện tại
    g = grad(x)

    # Tính giá trị hàm tại x hiện tại
    fx = f(x)

    # Công thức Gradient Descent
    x_new = x - eta * g

    # Tính giá trị hàm tại x mới
    f_new = f(x_new)

    # In kết quả
    print("Bước", i + 1)
    print("  x hiện tại =", x)
    print("  f'(x) =", g)
    print("  f(x) =", fx)
    print("  x mới =", x_new)
    print("  f(x mới) =", f_new)
    print("----------------------------------------")

    # Cập nhật x
    x = x_new