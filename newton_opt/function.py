import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# 目的関数（Rosenbrock function）
def obj_rosen(x):
    return scipy.optimize.rosen(x)


# 数値微分
def numerical_gradient(func, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for x_id in range(x.size):
        val = x[x_id]
        x[x_id] = val + h
        f_plus = func(x)

        x[x_id] = val - h
        f_minus = func(x)

        x[x_id] = val

        grad[x_id] = (f_plus - f_minus) / (2*h)

    return grad


# ヘッセ行列の数値計算
def hessian(func, x):
    n = x.size
    h = 1e-4
    hess = np.zeros((n, n))
    for i in range(n):
        val = x[i]
        x[i] = val + 2*h
        f_plus = func(x)

        x[i] = val - 2*h
        f_minus = func(x)

        x[i] = val

        hess[i][i] = (f_plus - 2*func(x) + f_minus) / (4*(h**2))

        for j in range(i+1, n):
            i_val = x[i]
            j_val = x[j]

            x[i] = i_val + h
            x[j] = j_val + h
            f_xip_xjp = func(x)

            x[j] = j_val - h
            f_xip_xjm = func(x)

            x[i] = i_val - h
            f_xim_xjm = func(x)

            x[j] = j_val + h
            f_xim_xjp = func(x)

            x[i] = i_val
            x[j] = j_val

            spd = (f_xip_xjp - f_xip_xjm - f_xim_xjp + f_xim_xjm) / (4*(h**2))
            hess[i][j] = hess[j][i] = spd

    return hess


# ニュートン法
def newton(x, func, epsilon_1, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    iterations = 0
    x_grad = numerical_gradient(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1
        # 勾配，ヘッセ行列，逆行列の計算
        x_grad = numerical_gradient(func, x)
        x_hess = hessian(func, x)
        x_hess_inv = np.linalg.inv(x_hess)

        # 次のxの決定
        x_next = x - np.dot(x_hess_inv,x_grad)
        fx_history.append(func(x_next))
        
        if iterations >= max_iterations:
            print("Reached max iterations")
            break
        
        x = x_next
        x_history.append(x)

    x_history = np.array(x_history)
    print("iterations:" + str(iterations))
    return x, x_history, fx_history


# バックトラック法
def backtrack(func, x, x_grad, x_hess_inv, alpha=1, beta=1e-4, rho=0.5):
    while func(x - alpha*x_hess_inv*x_grad) > (func(x) - beta*alpha*np.dot(x_hess_inv*x_grad, x_grad)):
            alpha *= rho

    return alpha


# 横軸にステップ幅，縦軸に次ステップのf(x)をプロットした断面図のプロット
def plot_cross_section(func, xk, alpha, x_grad):
    x = np.linspace(0, 1, 1000)
    y = []
    for a in x:
        y.append(func(xk - a*x_grad))

    plt.plot(x, y, label=r"$f(x^{(1)}+\alpha p)$")
    plt.plot(alpha, func(xk - alpha*x_grad), marker=".", label="actual alpha")
    #plt.yscale("log")
    plt.xlabel("alpha")
    plt.ylabel(r"$f(x^{(2)})$")
    plt.legend()
    plt.savefig("el_cs_bt.eps")
    plt.show()
    plt.clf()
    