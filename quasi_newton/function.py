import numpy as np

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


# 準ニュートン法
def quasi_newton(x, func, epsilon_1, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    iterations = 0
    x_grad = numerical_gradient(func, x)
    i = np.identity(x.size)
    h = hessian(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1

        dk = - h @ x_grad
        alpha = backtrack(func, x, x_grad, dk)
        x_n = x + alpha * dk
        fx_history.append(func(x_n))
        x_history.append(x_n)

        # 次のhの決定
        x_grad_n = numerical_gradient(func, x_n)
        s = x_n - x
        y = x_grad_n - x_grad
        ys = y.T @ s
        h = (i - (s @ y.T) / ys).T @ h @ (i - (y @ s.T) / ys) + (s @ s.T) / ys

        x = x_n
        x_grad = x_grad_n
    
        if iterations >= max_iterations:
            print("Reached max iterations")
            break
        
    x_history = np.array(x_history)
    print("iterations:" + str(iterations))
    return x, x_history, fx_history


# バックトラック法
def backtrack(func, x, x_grad, d, alpha=1, beta=1e-2, rho=0.8):
    while func(x + alpha*d) > (func(x) + beta*alpha*np.dot(x_grad, d)):
            alpha *= rho

    return alpha