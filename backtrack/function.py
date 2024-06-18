import numpy as np
import matplotlib.pyplot as plt
import math

# 目的関数1（Rosenbrock function）
def test_func1(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2


# 目的関数2
def test_func2(x):
    return 3*(x[0]-3)**2 + (x[1]-5)**2


# 目的関数3
def test_func3(x):
    a = 10 + (x[0]**2 - 10*np.cos(2*np.pi*x[0])) + 10 + (x[1]**2 - 10*np.cos(2*np.pi*x[1]))
    return a


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


# 最急降下法
def steepest_descent(x, func, lr, epsilon_1, epsilon_2, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    iterations = 0
    x_grad = numerical_gradient(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1
        x_grad = numerical_gradient(func, x)
        print(x_grad)

        x_next = x - lr * x_grad
        fx_history.append(func(x_next))
        
        if abs(np.linalg.norm(x_next-x, ord=2)) < epsilon_2:
            print("The objective function is flat")
            break

        if iterations >= max_iterations:
            print("Reached max iterations")
            break
        
        x = x_next
        x_history.append(x)

    x_history = np.array(x_history)
    print(iterations)
    return x, x_history, fx_history


# 黄金分割法でステップ幅を決定する最急降下法
def steepest_descent_golden(x, func, epsilon_1, epsilon_2, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    alpha_history = []
    iterations = 0
    x_grad = numerical_gradient(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1
        x_grad = numerical_gradient(func, x)

        alpha = golden_section(func, x, x_grad, 1)
        alpha_history.append(alpha)

        if iterations == 1:
            plot_cross_section(func, x, alpha, x_grad)
        
        x_next = x - alpha * x_grad
        fx_history.append(func(x_next))
        

        if abs(np.linalg.norm(x_next-x, ord=2)) < epsilon_2:
            print("The objective function is flat")
            break

        if iterations >= max_iterations:
            print("Reached max iterations")
            break
        
        x = x_next
        x_history.append(x)

    x_history = np.array(x_history)
    print("iterations:" + str(iterations))
    return x, x_history, fx_history, alpha_history


# バックトラック法を用いた最急降下法
def steepest_descent_backtrack(x, func, epsilon_1, epsilon_2, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    alpha_history = []
    iterations = 0
    rho = 0.6
    beta = 0.2
    x_grad = numerical_gradient(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1
        x_grad = numerical_gradient(func, x)

        alpha = backtrack(func, x, x_grad, 1, beta, rho)
        alpha_history.append(alpha)

        if iterations == 1:
            plot_cross_section(func, x, alpha, x_grad)
        
        x_next = x - alpha * x_grad
        fx_history.append(func(x_next))
        

        if abs(np.linalg.norm(x_next-x, ord=2)) < epsilon_2:
            print("The objective function is flat")
            break

        if iterations >= max_iterations:
            print("Reached max iterations")
            break
        
        x = x_next
        x_history.append(x)

    x_history = np.array(x_history)
    print("iterations:" + str(iterations))
    return x, x_history, fx_history, alpha_history


# バックトラック法によるステップ幅決定
def backtrack(func, x, x_grad, alpha=1, beta=1e-4, rho=0.5):
    while func(x - alpha*x_grad) > (func(x) - beta*alpha*np.dot(x_grad, x_grad)):
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


# 黄金分割法
def golden_section(func, xk, x_grad, alpha_max=1):
    max_itr = 1000
    iterations = 0
    gold = (1 + math.sqrt(5)) / 2
    x0 = 0
    x3 = alpha_max
    x1 = (x0*gold + x3) / (1 + gold)
    x2 = (x0 + x3*gold) / (1 + gold)

    while x3 - x0 > 1e-6:
        iterations += 1
        if func(xk - x1*x_grad) < func(xk - x2*x_grad):
        #if func(x1) < func(x2):
            x3 = x2
            x2 = x1
            x1 = (x0*gold + x3) / (1 + gold)
        else:
            x0 = x1
            x1 = x2
            x2 = (x0 + x3*gold) / (1 + gold)

        if iterations >= max_itr:
            break

    return x1

        

    