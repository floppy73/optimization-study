import numpy as np

# 目的関数1（Rosenbrock function）
def test_func1(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2


# 目的関数2
def test_func2(x):
    return 3*(x[0]-3)**2 + (x[1]-5)**2


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

# バックトラック法を用いた最急降下法
def steepest_descent_backtrack(x, func, epsilon_1, epsilon_2, max_iterations):
    fx_history = [func(x)]
    x_history = [x]
    iterations = 0
    rho = 0.5
    beta = 1e-4
    x_grad = numerical_gradient(func, x)

    while np.linalg.norm(x_grad, ord=2) > epsilon_1:
        iterations += 1
        x_grad = numerical_gradient(func, x)
        print(x_grad)

        alpha = backtrack(func, x, x_grad, 1, beta, rho)
        
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
    return x, x_history, fx_history


# バックトラック法
def backtrack(func, x, x_grad, alpha=1, beta=1e-4, rho=0.5):
    while func(x - alpha*x_grad) > (func(x) - beta*alpha*np.dot(x_grad, x_grad)):
            alpha *= rho

    return alpha