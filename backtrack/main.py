from function import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

# パラメータ
epsilon_1 = 1e-5
epsilon_2 = 1e-8
max_iterations = 2000
lr = 1e-5
x_init = np.array([8.0, 8.0])

# 最急降下法
#x_opt, x_history, fx_history = steepest_descent(x_init, test_func1, lr, epsilon_1, epsilon_2, max_iterations)

 # バックトラック法を用いた最急降下法
x_opt, x_history, fx_history, alpha_history = steepest_descent_backtrack(x_init, test_func2, epsilon_1, epsilon_2, max_iterations)

# 黄金分割法を用いた最急降下法
#x_opt, x_history, fx_history, alpha_history = steepest_descent_golden(x_init, test_func2, epsilon_1, epsilon_2, max_iterations)

# 各イテレーションでのf(x)のプロット
x = np.arange(len(fx_history))
plt.plot(x, fx_history)
plt.xlim(0, len(fx_history))
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel('$f(x)$')
plt.show() 
plt.clf()

# 各イテレーションでのステップ幅のプロット
x = np.arange(len(alpha_history))
plt.plot(x, alpha_history)
plt.xlim(0, len(alpha_history))
plt.xlabel("Iterations")
plt.ylabel('alpha')
plt.show() 
plt.clf()


# 等高線プロット上でのxの動きのプロット
x_grid = np.linspace(min(x_history[:, 0])-1, max(x_history[:, 0])+1, 100)
y_grid = np.linspace(min(x_history[:, 1])-1, max(x_history[:, 1])+1, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
fx = test_func1(np.array((x_mesh, y_mesh)))
plt.contourf(x_mesh, y_mesh, fx, levels=15, cmap="Blues", norm=LogNorm())
plt.colorbar()
plt.plot(x_history[:, 0], x_history[:, 1], "-o", color="r")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
plt.clf()
