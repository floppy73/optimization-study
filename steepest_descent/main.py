from function import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

# パラメータ
epsilon_1 = 1e-5
epsilon_2 = 1e-7
max_iterations = 10000
lr = 1e-5
x_init = np.array([1.5, 0.7])

"""ステップ幅固定時"""
x_opt, x_history, fx_history = steepest_descent(x_init, test_func1, lr, epsilon_1, epsilon_2, max_iterations)

"""バックトラック法利用時"""
#x_opt, x_history, fx_history = steepest_descent_backtrack(x_init, test_func1, epsilon_1, epsilon_2, max_iterations)

# 各イテレーションでのf(x)をプロット
x = np.arange(len(fx_history))
plt.plot(x, fx_history)
plt.xlim(0, len(fx_history))
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel('$f(x)$')
plt.show() 
plt.clf()

# 等高線上でのxの動きのプロット
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

