from function import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import time
import scipy.optimize

# パラメータ
epsilon_1 = 1e-5
max_iterations = 2000
x_init = np.full(2, 2.0)

# 時間測定
start = time.perf_counter()

x_opt, x_history, fx_history = quasi_newton(x_init, scipy.optimize.rosen, epsilon_1, max_iterations)

end = time.perf_counter()
print("time: " + str(end - start))

# 各イテレーションにおけるf(x)
x = np.arange(len(fx_history))
plt.plot(x, fx_history)
plt.xlim(0, len(fx_history)-1)
plt.yscale("log")
plt.xlabel("Iterations")
plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
plt.ylabel('$f(x)$')
plt.show() 
plt.clf()

# 各イテレーションにおけるx
x = np.arange(len(fx_history))
plt.plot(x, x_history[:, 0], label="$x_0$")
plt.plot(x, x_history[:, 1], label="$x_1$")
plt.xlim(0, len(fx_history)-1)
plt.xlabel("Iterations")
plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
plt.ylabel('$x_n$')
plt.legend()
plt.show()

# 等高線上でのxの動きのプロット（変数が2次元のとき）
x_grid = np.linspace(min(x_history[:, 0])-1, max(x_history[:, 0])+1, 100)
y_grid = np.linspace(min(x_history[:, 1])-1, max(x_history[:, 1])+1, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
fx = scipy.optimize.rosen(np.array((x_mesh, y_mesh)))
plt.contourf(x_mesh, y_mesh, fx, levels=15, cmap="Blues", norm=LogNorm())
plt.colorbar()
plt.plot(x_history[:, 0], x_history[:, 1], "-o", color="r")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
plt.clf()
