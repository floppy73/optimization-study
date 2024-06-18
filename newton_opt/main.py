from function import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import time

# パラメータ
epsilon_1 = 1e-5
max_iterations = 2000
lr = 1e-5
x_init = np.full(2, 5.0)


# 時間測定
start = time.perf_counter()

x_opt, x_history, fx_history = newton(x_init, obj_rosen, epsilon_1, max_iterations)

end = time.perf_counter()
print("time: " + str(end - start))

# 各イテレーションにおけるf(x)のプロット
x = np.arange(len(fx_history))
plt.plot(x, fx_history)
plt.xlim(0, len(fx_history)-1)
plt.yscale("log")
plt.xlabel("Iterations")
plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
plt.ylabel('$f(x)$')
plt.show() 
plt.clf()

# 各イテレーションにおけるx_iの値のプロット
x = np.arange(len(fx_history))
plt.plot(x, x_history[:, 0], label="$x_0$")
plt.plot(x, x_history[:, 19], label="$x_{19}$")
plt.plot(x, x_history[:, 39], label="$x_{39}$")
plt.plot(x, x_history[:, 59], label="$x_{59}$")
plt.plot(x, x_history[:, 79], label="$x_{79}$")
plt.plot(x, x_history[:, 99], label="$x_{99}$")
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
fx = obj_rosen(np.array((x_mesh, y_mesh)))
plt.contourf(x_mesh, y_mesh, fx, levels=15, cmap="Blues", norm=LogNorm())
plt.colorbar()
plt.plot(x_history[:, 0], x_history[:, 1], "-o", color="r")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
plt.clf()
