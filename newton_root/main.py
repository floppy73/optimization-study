import numpy as np
from functions import *
from objfunction import *
import matplotlib.pyplot as plt

# パラメータ設定
a = -0.1
b = 3
eps = 0.00001
max_itr = 100

# 二分法，はさみうち法，セカント法，ニュートン法でそれぞれ求根
xopt_bi, m_list_bi, fm_list_bi = Bisection(ObjFunction1, a, b, eps, max_itr)
xopt_fp, m_list_fp, fm_list_fp = FalsePosition(ObjFunction1, a, b, eps, max_itr)
xopt_sc, m_list_sc, fm_list_sc = Secant(ObjFunction1, a, b, eps, max_itr)
xopt_nt, m_list_nt, fm_list_nt = Newton(ObjFunction2, DiffObjFunction2, 10, eps, max_itr)

# 最終的な解と各イテレーションでのxとの差
m_bi_diff = [abs(x - xopt_bi) for x in m_list_bi]
m_fp_diff = [abs(x - xopt_fp) for x in m_list_fp]
m_sc_diff = [abs(x - xopt_sc) for x in m_list_sc]
m_nt_diff = [abs(x - xopt_nt) for x in m_list_nt]

# 収束の様子をプロット
elm = max(len(m_bi_diff), len(m_fp_diff), len(m_sc_diff), len(m_nt_diff))
plt.plot(m_bi_diff, '-o', label="Bisection")
plt.plot(m_fp_diff, '-o', label="False Position")
plt.plot(m_sc_diff, '-o', label="Secant")
plt.plot(m_nt_diff, '-o', label="Newton")
plt.xticks(np.arange(0, 100, 5))
plt.xlim(0, elm)
plt.yscale("log")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Difference")
plt.show()
