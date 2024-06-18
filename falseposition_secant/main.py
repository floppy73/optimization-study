import numpy as np
from functions import *
from objfunction import *
import matplotlib.pyplot as plt

a = 0
b = 1000
eps = 0.00001
max_itr = 1000

xopt_bi, m_list_bi, fm_list_bi = Bisection(ObjFunction2, a, b, eps, max_itr)
xopt_fp, m_list_fp, fm_list_fp = FalsePosition(ObjFunction2, a, b, eps, max_itr)
xopt_sc, m_list_sc, fm_list_sc = Secant(ObjFunction2, a, b, eps, max_itr)

print(xopt_bi, xopt_fp, xopt_sc)

m_bi_diff = [abs(x - xopt_bi) for x in m_list_bi]
m_fp_diff = [abs(x - xopt_fp) for x in m_list_fp]
m_sc_diff = [abs(x - xopt_sc) for x in m_list_sc]

elm = max(len(m_bi_diff), len(m_fp_diff), len(m_sc_diff))
plt.plot(m_bi_diff, '-o', label="Bisection")
plt.plot(m_fp_diff, '-o', label="False Position")
plt.plot(m_sc_diff, '-o', label="Secant")
plt.xticks(np.arange(0, 30, 2))
plt.xlim(0, elm)
plt.yscale("log")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Difference")
plt.show()
