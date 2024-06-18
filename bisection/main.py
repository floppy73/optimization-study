import sys
from functions import *
import matplotlib.pyplot as plt

a = 0  # 初期範囲の下限
b = 1000  # 初期範囲の上限
eps = 0.00001  # 終了条件
max_itr = 1000  # 最大の繰り返し回数

# 分割比ごとの比較
xopt_bi, m_list_bi, fm_list_bi = Bisection(ObjFunction2, a, b, eps, max_itr)
xopt_gold, m_list_gold, fm_list_gold = Goldensection(ObjFunction2, a, b, eps, max_itr)
xopt_quat, m_list_quat, fm_list_quat= Quatsection(ObjFunction2, a, b, eps, max_itr)

# 最終的に求められた値と各イテレーションでのxの差
m_bi_diff = [abs(xopt_bi - x) for x in m_list_bi]
m_gold_diff = [abs(xopt_gold - x) for x in m_list_gold]
m_quat_diff = [abs(xopt_quat - x) for x in m_list_quat]

# 最終的に求められた値と各イテレーションでのxの差のプロット
print(xopt_bi, xopt_gold, xopt_quat)
elm = max(len(m_bi_diff), len(m_gold_diff), len(m_quat_diff))
plt.plot(m_bi_diff, label="Bisection")
plt.plot(m_gold_diff, label="Golden number:1")
plt.plot(m_quat_diff, label="3:1")
plt.xlim(0, elm-1)
plt.yscale("log")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Difference")
plt.show()
