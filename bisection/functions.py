import random
import math

# 目的関数1（Rosenbrock function）
def ObjFunction1(x):
  return x*x - 5*x + 4


# 目的関数2
def ObjFunction2(x):
  temp = 0
  for i in range(6):
    temp += math.cos(i * x)
  return x * temp - 3


# 初期値を自動決定する関数
def SelectInitialNum(func, max, min, max_itr):
  itr = 0
  while True:
    itr += 1
    n, m = random.uniform(min, max), random.uniform(min, max)
    if func(n) * func(m) < 0:
      if n < m:
        a = n
        b = m
      else:
        a = m
        b = n
      break
    if itr == max_itr:
      print("reached max iteration")
      a, b = None, None
      break

  return a, b


# 二分法
def Bisection(func, a, b, eps, max_itr):
  itr = 0
  m_list = []
  fm_list = []

  while abs(a - b) > eps:
    fa = func(a)
    fb = func(b)

    m = (a + b) / 2
    m_list.append(m)
    fm = func(m)
    fm_list.append(fm)

    if fm == 0:
      break

    if fa * fm < 0:
      b = m
    else:
      a = m
    #print(itr, fa, fm, a, b)
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
    
  print("bi iterations:" + str(itr))
  return m, m_list, fm_list


# 分割比が黄金比の二分法
def Goldensection(func, a, b, eps, max_itr):
  itr = 0
  m_list = []
  fm_list = []

  while abs(a - b) > eps:
    fa = func(a)
    gold = (1 + math.sqrt(5)) / 2

    m = (a + b * gold) / (1 + gold)
    m_list.append(m)
    fm = func(m)
    fm_list.append(fm)

    if fm == 0:
      break

    if fa * fm < 0:
      b = m
    else:
      a = m
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
  
  print("gold iterations:" + str(itr))
  return m, m_list, fm_list


# 分割比が1:4の二分法
def Quatsection(func, a, b, eps, max_itr):
  itr = 0
  m_list = []
  fm_list = []

  while abs(a - b) > eps:
    fa = func(a)

    m = (a + b * 3) / 4
    m_list.append(m)
    fm = func(m)
    fm_list.append(fm)

    if fm == 0:
      break

    if fa * fm < 0:
      b = m
    else:
      a = m
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
  
  print("quat iterations:" + str(itr))
  return m, m_list, fm_list
