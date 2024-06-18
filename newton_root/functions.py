# 二分法
def Bisection(func, a, b, eps, max_itr):
  itr = 0
  m_list = []
  fm_list = []

  while abs(a - b) > eps:
    fa = func(a)

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
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
    
  print("bi iterations:" + str(itr))
  return m, m_list, fm_list


# はさみうち法
def FalsePosition(func, a, b, eps, max_itr):
  itr = 0
  m_list = []
  fm_list = []

  while True:
    fa = func(a)
    fb = func(b)

    m = a - ((b - a) / (fb - fa)) * fa 
    m_list.append(m)
    fm = func(m)
    fm_list.append(fm)

    if abs(fm) < eps:
      break

    if fa * fm < 0:
      b = m
    else:
      a = m
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
    
  print("fp iterations:" + str(itr))
  return m, m_list, fm_list


# セカント法
def Secant(func, x0, x1, eps, max_itr):
  itr = 0
  x_list = []
  fx_list = []

  while True:
    fx0 = func(x0)
    fx1 = func(x1)

    x2 = x1 - ((x1 - x0) / (fx1 - fx0)) * fx1 
    x_list.append(x2)
    fx2 = func(x2)
    fx_list.append(fx2)

    if fx2 == 0:
      break

    if abs(fx2) < eps:
      break
    else:
      x0 = x1
      x1 = x2
  
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break
    
  print("secant iterations:" + str(itr))
  return x2, x_list, fx_list


# ニュートン法
def Newton(func, diff_func, x0, eps, max_itr):
  itr = 0
  x_list = []
  fx_list = []

  while True:
    fx0 = func(x0)
    d_fx0 = diff_func(x0)

    x1 = x0 - (fx0 / d_fx0)
    x_list.append(x1)
    fx1 = func(x1)
    fx_list.append(fx1)

    if abs(fx1) < eps:
      break
    else:
      x0 = x1
  
    itr += 1

    if itr >= max_itr:
      print('reached max iterations')
      break

  print("nt iterations:" + str(itr))
  return x1, x_list, fx_list