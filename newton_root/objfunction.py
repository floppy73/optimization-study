import math

# 目的関数1
def ObjFunction1(x):
    return x*x - 5*x + 4


# 目的関数1の一階微分
def DiffObjFunction1(x):
    return 2*x - 5


# 目的関数2
def ObjFunction2(x):
    temp = 0
    for i in range(6):
        temp += math.cos(i * x)
    return x * temp - 3


# 目的関数2の一階微分
def DiffObjFunction2(x):
    return NumericalDiff(ObjFunction2, x)


# 目的関数3
def ObjFunction3(x):
    if x >= 0:
        y = math.sqrt(x)+1
    else:
        y = -x**2+1
    return y


# 目的関数3の一階微分
def DiffObjFunction3(x):
    return NumericalDiff(ObjFunction3, x)


# 数値微分
def NumericalDiff(func, x):
    h = 1e-4
    return (func(x+h) - func(x-h)) / (2*h)
