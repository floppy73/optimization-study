import math

# 目的関数1
def ObjFunction1(x):
    return x*x - 5*x + 4

# 目的関数2
def ObjFunction2(x):
    temp = 0
    for i in range(6):
        temp += math.cos(i * x)
    return x * temp - 3
