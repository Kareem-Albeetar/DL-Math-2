def derive(f, x, h=0.001):

    derivative = (f(x + h) - f(x)) / h
    return derivative