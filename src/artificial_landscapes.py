import numpy as np

# Rastrigin function


def rastrigin(x1, x2):
    """_summary_

    Args:
        x1 (_type_): _description_
        x2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = 2
    A = x1**2 - 10*np.cos(2*np.pi*x1)
    B = x2**2 - 10*np.cos(2*np.pi*x2)
    return 10*n + A + B


def ackley(x1, x2):
    A = np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    B = np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return -20.0 * A - B + np.exp(1) + 20


def rosenbrock(x1, x2):
    a = 1
    b = 100
    return (a - x1)**2 + b*(x2 - x1**2)**2


def beale(x1, x2):
    A = (1.5 - x1 + x1*x2)**2
    B = (2.25 - x1 + x1*x2**2)**2
    C = (2.625 - x1 + x1*x2**3)**2
    return A + B + C
