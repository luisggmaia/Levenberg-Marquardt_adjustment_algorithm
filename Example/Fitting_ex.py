import numpy as np
import matplotlib.pyplot as plt
from Levenberg_Marquardt_algorithm_v2 import Levenberg_Marquardt as lma
import Functions as func

data = np.loadtxt('ex_data.txt')

t = data[:, 0]
t -= t[0]
y = data[:, 1]

beta_initial_guess = [-0.32, -0.013, 0.04, 0.84, 1.2] # (ym, dy, gamma, omegal, phi)

LM = lma(func.odf, t, y, beta_initial_guess, grad_f = func.grad_odf, Var = 0.000001)
optimized_beta = LM.set_min_beta( )

y_fitted = func.odf(t, optimized_beta)

plt.plot(t, y, label = "Measured points")
plt.plot(t, y_fitted, label = "Fitted curve")
plt.legend( )
plt.grid(True)
plt.title("Oscillation fitted curve")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.show( )