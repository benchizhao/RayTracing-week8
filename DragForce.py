import input
import numpy as np

def drag_force(v):
    C_d = 0.47
    return 0.5*input.rho*v**2*C_d * np.pi*input.radius**2