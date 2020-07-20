'''
Date:15/01/16; Author: Michael A. Silva & Christopher Morrison;

Module to calculate optical forces acting on target using method from:
"Influence of absorption on optical trapping force of spherical particles in a focused Gaussian beam"-Yanfeng Zhang et. al. &
"Radiation forces on spheres in loosely focused Gaussian beam: ray-optics regime"-Sang Bok Kim and Sang Soo Kim.
'''

import numpy as np
import sys
import matplotlib.pyplot as plt
import input
from scipy import integrate
from scipy.integrate import tplquad,dblquad,quad
input.medium_n = 1
input.target_n = 1.5
def snell(theta1):
    '''
    snells(theta1)
        Calculates theta2 from snells law in radians.

    Parameters
    ----------
    theta1: float
        Angle in to refracting surface.

    Returns
    -------
    theta2: float
        Angle out of refracting surface.

    inputut Script Parameters
    ---------------------------
    medium_n: float
        Refractive index of the surrounding medium.
    target_n: float
        Refractive index of target.

    '''
    theta2 = np.arcsin((input.medium_n / input.target_n) * np.sin(theta1))
    return theta2


def R1(theta1):
    '''
    reflectance(theta1,theta2)
        Calculates the Fresnel power reflectivity as a function of theta1 and theta2.

    Parameters
    -------------
    theta1: float
        Angle in to refracting surface.
    theta2: float
        Angle out of refracting surface.

    Returns
    ---------
    R: float
        Fresnel power reflectance as a function of theta1 and theta2.

    inputut Script Parameters
    ---------------------------
    polarisation: string
        Polarisation of laser beam used.
     medium_n: float
        Refractive index of the surrounding medium.
    target_n: float
        Refractive index of target.

    '''
    polarisation = 'p'
    theta2 = snell(theta1)
    if polarisation == 's':
        rs = (input.medium_n * np.cos(theta1) - (input.target_n * np.cos(theta2))) / (
                    input.medium_n * np.cos(theta1) + (input.target_n * np.cos(theta2)))
        R = rs ** 2
        return R
    elif polarisation == 'p':
        rp = (input.medium_n * np.cos(theta2) - input.target_n * np.cos(theta1)) / (
                    input.medium_n * np.cos(theta2) + input.target_n * np.cos(theta1))
        R = rp ** 2
        return R
    else:
        sys.exit("Polarisation is not defined it must be set to 'p','un_kim', 'un_gauthier' or 's' check inputut file")

def T1(theta1):
    return 1-R1(theta1)

def F():
    '''
    Calculate the scattering force.
    :return total_forcr: list
        The total_foce contains two element, the first one is the force along x-aixs, and the second one is the force along y-axis.
    '''

    plt.figure(1)
    maxtheta = np.pi/2
    Qs = []
    Qg = []
    theta = np.linspace(0, maxtheta, 1000)
    for i in range(len(theta)):
        theta1 = theta[i]
        theta2 = snell(theta1)  # Refracted angle

        Q_s = 1 + R1(theta1)*np.cos(2 * theta1) - (T1(theta1) ** 2 * (np.cos(2 * theta1 - 2 * theta2) + R1(theta1) * np.cos(2 * theta1))) / (
                                                        1 + R1(theta1) ** 2 + 2 * R1(theta1) * np.cos(2 * theta2))
        Q_g =R1(theta1)*np.sin(2*theta1)-(T1(theta1)**2*(np.sin(2*theta1-2*theta2)+R1(theta1)*np.sin(2*theta1)))/(1+R1(theta1)**2+2*R1(theta1)*np.cos(2*theta2))
        Qs.append(Q_s)
        Qg.append(abs(Q_g))
    plt.plot(theta, Qs, 'k', label='Q_s')
    plt.plot(theta, Qg, 'g', label='Q_g')
    plt.title('Ashkin factor Q changes with incident angle', fontsize = 14)
    plt.xlabel('Incident angle in rad', fontsize = 14)
    plt.ylabel('Ashkin factor Q', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.show()


if __name__ == '__main__':
    F()
    def Q_g(theta1):
        theta2 = snell(theta1)
        return abs(R1(theta1) * np.sin(2 * theta1) - (
                T1(theta1) ** 2 * (np.sin(2 * theta1 - 2 * theta2) + R1(theta1) * np.sin(2 * theta1))) / (
                      1 + R1(theta1) ** 2 + 2 * R1(theta1) * np.cos(2 * theta2)))
    def Q_s(theta1):
        theta2 = snell(theta1)
        return 1 + R1(theta1) * np.cos(2 * theta1) - (
                    T1(theta1) ** 2 * (np.cos(2 * theta1 - 2 * theta2) + R1(theta1) * np.cos(2 * theta1))) / (
                      1 + R1(theta1) ** 2 + 2 * R1(theta1) * np.cos(2 * theta2))


    val1, err1 = quad(lambda x: Q_g(x), 0, np.pi / 2)
    val2,err2 = quad(lambda x: Q_s(x),0,np.pi/2)

    print('Integrate Q_g', val1)
    print ('Integrate Q_s：',val2)
