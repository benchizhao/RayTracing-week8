import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from NonABCD import RayTracing
import NewNonABCD
import input
import timeit


start = timeit.default_timer()
radius = 10
initial_degree = 0
Ray_type = 'f' #'g' for guassian; 'f' for flat
width = 40
number = 10

sigma = 15 #gaussian laser

lens_pos = 30
lens_f = 200
len_thickness = 20

cir_pos = 150



def bundle(no_rays):
    '''
    Generate the rays and its path.
    :param no_rays: int
        How many rays we would like to trace
    :return: bundle_forward, bundle_backward
        ray state of refraction and reflection.
    '''

    distribution = np.linspace(-width, width, no_rays)
    bundle_forward = []
    bundle_backward = []
    for i in distribution:
        RT = RayTracing(0, i, initial_degree)
        RT.ray(Ray_type,sigma)
        RT.lens_trace(f=lens_f, pos=lens_pos, thick=len_thickness)  # cm
        RT.free_propagate(10)
        RT.propagate(140)

        RT.circle(radius, cir_pos)
        RT.free_propagate(10)
        RT.propagate(100)

        bundle_forward.append(RT.state)
        bundle_backward.append(RT.deflection)
    return bundle_forward, bundle_backward


def make_table(n):
    for i in range(n):
        data_1 = bundle(n)[0][i]
        data_2 = bundle(n)[1][i]
        print(np.shape(data_1))
        df_1 = pd.DataFrame(data_1, columns=['x', 'y', 'theta', 'intensity'])
        df_2 = pd.DataFrame(data_2, columns=['x', 'y', 'theta', 'intensity'])
        print('Ray', i + 1)
        print(df_1)
        print(df_2)


def plot(rays):
    '''
    plot_ray(self)
        Plot the ray path which is described in self.state and self.deflection.
    '''
    plt.figure('Non-ABCD')
    Ray_thickness = 1
    forwards = rays[0]
    backwards = rays[1]
    for j in range(len(forwards)):
        for i in range(len(forwards[j]) - 1):
            x_1 = np.linspace(forwards[j][i][0], forwards[j][i + 1][0])
            y_1 = np.linspace(forwards[j][i][1], forwards[j][i + 1][1], len(x_1))
            plt.plot(x_1, y_1, 'k-', linewidth=forwards[j][i][3] * Ray_thickness)

            # x_2 = np.linspace(backwards[j][i + 1][0], forwards[j][i][0])
            # y_2 = np.linspace(backwards[j][i + 1][1], forwards[j][i][1], len(x_2))
            # plt.plot(x_2, y_2, 'k-', linewidth=backwards[j][i][3] * Ray_thickness)

    x_main = np.linspace(0, max(x_1))
    y_main = [0] * len(x_main)
    plt.xlim(0, max(x_1) + 3)
    plt.ylim(-70, 70)
    plt.plot(x_main, y_main, '--', linewidth=0.4)

    # Plot circle
    theta = np.linspace(0, 2 * math.pi)
    x_c = radius * np.cos(theta) + cir_pos
    y_c = radius * np.sin(theta)
    plt.plot(x_c, y_c)

    #plot lens
    theta1 = np.linspace(3 * math.pi/4, 5*math.pi/4)
    theta2 = np.linspace(- math.pi / 4,  math.pi / 4)
    x_1 = lens_f * np.cos(theta1) + lens_pos + lens_f - len_thickness / 2
    x_2 = lens_f * np.cos(theta2) + lens_pos - lens_f + len_thickness / 2
    y_1 = lens_f * np.sin(theta1)
    y_2 = lens_f * np.sin(theta2)
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)

    plt.show()
#

def detector(x_position):
    plt.figure('cross_section intensity')
    y_values = []
    intns_values = []
    for i in range(number):
        data_1 = bundle(number)[0][i]
        refract_x = [i[0] for i in data_1]
        refract_y = [i[1] for i in data_1]
        refract_theta = [i[2] for i in data_1]
        refract_intensity = [i[3] for i in data_1]

        for j in range(len(refract_x)):
            if x_position > refract_x[j] and x_position < refract_x[j + 1]:
                y = np.tan(np.deg2rad(refract_theta[j])) * (x_position - refract_x[j]) + refract_y[j]
                y_values.append(y)
                intns_values.append(refract_intensity[j])
        # data_2 = bundle(number)[1][i]
    plt.plot(y_values, intns_values, 'k.', ms=1)


start = timeit.default_timer()
# time the programme
# detector(10)
# detector(23)
# detector(16)
make_table(number)

stop = timeit.default_timer()
print('Time: ', stop - start)
# plt.show()
# print the running time
plot(bundle(number))


