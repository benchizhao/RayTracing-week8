
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from Rays import RayTracing
import timeit

start = timeit.default_timer()




radius = 4
initial_slope = 0
Ray_type = 'f'
def bundle(no_rays):
    '''
    Generate the rays and its path.
    :param no_rays: int
        How many rays we would like to trace
    :return: bundle_forward, bundle_backward
        ray state of refraction and reflection.
    '''
    width = 5
    distribution = np.linspace(-width, width, no_rays)
    bundle_forward = []
    bundle_backward = []
    for i in distribution:
        RT = RayTracing(0, i, initial_slope)
        RT.ray(Ray_type)
        RT.free_propagate(10)
        # RT.lens(15)
        # RT.free_propagate(10)
        # RT.lens(-15)
        # RT.mirror()
        # RT.flat_interface()
        RT.curved_interface(radius)
        # RT.free_propagate(20)
        # RT.free_propagate(30)

        bundle_forward.append(RT.state)
        bundle_backward.append(RT.deflection)
    return bundle_forward, bundle_backward

def plot(rays):
    '''
    plot_ray(self)
        Plot the ray path which is described in self.state and self.deflection.
    '''
    plt.figure('ABCD')
    Ray_thickness = 1
    forwards = rays[0]
    backwards = rays[1]
    for j in range(len(forwards)):
        for i in range(len(forwards[j])-1):
            x_1 = np.linspace(forwards[j][i][0], forwards[j][i + 1][0])
            y_1 = np.linspace(forwards[j][i][1], forwards[j][i + 1][1], len(x_1))
            plt.plot(x_1, y_1, 'k-', linewidth=forwards[j][i][3] * Ray_thickness)

            x_2 = np.linspace(backwards[j][i][0], backwards[j][i + 1][0])
            y_2 = np.linspace(backwards[j][i][1], backwards[j][i + 1][1], len(x_2))
            plt.plot(x_2, y_2, 'k-', linewidth=backwards[j][i][3] * Ray_thickness)
            # backwards[j][i][3] * Ray_thickness)

    x_main = np.linspace(0, max(x_1))
    y_main = [0] * len(x_main)
    plt.xlim(0, max(x_1)+3)
    plt.ylim(-8,20)
    plt.plot(x_main, y_main, '--',linewidth=0.4)

    # Plot circle
    theta = np.linspace(math.pi/2,  3*math.pi/2)
    x_c = 4 * np.cos(theta) + 10
    y_c = 4 * np.sin(theta)
    plt.plot(x_c, y_c)
    plt.show()

def make_table(n):
    for i in range(n):
        data_1 = bundle(n)[0][i]
        data_2 = bundle(n)[1][i]
        df_1 = pd.DataFrame(data_1, columns=list('ABCD'))
        df_2 = pd.DataFrame(data_2, columns=list('ABCD'))
        print('Ray', i+1)
        print(df_1)
        print(df_2)

def cut_intensity(n):
    plt.figure(2)
    y = []
    intens = []
    for i in range(n):
        data_1 = bundle(n)[0][i]
        data_2 = bundle(n)[1][i]
        for j in range(len(data_1)):
            pass

make_table(11)


stop = timeit.default_timer()

print('Time: ', stop - start)
plot(bundle(11))