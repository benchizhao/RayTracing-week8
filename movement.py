import numpy as np
import input
import Radiation_Force
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import timeit
import pickle



def mass():
    r = input.radius * 1E-6
    V = 4 / 3 * np.pi * r ** 3
    m = input.density * V
    return m

def gravity():
    N = mass()*input.g
    return np.array([-N, float(0.)])

def total_force(ray_state):
    g_f = gravity()
    rad_f = Radiation_Force.radiation_force(ray_state)
    return g_f + rad_f

def acc(ray_state):
    return total_force(ray_state)/mass()

def stable_point(rang):
    x = np.linspace(rang[0],rang[1],150)
    minimum = 10
    plt.figure(3)
    for i in x:
        a = acc([i,0,0,0])[0]
        print(i, a)
        plt.plot(i,a,'.')
        if abs(a) < minimum:
            minimum = a
            idex = i
    plt.title('Acceleration of the droplet when moving on x-axis')
    plt.xlabel('x position of droplet')
    plt.ylabel('Acceleration on x axis')
    plt.show()
    return idex, minimum

def diff2(d_list, t):
    x, y, vx, vy = d_list
    ax = acc(d_list)[0]
    ay = acc(d_list)[1]
    print(t)
    return np.array([vx, vy, ax, ay])


if __name__ == '__main__':
    # idex, a = stable_point([700,900])
    # print(idex, a)



    # start = timeit.default_timer()
    # # print(acc(input.droplet_pos))
    # t = np.linspace(0, 15, 300)
    # result = odeint(diff2, input.droplet_pos, t)
    # # ave data
    # f = open('t=[0,15],400rays.txt', 'wb')
    # pickle.dump(result, f)
    # f.close()
    # read data from
    f = open('t=[0,15],400rays.txt', 'rb')
    d = pickle.load(f)
    f.close()
    print(d)
    #
    t = np.linspace(0, 15, 300)
    plt.figure('x_position')
    plt.plot(t, d[:, 0])
    plt.title('x-position of droplet')
    plt.xlabel('Time')
    plt.ylabel('x-axis')
    plt.show()

    plt.figure('y_position')
    plt.plot(t, d[:, 1], '.')
    plt.title('y-position of droplet')
    plt.xlabel('Time')
    plt.ylabel('y-axis')
    plt.show()

    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

