# -*- coding: utf-8 -*-
"""
Time: 2020/Jul/06

@author: Benchi Zhao
Python version: 3.7
Functionality: This module calculate the force caused by radiation pressure.
Dependence: Need import 'NewNonABCD', all parameters of the experiment is saved in that file 'input'.
"""

import sys
import numpy as np
import input
import NewNonABCD as NonABCD
import copy
import timeit
import matplotlib.pyplot as plt
import showAshkin as SA

all_forwards = []
all_backwards = []
def intensity(position):
    I = []
    for i in range(len(position)):
        intens = NonABCD.gaussian(position[i], input.sigma)
        I.append(intens)
    normalise_I = I / sum(I)
    return normalise_I

def bundle(ray_state):
    list_of_rays = np.linspace(-input.width, input.width, input.no_of_rays)
    for i in range(input.no_of_rays):
        NonABCD.ray(0, list_of_rays[i] + input.y_displacement, 0, intensity(list_of_rays)[i])
        NonABCD.lens_trace(input.lens_f, input.lens_pos, input.len_thickness)
        NonABCD.free_propagate(10)
        NonABCD.propagate(ray_state[0] - input.lens_pos - 10)
        NonABCD.circle(input.radius,ray_state)
        NonABCD.free_propagate(15)
        NonABCD.propagate(200)
        a = copy.deepcopy(NonABCD.forward)
        b = copy.deepcopy(NonABCD.backward)
        NonABCD.forward.clear()
        NonABCD.backward.clear()
        all_forwards.append(a)
        all_backwards.append(b)

def useful_data_for():
    '''
    Clean the data, some rays in all_forwards will not interact with the droplet, which are useless.
    This function will keep those rays interact with the droplet and delete those rays have no interaction with the droplet.
    '''
    useful_for = copy.deepcopy(all_forwards)
    for i in range(len(useful_for)-1,-1,-1): # iterate from right to left
        if len(useful_for[i]) != 12:
            useful_for.pop(i)
    return useful_for

def useful_data_back():
    '''
    Clean the data, some rays in all_backwards will not interact with the droplet, which are useless.
    This function will keep those rays interact with the droplet and delete those rays have no interaction with the droplet.
    '''
    useful_back = copy.deepcopy(all_backwards)
    for i in range(len(useful_back)-1,-1,-1): # iterate from right to left
        if len(useful_back[i]) != 12:
            useful_back.pop(i)
    return useful_back

def F_s(ray_state):
    '''
    Calculate the scattering force.
    :return total_forcr: list
        The total_foce contains two element, the first one is the force along x-aixs, and the second one is the force along y-axis.
    '''
    data = useful_data_for()
    force = []
    total_force = np.array([0, 0])
    # plt.figure()
    for i in range(len(data)):
        x = data[i][6][0]
        y = data[i][6][1]
        vec_1 = [x-ray_state[0], y-ray_state[1]]
        vec_2 = [-1, -np.tan(data[i][6][2])]
        unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
        unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        theta1 = np.arccos(dot_product)
        theta2 = NonABCD.snell(theta1)  # Refracted angle

        # print(data[i][6][2],theta1,theta2)
        P = input.power * data[i][6][3]
        F = input.medium_n * P / input.c*(1+NonABCD.R(theta1)*np.cos(2*theta1)- (NonABCD.T(theta1) ** 2 * (np.cos(2 * theta1 - 2 * theta2) + NonABCD.R(theta1) * np.cos(2 * theta1))) / (1 + NonABCD.R(theta1) ** 2 + 2 * NonABCD.R(theta1) * np.cos(2 * theta2)))
        # print(input.medium_n * P / input.c )
        Fx = abs(F) * np.cos(data[i][6][2])
        Fy = abs(F) * np.sin(data[i][6][2])
        # plt.plot(y,abs(F),'.')
        force.append([Fx, Fy])
        total_force = total_force + np.array([Fx, Fy])
    # plt.show()
    return total_force

def F_g(ray_state):
    '''
        Calculate the gradient force.
        :return total_forcr: list
            The total_foce contains two element, the first one is the force along x-aixs, and the second one is the force along y-axis.
        '''
    data = useful_data_for()
    total_force = np.array([0, 0])
    # print(np.shape(data))
    for i in range(len(data)):
        x = data[i][6][0]
        y = data[i][6][1]
        vec_1 = [x - ray_state[0], y - ray_state[1]]
        vec_2 = [-1, -np.tan(data[i][6][2])]
        unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
        unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        theta1 = np.arccos(dot_product)
        theta2 = NonABCD.snell(theta1) # Refracted angle

        # print(y, data[i][6][2]/np.pi*180, theta1/np.pi*180, theta2/np.pi*360)
        P = input.power * data[i][6][3]
        F = input.medium_n*P/input.c *(NonABCD.R(theta1)*np.sin(2*theta1)-(NonABCD.T(theta1)**2*(np.sin(2*theta1-2*theta2)+NonABCD.R(theta1)*np.sin(2*theta1)))/(1+NonABCD.R(theta1)**2+2*NonABCD.R(theta1)*np.cos(2*theta2)))
        if data[i][6][2] < np.pi:
            Fx = abs(F) * np.cos(data[i][6][2]-np.pi/2)
            Fy = abs(F) * np.sin(data[i][6][2]-np.pi/2)
        else:
            Fx = abs(F) * np.cos(data[i][6][2] + np.pi / 2)
            Fy = abs(F) * np.sin(data[i][6][2] + np.pi / 2)
        # print(y, Fx, Fy, i)
        total_force = total_force + np.array([Fx, Fy])
        # print(total_force)
    return total_force

def radiation_force(ray_state):
    bundle(ray_state)
    useful_data_for()
    Force = copy.deepcopy(F_g(ray_state)+F_s(ray_state))
    all_forwards.clear()
    return Force

if __name__ == '__main__':
    def inter(x_range, y_range):
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        plt.figure()
        for i in range(len(x)):
            ray_state = [x[i], y[i], 0, 0]
            bundle(ray_state)
            useful_data_for()
            print(x[i], np.shape(useful_data_for()),F_g(ray_state))
            # plt.plot(y[i], F_g(ray_state)[1], 'k.')
            plt.plot(x[i], F_s(ray_state)[0], 'g.')
            # print(F_s(), F_g())
            # print(np.shape(all_forwards),np.shape(useful_data_for()))
            all_forwards.clear()
        plt.show()

    start = timeit.default_timer()

    bundle(input.droplet_pos)
    useful_data_for()
    # print(input.droplet_pos)
    # print(radiation_force([504, 0, 0, 0]))
    # inter([270,1000],[0,0])
    print(F_g(input.droplet_pos))
    print(F_s(input.droplet_pos))
    # bundle(input.droplet_pos)
    # F_s(input.droplet_pos)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

