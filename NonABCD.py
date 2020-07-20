# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:42:19 2020

@author: Benchi Zhao
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import copy
import timeit

class RayTracing:

    def __init__(self, x, y, theta):
        '''
        __init__ (self,x,y,theta)
            Gives the initial state of the ray.

        Parameters
        ------------
        self.x: float
            Initial x-position of the ray.
        self.y: float
            Initial y-position of the ray.
        self.theta: float
            The angle between the horizontal and the ray path (in degree).
        self.state: list
            When ray interacting with the optical equipments, the ray state will change,
            all states are recorded in self.state, which contains four variables:
            x, y, theta, intensity.
        self.deflection: list
            All reflected ray state is saved in this list, which contains four variables:
            x, y, theta, intensity.
        self.n_1: float
            The refractive index of air.
        self.n_2: float
            The refractive index of the droplet.
        '''
        self.x = x
        self.y = y
        self.theta = theta
        self.state = []
        self.deflection = []
        self.n_1 = 1
        self.n_2 = 1.5
        self.if_lens = False
        self.if_circle = False

    def degree2rad(self, degree):
        return degree/180 * math.pi
    def rad2degree(self, rad):
        return rad/math.pi * 180

    def slope2rad(self, slope):
        return np.arctan(slope)
    def rad2slope(self, rad):
        return np.tan(rad)

    def degree2slope(self, degree):
        return np.tan(np.deg2rad(degree))
    def slope2degree(self, slope):
        return np.rad2deg(np.arctan(slope))

    def gaussian(self, position, sigma):
        return 1 / (np.sqrt(2 * math.pi)*sigma) * math.exp(-position ** 2 / (2*sigma**2)) *10

    def R(self, angle):
        '''
        Reflection, to calculate how many rays reflected from the optical lens.
        :param angle: float
            the angle incident the lens (in rad).
        :return: float
            the percentage of haw many ray reflected.
        '''
        theta_i = self.degree2rad(angle)
        theta_t = np.sqrt(1 - (self.n_1 / self.n_2 * np.sin(theta_i)) ** 2)
        result = (abs((self.n_1 * np.cos(theta_t) - self.n_2 * np.cos(theta_i)) / (
                    self.n_1 * np.cos(theta_t) + self.n_2 * np.cos(theta_i)))) ** 2
        return result

    def T(self, angle):
        '''
        Transmission, to calculate how many rays pass the optical lens.
        :param angle: float
            the angle incident the lens (in rad).
        :return: float
            the percentage of haw many ray pass.
        '''
        return 1 - self.R(angle)

    def hit_point_front(self, r):
        '''
        Calculate the position where light incident the lens.
        :param r: float
            the radius of lens
        '''
        state = copy.deepcopy(self.state[-1])
        theta = self.degree2rad(self.state[-1][2])
        y = self.state[-1][1]
        m = y * np.cos(math.pi/2 -theta) + np.sqrt((y ** 2) * (np.cos(math.pi/2 - theta)) ** 2 - (y ** 2 - r ** 2))
        x = m * np.sin(math.pi / 2 - theta)
        y = m * np.sin(theta)

        self.state[-1][0] -= x
        self.state[-1][1] -= y
        self.deflection[-1][0] -= x
        self.deflection[-1][1] -= y

    def ray(self, distribution, sigma):
        '''
        ray(self)
            Append the initial ray state into the total ray state.
            4 elements describe the refraction saved in self.state;
            4 elements describe the deflection saved in self.deflection.
        '''
        if distribution in ['Gaussian', 'gaussian', 'G', 'g']:
            ray_state = [self.x, self.y, self.theta, self.gaussian(self.y, sigma)]
            self.state.append(ray_state)
            self.deflection.append([self.x, self.y, self.theta, 0])
        elif distribution in ['Flat', 'flat', 'f', 'F']:
            ray_state = [self.x, self.y, self.theta, 1]
            self.state.append(ray_state)
            self.deflection.append([self.x, self.y, self.theta, 0])
        else:
            raise NameError

    def free_propagate(self, distance):
        '''
        Describe the ray state when it propagate for a distance
        :param distance: float
            The distance that the ray propagate
        '''
        state = self.state[-1]

        if state[2] < 0:
            state[2] += 360

        if state[2] == 90:
            ray_state_1 = [state[0], state[1] + distance, state[2], state[3]]
        elif state[2] == 270:
            ray_state_1 = [state[0], state[1] - distance, state[2], state[3]]
        elif state[2] > 90 and state[2] < 270:
            slope = self.degree2slope(state[2])
            x = state[0] - distance
            y = -slope * distance + state[1]
            ray_state_1 = [x, y, state[2], state[3]]
        else:
            slope = self.degree2slope(state[2])
            x = state[0] + distance
            y = slope * distance + state[1]
            ray_state_1 = [x, y, state[2], state[3]]

        deflection = self.deflection[-1]
        if deflection[2] == 90:
            ray_state_2 = [state[0], state[1] + distance, deflection[2], deflection[3]]
        elif deflection[2] == 270:
            ray_state_2 = [state[0], state[1] - distance, deflection[2], deflection[3]]
        elif deflection[2] > 90 and deflection[2] < 270:
            slope = self.degree2slope(deflection[2])
            x = state[0] - distance
            y = -slope * distance + state[1]
            ray_state_2 = [x, y, deflection[2], deflection[3]]
        else:
            slope = self.degree2slope(deflection[2])
            x = state[0] + distance
            y = slope * distance + state[1]
            ray_state_2 = [x, y, deflection[2], deflection[3]]

        self.state.append(ray_state_1)
        self.deflection.append(ray_state_2)

    def propagate(self, distance):
        '''
        Describe the ray state when it propagate for a distance
        :param distance: float
            The distance that the ray propagate
        '''
        state = self.state[-1]

        if state[2] < 0:
            state[2] += 360

        if state[2] == 90:
            ray_state_1 = [state[0], state[1] + distance, state[2], state[3]]
        elif state[2] == 270:
            ray_state_1 = [state[0], state[1] - distance, state[2], state[3]]
        elif state[2] > 90 and state[2] < 270:
            slope = self.degree2slope(state[2])
            x = state[0] - distance
            y = -slope * distance + state[1]
            ray_state_1 = [x, y, state[2], state[3]]
        else:
            slope = self.degree2slope(state[2])
            x = state[0] + distance
            y = slope * distance + state[1]
            ray_state_1 = [x, y, state[2], state[3]]

        self.state.append(ray_state_1)
        ray_state_2 = copy.deepcopy(ray_state_1)
        ray_state_2[3] = 0
        self.deflection.append(ray_state_2)

    def full_reflection(self, tilt):
        '''
        Describe the ray state after full reflection
        :param tilt: float
            the angle tilted of mirror
        :return: list
            ray_state, containing 4 elements: x, y, theta, intensity
        '''
        state = self.state[-1]
        ray_state = [state[0], state[1], 180 - state[2] + 2 * tilt, state[3]]
        self.state.append(ray_state)
        self.deflection.append(ray_state)
        return ray_state

    def flat_interface(self, tilt, n1, n2):
        '''
        Describe how the ray state change after the ray pass through the flat interface.
        tilt is in degree
        '''
        state = self.state[-1]
        # snell's law method
        incident_rad = self.degree2rad(state[2] - tilt)
        refracted_deg = self.rad2degree(np.arcsin(n1 * np.sin(incident_rad)/n2)) + tilt
        if refracted_deg < 0:
            refracted_deg += 360
        ray_state_1 = [state[0], state[1], refracted_deg, state[3] * self.T(state[2]-tilt)]

        reflected_angle = 180 - state[2] + 2 * tilt
        if reflected_angle < 0:
            reflected_angle += 360
        ray_state_2 = [state[0], state[1], reflected_angle, state[3] * self.R(state[2]-tilt)]

        self.state.append(ray_state_1)
        self.deflection.append(ray_state_2)

    def circle(self, r, central_point):
        '''
        Describe how the ray state change after the ray pass through the flat interface.
        '''
        if self.state[-1][0] < central_point:
            raise ValueError
        else:
            self.state[-1][1] = np.tan(np.deg2rad(self.state[-1][2])) * (central_point - self.state[-1][0]) + self.state[-1][1]
            self.state[-1][0] = central_point
            self.deflection[-1][1] = np.tan(np.deg2rad(self.state[-1][2])) * (central_point - self.state[-1][0]) + \
                                self.state[-1][1]
            self.deflection[-1][0] = central_point


        if abs(self.state[-1][1]) >= r:
            self.free_propagate(r)
        else:
            self.hit_point_front(r)
            # # snell's law
            tilt_1 = self.rad2degree(np.arcsin(self.state[-1][1] / r))
            self.flat_interface(-tilt_1, self.n_1, self.n_2)

            a = central_point
            b = 0
            k = self.degree2slope(self.state[-1][2])
            c = self.state[-1][1] - k*self.state[-1][0]

            delt = (-a + (c-b)*k)**2 - (1+k**2) * (a**2+(c-b)**2 - r**2)
            x1 = (a-(c-b)*k - np.sqrt(delt)) / (1+k**2)
            x2 = (a-(c-b)*k + np.sqrt(delt)) / (1+k**2)
            y1 = k*x1 + c
            y2 = k*x2 + c
            self.free_propagate(x2-x1)
            tilt_2 = self.rad2degree(np.arcsin(self.state[-1][1] / r))
            print('tilt2:', self.degree2rad(tilt_2))
            self.flat_interface(tilt_2, self.n_2, self.n_1)


            # #ABCD method
            # tilt = np.arcsin(self.state[-1][1] / r)
            # angle_in = self.degree2rad(self.state[-1][2])
            # ABCD = np.dot(np.array([[1, 0], [((self.n_1 - self.n_2) / (r * self.n_2)), self.n_1 / self.n_2]]),
            #                       np.array([self.state[-1][1],self.degree2slope(self.state[-1][2])]))
            # intensity = self.state[-1][3] * self.T(self.rad2degree(angle_in + tilt))
            # ray_state_1 = [self.state[-1][0], ABCD[0], self.slope2degree(ABCD[1]), intensity]
            #
            #
            # ray_state_2 = [self.state[-1][0], self.state[-1][1], 180 - self.state[-1][2] - 2 * self.rad2degree(tilt), self.state[-1][3] * self.R(self.state[-1][2]-self.rad2degree(tilt))]
            # self.state.append(ray_state_1)
            # self.deflection.append(ray_state_2)

    def lens_trace(self,f,pos,thick):
        self.if_lens = True
        r = f
        self.free_propagate(pos + r - thick/2)
        self.hit_point_front(r)
        tilt_1 = self.rad2degree(np.arcsin(self.state[-1][1] / r))
        self.flat_interface(-tilt_1, self.n_1, self.n_2)

        a = pos - r + thick/2
        b = 0
        k = self.degree2slope(self.state[-1][2])
        c = self.state[-1][1] - k * self.state[-1][0]

        delt = (-a + (c - b) * k) ** 2 - (1 + k ** 2) * (a ** 2 + (c - b) ** 2 - r ** 2)
        x1 = (a - (c - b) * k - np.sqrt(delt)) / (1 + k ** 2)
        x2 = (a - (c - b) * k + np.sqrt(delt)) / (1 + k ** 2)
        self.free_propagate(x2 - self.state[-1][0])#- 2*r+thickness
        tilt_2 = self.rad2degree(np.arcsin(self.state[-1][1] / r))
        self.flat_interface(tilt_2, self.n_2, self.n_1)




        #
        # deflection = self.full_reflection(self.state[-1][2])
        # self.state.append(np.array([self.state[-1][0], self.state[-1][1], self.state[-1][2], 0]))
        # self.deflection.append(deflection)



if __name__ == '__main__':

    pass