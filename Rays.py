# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:42:19 2020

@author: Benchi Zhao
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


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

    def slope2rad(self, slope):
        return np.arctan(slope)

    def angle2slope(self, angle):
        return np.tan(np.deg2rad(angle))

    def rad2slope(self,rad):
        return np.tan(rad)

    def gaussian(self, position):
        return 1 / (np.sqrt(2 * math.pi)) * math.exp(-position ** 2 / 2)

    def R(self, angle):
        '''
        Reflection, to calculate how many rays reflected from the optical lens.
        :param angle: float
            the angle incident the lens (in rad).
        :return: float
            the percentage of haw many ray reflected.
        '''
        theta_i = self.slope2rad(angle)
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

    def hit_point(self, r):
        '''
        Calculate the position where light incident the lens.
        :param r: float
            the radius of lens
        '''
        theta = self.slope2rad(self.state[-1][2])
        y = self.state[-1][1]
        m = y * np.cos(math.pi/2 -theta) + np.sqrt((y ** 2) * (np.cos(math.pi/2 - theta)) ** 2 - (y ** 2 - r ** 2))
        x = m * np.sin(math.pi / 2 - theta)
        y = m * np.sin(theta)

        self.state[-1][0] -= x
        self.state[-1][1] -= y

    def ray(self, distribution):
        '''
        ray(self)
            Append the initial ray state into the total ray state.
            4 elements describe the refraction saved in self.state;
            4 elements describe the deflection saved in self.deflection.
        '''
        if distribution in ['Gaussian', 'gaussian', 'G', 'g']:
            ray_state = np.array(
                [self.x, self.y, self.theta, self.gaussian(self.y)])
            self.state.append(ray_state)
            self.deflection.append(np.array([self.x, self.y, self.theta, 0]))
        elif distribution in ['Flat', 'flat', 'f', 'F']:
            ray_state = np.array([self.x, self.y, self.theta, 1])
            self.state.append(ray_state)
            self.deflection.append(np.array([self.x, self.y, self.theta, 0]))
        else:
            raise NameError

    def free_propagate(self, distance):
        '''
        Describe the ray state when it propagate for a distance
        :param distance: float
            The distance that the ray propagate
        '''
        ray_state_1 = np.dot(np.array([[1, distance], [0, 1]]), self.state[-1][1:3])
        ray_state_1 = np.append(self.state[-1][0] + distance, ray_state_1)
        ray_state_1 = np.append(ray_state_1, self.state[-1][3])

        ray_state_2 = np.dot(np.array([[1, -distance], [0, 1]]), self.deflection[-1][1:3])
        ray_state_2 = np.append(self.deflection[-1][0] - distance, ray_state_2)
        ray_state_2 = np.append(ray_state_2, self.deflection[-1][3])

        self.state.append(ray_state_1)
        self.deflection.append(ray_state_2)

    def lens(self, f):
        '''
        Describe how the ray change when pass the thin lens
        :param f: f is the focal length of the lens
        '''
        ray_state_1 = np.dot(np.array([[1, 0], [-1 / f, 1]]), self.state[-1][1:3])
        ray_state_1 = np.append(self.state[-1][0], ray_state_1)
        ray_state_1 = np.append(ray_state_1, self.state[-1][3]*self.T())

        ray_state_2 = self.full_reflection()
        ray_state_2[3] = ray_state_2[3]*self.R()

        self.state.append(ray_state_1)
        self.deflection.append(ray_state_2)

    def full_reflection(self, tilt):
        '''
        Describe the ray state after full reflection
        :param tilt: float
            the angle tilted of mirror
        :return: list
            ray_state, containing 4 elements: x, y, theta, intensity
        '''
        ray_state = np.dot(np.array([[1, 0], [0, -1]]), np.array([self.state[-1][1], self.state[-1][2]]))
        reflected_angle = ray_state[1]
        ray_state = np.append(self.state[-1][0], ray_state)
        ray_state = np.append(ray_state, self.state[-1][3])
        return ray_state

    def mirror(self):
        deflection = self.full_reflection(self.state[-1][2])
        self.state.append(np.array([self.state[-1][0], self.state[-1][1], self.state[-1][2], 0]))
        self.deflection.append(deflection)

    def flat_interface(self):
        '''
        Describe how the ray state change after the ray pass through the flat interface.
        '''
        self.M = False
        self.hit_point = False
        ray_state_1 = np.dot(np.array([[1, 0], [0, self.n_1 / self.n_2]]), self.state[-1][1:3])
        ray_state_1 = np.append(self.state[-1][0], ray_state_1)
        ray_state_1 = np.append(ray_state_1, self.state[-1][-1] * self.T(self.state[-1][2]))
        ray_state_2 = self.full_reflection(self.state[-1][2])
        ray_state_2[-1] = ray_state_2[-1] * self.R(self.state[-1][2])

        self.state.append(ray_state_1)
        self.deflection.append(ray_state_2)

    def curved_interface(self, r):
        '''
        Describe how the ray state change after the ray pass through the flat interface.
        '''
        if abs(self.state[-1][1]) > r:
            pass
        else:
            self.hit_point(r)
            tilt = np.arcsin(self.state[-1][1] / r)
            # print(tilt/math.pi *180)
            angle_in = self.slope2rad(self.state[-1][2])
            ray_state_1 = np.dot(np.array([[1, 0], [((self.n_1 - self.n_2) / (r * self.n_2)), self.n_1 / self.n_2]]),
                                 self.state[-1][1:3])
            ray_state_1 = np.append(self.state[-1][0], ray_state_1)
            ray_state_1 = np.append(ray_state_1, self.state[-1][3] * self.T(angle_in + tilt))

            reflected_angle = np.dot(np.array([[1, 0], [0, -1]]), np.array([self.state[-1][1], self.state[-1][2]]))[1]
            # print(reflected_angle/math.pi *180)
            reflected_angle = self.slope2rad(reflected_angle) + math.pi - 2 * tilt
            # print(reflected_angle / math.pi * 180)
            reflected_angle = self.rad2slope(reflected_angle)

            intensity = self.state[-1][3]*self.R(angle_in + tilt)
            ray_state_2 = np.array([self.state[-1][0], self.state[-1][1], reflected_angle, intensity])

            self.state.append(ray_state_1)
            self.deflection.append(ray_state_2)


if __name__ == '__main__':

    pass

