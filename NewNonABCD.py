# -*- coding: utf-8 -*-
"""
Time: 2020/Jul/06

@author: Benchi Zhao
Python version: 3.7
Functionality: This module trace the ray and plot the diagram. Also plot the intensity of cross-section.
Dependence: Need import 'input', all parameters of the experiment is saved in that file 'input'.
"""
import pandas as pd
import numpy as np
import math
import copy
import input
import matplotlib.pyplot as plt
import timeit
import sys
from numba import jit
import timeit
from multiprocessing import Pool


forward = []
backward = []
tilt = []
def degree2rad(degree):
    return degree/180 * math.pi
def rad2degree(rad):
    return rad/math.pi * 180

def slope2rad(slope):
    return np.arctan(slope)
def rad2slope(rad):
    return np.tan(rad)

def degree2slope(degree):
    return np.tan(np.deg2rad(degree))
def slope2degree(slope):
    return np.rad2deg(np.arctan(slope))

def gaussian(position, sigma):
    return 1 / (np.sqrt(2 * math.pi)*sigma) * math.exp(-position ** 2 / (2*sigma**2))

def snell(theta1):
    '''
        snells(theta1)
            Calculates theta2 from snells law in radians.

        Parameters
        ----------
        theta1: float
            Angle in to refracting surface (in rad).

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

def R(theta1): #in rad
    '''
    R(theta1)
        Calculates the Fresnel reflectivity as a function of theta1.

    Parameters
    -------------
    theta1: float
        Angle in to refracting surface.

    Returns
    ---------
    R: float
        Fresnel power reflectance as a function of theta1 and theta2.

    Input Script Parameters
    ---------------------------
    polarisation: string
        Polarisation of laser beam used.
    medium_n: float
        Refractive index of the surrounding medium.
    target_n: float
        Refractive index of target.

    '''
    theta2 = snell(theta1)
    if input.polarisation == 'p':
        result = (abs((input.medium_n * np.cos(theta2) - input.target_n * np.cos(theta1)) / (
                    input.medium_n * np.cos(theta2) + input.target_n * np.cos(theta1)))) ** 2
        return result
    elif input.polarisation == 's':
        result = ((input.medium_n*np.cos(theta1)-(input.target_n*np.cos(theta2)))/(input.medium_n*np.cos(theta1)+(input.target_n*np.cos(theta2))))**2
        return result
    elif input.polarisation == 'circle':
        result = 0.5*(((np.sin(theta1-theta2)**2)/(np.sin(theta1+theta2)**2))+((np.tan(theta1-theta2)**2)/(np.tan(theta1+theta2)**2)))
        return result
    else:
        sys.exit("Polarisation is not defined it must be set to 'p','circle' or 's', check input file")
def T(theta1):
    '''
    T(theta1)
        Calculates the Fresnel reflectivity as a function of theta1.

    Parameters
    -------------
    theta1: float
        Angle in to refracting surface.

    Returns
    ---------
    R: float
        Fresnel refractivity as a function of theta1.

    Input Script Parameters
    ---------------------------
    polarisation: string
        Polarisation of laser beam used.
    medium_n: float
        Refractive index of the surrounding medium.
    target_n: float
        Refractive index of target.

    '''
    if input.target_n >= input.medium_n:
        reflection = R(theta1)
    else:
        thetac = np.arcsin(input.target_n / input.medium_n)
        if theta1 >= thetac:
            reflection = 1
        else:
            reflection = R(theta1)
    T = 1 - reflection
    return T

def hit_point_front(r,central):
    '''
    central_point: list
        get two parameter [x, y]
    Calculate the position where light incident the lens.
    :param r: float
        the radius of lens
    '''
    state = copy.deepcopy(forward[-1])

    theta = state[2]
    y = state[1] + central[1]
    m = y * np.cos(math.pi/2 - theta) + np.sqrt((y ** 2) * (np.cos(math.pi/2 - theta)) ** 2 - (y ** 2 - r ** 2))
    x = m * np.sin(math.pi / 2 - theta)
    y = m * np.sin(theta)

    forward[-1][0] -= x
    forward[-1][1] -= y
    backward[-1][0] -= x
    backward[-1][1] -= y

def ray(x, y, theta, intensity):
    '''
    ray(self)
        Fire the beam at the beginning
        4 elements describe the refraction are appended in forwards;
        4 elements describe the deflection are appended backwards.
    Parameters
    -------------
    x: float
        x position of the initial beam.
    y: float
        y position of the initial beam.
    theta: float
        the tilt angle of the beam. (in rad)
    intensity: float
        The intensity of the beam.

    Returns
    ---------

    '''
    if input.ray_type in ['Gaussian', 'gaussian', 'G', 'g']:
        ray_state = [x, y, theta, intensity]
        forward.append(ray_state)
        backward.append([x, y, theta, 0])
    elif input.ray_type in ['Flat', 'flat', 'f', 'F']:
        ray_state = [x, y, theta, 1]
        forward.append(ray_state)
        backward.append([x, y, theta, 0])
    else:
        raise NameError

def free_propagate(distance):
    '''
    Describe the ray state when it propagate for a distance
    :param distance: float
        The distance that the ray propagate
    '''
    state = copy.deepcopy(forward[-1])

    if state[2] < 0:
        state[2] += 2 * np.pi

    if state[2] == np.pi/2:
        ray_state_1 = [state[0], state[1] + distance, state[2], state[3]]
    elif state[2] == 3 * np.pi/2:
        ray_state_1 = [state[0], state[1] - distance, state[2], state[3]]
    elif state[2] > np.pi/2 and state[2] < 3 * np.pi/2:
        slope = rad2slope(state[2])
        x = state[0] - distance
        y = -slope * distance + state[1]
        ray_state_1 = [x, y, state[2], state[3]]
    else:
        slope = rad2slope(state[2])
        x = state[0] + distance
        y = slope * distance + state[1]
        ray_state_1 = [x, y, state[2], state[3]]

    deflection = backward[-1]
    if deflection[2] == np.pi/2:
        ray_state_2 = [state[0], state[1] + distance, deflection[2], deflection[3]]
    elif deflection[2] == 3 * np.pi/2:
        ray_state_2 = [state[0], state[1] - distance, deflection[2], deflection[3]]
    elif deflection[2] > np.pi/2 and deflection[2] < 3 * np.pi/2:
        slope = rad2slope(deflection[2])
        x = state[0] - distance
        y = -slope * distance + state[1]
        ray_state_2 = [x, y, deflection[2], deflection[3]]
    else:
        slope = rad2slope(deflection[2])
        x = state[0] + distance
        y = slope * distance + state[1]
        ray_state_2 = [x, y, deflection[2], deflection[3]]

    forward.append(ray_state_1)
    backward.append(ray_state_2)

def propagate(distance):
    '''
    Describe the ray state when it propagate for a distance
    :param distance: float
        The distance that the ray propagate
    '''
    state = copy.deepcopy(forward[-1])

    if state[2] < 0:
        state[2] += 2 * np.pi

    if state[2] == np.pi/2:
        ray_state_1 = [state[0], state[1] + distance, state[2], state[3]]
    elif state[2] == 3 * np.pi/2:
        ray_state_1 = [state[0], state[1] - distance, state[2], state[3]]
    elif state[2] > np.pi/2 and state[2] < 3 * np.pi/2:
        slope = rad2slope(state[2])
        x = state[0] - distance
        y = -slope * distance + state[1]
        ray_state_1 = [x, y, state[2], state[3]]
    else:
        slope = rad2slope(state[2])
        x = state[0] + distance
        y = slope * distance + state[1]
        ray_state_1 = [x, y, state[2], state[3]]

    forward.append(ray_state_1)
    ray_state_2 = copy.deepcopy(backward[-1])
    ray_state_2[3] = 0
    backward.append(ray_state_2)

def full_reflection(tilt): #in rad
    '''
    Describe the ray state after full reflection
    :param tilt: float
        the angle tilted of mirror
    :return: list
        ray_state, containing 4 elements: x, y, theta, intensity
    '''
    state = forward[-1]
    back_angle = np.pi - state[2] + 2 * tilt
    return  back_angle

def flat_interface(tilt, n1, n2):
    '''
    Describe how the ray state change after the ray pass through the flat interface.
    Append the refractive beam into forward, and append the reflected beam into backward.

    parameter
    -------------
    tilt: float
        The angle tilted by the interface, if the tile is 0, then the interface is verticle;
                                            if the tile is positive, the interface rotate anti-clockwise;
                                            if the tilt is negative, the interface rotate clockwise. (in rad)
    n1: float
        The refractive index of the incident beam.
    n2： float
        The refractive index of the refractive beam.
    '''
    state = forward[-1]
    # snell's law method
    incident_rad = state[2] - tilt
    refracted_deg = np.arcsin(n1 * np.sin(incident_rad)/n2) + tilt
    reflected_angle = np.pi - state[2] + 2 * tilt
    if refracted_deg < 0:
        refracted_deg += 2 * np.pi
    if reflected_angle < 0:
        reflected_angle += 2 * np.pi
    ray_state_1 = [state[0], state[1], refracted_deg, state[3] * T(state[2]-tilt)]
    ray_state_2 = [state[0], state[1], reflected_angle, state[3] * R(state[2]-tilt)]

    forward.append(ray_state_1)
    backward.append(ray_state_2)

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def circle(r,ray_state):
    '''
    Trace the ray in the circle.
    Append the refractive beam into forward, and append the reflected beam into backward.
    '''

    if forward[-1][0] < ray_state[0]:
        raise ValueError
    else:
        forward[-1][1] = np.tan(forward[-1][2]) * (ray_state[0] - forward[-1][0]) + forward[-1][1]
        forward[-1][0] = ray_state[0]
        backward[-1][1] = np.tan(forward[-1][2]) * (ray_state[0] - forward[-1][0]) + forward[-1][1]
        backward[-1][0] = ray_state[0]

    if forward[-1][1] >= r + ray_state[1] or forward[-1][1] <= ray_state[1] - r:
        propagate(r)

    else:
        circle_center = (ray_state[0], ray_state[1])
        pt1 = (forward[-2][0], forward[-2][1])
        pt2 = (forward[-1][0], forward[-1][1])
        point = circle_line_segment_intersection(circle_center, input.radius, pt1, pt2, full_line=True, tangent_tol=1e-9)

        forward[-1][0] = point[0][0]
        forward[-1][1] = point[0][1]
        backward[-1][0] = point[0][0]
        backward[-1][1] = point[0][1]

        # # snell's law
        tilt_1 = np.arcsin((forward[-1][1]-ray_state[1])/ r)
        flat_interface(-tilt_1, input.medium_n, input.target_n)
        state = forward[-1]
        a = ray_state[0]
        b = ray_state[1]
        k = rad2slope(state[2])
        c = state[1] - k*state[0]

        delt = (-a + (c-b)*k)**2 - (1+k**2) * (a**2+(c-b)**2 - r**2)
        x1 = (a-(c-b)*k - np.sqrt(delt)) / (1+k**2)
        x2 = (a-(c-b)*k + np.sqrt(delt)) / (1+k**2)
        free_propagate(x2-x1)
        tilt_2 = np.arcsin((forward[-1][1]-ray_state[1]) / r)
        flat_interface(tilt_2, input.medium_n, input.target_n)

def lens_trace(f, pos, thick):
    '''
    Trace the ray in the lens.
    Append the refractive beam into forward, and append the reflected beam into backward.

    :param f: float
        The focal length of the lens we use
    :param pos: float
        The x position of central lens
    :param thick: float
        The thickness of the lens
    '''
    state = copy.deepcopy(forward[-1])
    r = f
    free_propagate(pos + r - thick/2)
    hit_point_front(r, [pos + r - thick/2, 0])
    tilt_1 = np.arcsin(state[1] / r)
    flat_interface(-tilt_1, input.medium_n, input.target_n)
    a = pos - r + thick/2
    b = 0
    k = rad2slope(forward[-1][2])
    c = forward[-1][1] - k * forward[-1][0]

    delt = (-a + (c - b) * k) ** 2 - (1 + k ** 2) * (a ** 2 + (c - b) ** 2 - r ** 2)
    # x1 = (a - (c - b) * k - np.sqrt(delt)) / (1 + k ** 2)
    x2 = (a - (c - b) * k + np.sqrt(delt)) / (1 + k ** 2)

    free_propagate(x2 - forward[-1][0])
    tilt_2 = np.arcsin(state[1] / r)
    flat_interface(tilt_2, input.target_n, input.medium_n)


if __name__ == '__main__':
    ray_state = input.droplet_pos
    all_forwards = []
    all_backwards = []
    max_x = ray_state[0]*1.5


    def intensity(position):
        I = []
        for i in range(len(position)):
            intens = gaussian(position[i], input.sigma)
            I.append(intens)
        normalise_I = I / sum(I)
        return normalise_I

    def bundle():
        '''
        Trace a bundle of rays, the number of rays i determined by 'input' file, no_of_rays.
        All data are saved in the all_forwards and all_backwards. The shape the two list are identical （3-D list）.
        '''
        list_of_rays = np.linspace(-input.width, input.width, input.no_of_rays)
        for i in range(input.no_of_rays):
            print(i)
            ray(0, list_of_rays[i]+input.y_displacement, 0, intensity(list_of_rays)[i])
            lens_trace(input.lens_f, input.lens_pos, input.len_thickness)
            free_propagate(10)
            propagate(ray_state[0]-input.lens_pos-10)
            circle(input.radius,ray_state)
            free_propagate(15)
            propagate(200)
            all_forwards.append(copy.deepcopy(forward))
            all_backwards.append(copy.deepcopy(backward))
            forward.clear()
            backward.clear()

    def make_table():
        '''
        Show the all_forwards and all_backwards in table form,  making it clean.
        '''
        for i in range(input.no_of_rays):
            data_1 = all_forwards[i]
            data_2 = all_backwards[i]
            print(np.shape(data_1))
            df_1 = pd.DataFrame(data_1, columns=['x', 'y', 'theta', 'intensity'])
            df_2 = pd.DataFrame(data_2, columns=['x', 'y', 'theta', 'intensity'])
            print('Ray', i + 1)
            print(df_1)
            print(df_2)

    def plot():
        '''
        PLot the trace figure.
        '''
        for i in range(input.no_of_rays):
            for j in range(len(all_forwards[i])-1):
                x_1 = np.linspace(all_forwards[i][j][0], all_forwards[i][j+1][0])
                y_1 = np.linspace(all_forwards[i][j][1], all_forwards[i][j+1][1])

                x_2 = np.linspace(all_backwards[i][j][0], all_forwards[i][j - 1][0])
                y_2 = np.linspace(all_backwards[i][j][1], all_forwards[i][j - 1][1])


                plt.plot(x_1, y_1, 'k-', linewidth=all_forwards[i][j][3]*20)
                # plt.plot(x_2, y_2, 'k-', linewidth=all_backwards[i][j][3]*100)


            plt.xlim(0, max_x)
            plt.ylim(-50, 50)
            theta = np.linspace(0, 2 * math.pi)
            x_c = input.radius * np.cos(theta) + ray_state[0]
            y_c = input.radius * np.sin(theta) + ray_state[1]
            plt.plot(x_c, y_c)

        # # plot lens
        theta1 = np.linspace(3 * math.pi / 4, 5 * math.pi / 4)
        theta2 = np.linspace(- math.pi / 4, math.pi / 4)
        x_1 = input.lens_f * np.cos(theta1) + input.lens_pos + input.lens_f - input.len_thickness / 2
        x_2 = input.lens_f * np.cos(theta2) + input.lens_pos - input.lens_f + input.len_thickness / 2
        y_1 = input.lens_f * np.sin(theta1)
        y_2 = input.lens_f * np.sin(theta2)
        plt.plot(x_1, y_1)
        plt.plot(x_2, y_2)
        # plt.grid()
        plt.show()

    def detector(x_position):
        '''
        Detect the intensity of rays at position x.
        :param x_position: float
            The position we would like to detect the bundle of rays
        :return y_values: list
            positions of rays in y-axis.
        :return intens_value: list
            the corresponding intensity values at position y.

        '''
        y_values = []
        intns_values = []
        data_1 = copy.deepcopy(all_forwards)
        for i in range(input.no_of_rays):
            refract_x = np.array(data_1[i])[:, 0]
            refract_y = np.array(data_1[i])[:, 1]
            refract_theta = np.array(data_1[i])[:, 2]
            refract_intensity = np.array(data_1[i])[:, 3]
            for j in range(len(refract_x)-1):
                if x_position > refract_x[j] and x_position < refract_x[j + 1]:
                    y = np.tan(refract_theta[j]) * (x_position - refract_x[j]) + refract_y[j]
                    y_values.append(y)
                    intns_values.append(refract_intensity[j])
        return y_values, intns_values

    def bin(position):
        new_y = []
        new_int = []
        y_values = detector(position)[0]
        intns_values = detector(position)[1]
        point = [[x,y] for x,y in zip(y_values,intns_values)]
        group = np.linspace(min(y_values), max(y_values), int((max(y_values)-min(y_values))/1))
        for i in range(len(group)-1):
            a = 0
            b = 0
            y = [j for j in point if group[i+1] > j[0] >= group[i]]
            for j in range(len(y)):
                a += y[j][0]
                b += y[j][1]

            if len(y) !=0:
                new_y.append(a/len(y))
                new_int.append(b)
        return new_y, new_int

    def plot_bin(positions):
        plt.figure('bin')
        for i in range(len(positions)):
            y_values,intns_values = bin(positions[i])[0],bin(positions[i])[1]
            plt.plot(y_values, intns_values, ms=1, label="x = %d" % (positions[i]))
        plt.legend(fontsize = 12)
        plt.show()


    # p = Pool(10)  # 一般为CPU核数+1
    # res_list = []
    # for i in range(100):
    #     res = p.apply_async(bundle(),)
    #     res_list.append(res)
    # p.close()
    # p.join()
    # plt.figure()
    # for i in range(100):
    #     start = timeit.default_timer()
    #     bundle()
    #     stop = timeit.default_timer()
    #     time = stop - start
    #     plt.plot(i, time)
    #     print('Time: ', time)
    # plt.show()
    start = timeit.default_timer()
    bundle()
    # make_table()
    plot()
    # plot_bin([400,500,600,700])


    stop = timeit.default_timer()
    print('Time: ', stop - start)