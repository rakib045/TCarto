import cvxopt
import numpy as np
import random
# import quadprog
from sympy.geometry import *
from time import time
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import sys
import multiprocessing
import math as m
import shapefile
import colorsys
import multiprocessing as mp
import seaborn as sns

from cvxopt import matrix, solvers

from energyMinimization import *
from sympy import symbols
from sympy.plotting import plot
from skimage.transform import PiecewiseAffineTransform, warp

######################################
# These are the helper functions
######################################
# a function that solves the quadratic programming
# https://scaron.info/blog/quadratic-programming-in-python.html
# Jyoti: This link had a problem and I emailed the author to fix, and I think they fixed that now; anyways - what I did below is correct - so no worries

# imp_cell_threshold = 1.25
imp_cell_threshold = 1.25


###################  Grid Generation ###########################

def grid_node_generation(node, grid_horiz, grid_vert):
    nodes = []
    for i in range(grid_horiz + 1):
        x = []
        for j in range(grid_vert + 1):
            x.append(node(str(i) + "" + str(j), Point(i, j)))
        nodes.append(x)
    return nodes


###################  Grid Generation ###########################

################### Read From Input File #######################
def read_text_file_actual_order(input_data_file, grid_horiz, grid_vert):
    sample_val = []
    input_file = open(input_data_file, "r")
    in_total_str = ''
    in_str = input_file.readlines()
    for i in range(len(in_str)):
        in_total_str += in_str[i].replace('\n', '').replace(' ', '')

    val_str = in_total_str.split(",")
    input_file.close()

    for v in val_str:
        sample_val.append(float(v))

    values = np.zeros((grid_horiz, grid_vert))
    sample_val_count = 0
    for i in range(grid_horiz):
        for j in range(grid_vert):
            values[i][j] = sample_val[sample_val_count]
            sample_val_count += 1

    return values


def read_text_file(input_data_file, grid_horiz, grid_vert):
    sample_val = []
    input_file = open(input_data_file, "r")
    in_total_str = ''
    in_str = input_file.readlines()
    for i in range(len(in_str)):
        in_total_str += in_str[i].replace('\n', '').replace(' ', '')

    val_str = in_total_str.split(",")
    input_file.close()

    for v in val_str:
        sample_val.append(float(v))

    values = np.zeros((grid_horiz, grid_vert))
    sample_val_count = 0
    for j in range(grid_vert - 1, -1, -1):
        for i in range(grid_horiz):
            values[i][j] = sample_val[sample_val_count]
            sample_val_count += 1

    # we now normalize the cell values so that
    # all values sum to 1
    # values = values / values.sum()

    # all values sum to totalarea
    # values = values * grid_vert * grid_horiz

    return values


################### Read From Input File #######################

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


# this is to check which side the point p is on; L denotes a line ax+by+c = 0

def getSign(p, L):
    testvalue = (L.coefficients[0] * p.args[0] + L.coefficients[1] * p.args[1] + L.coefficients[2]) * L.coefficients[1]
    if (testvalue >= 0):
        return testvalue
    return -1


# a, b, c are the coefficients of the in-equility ax + by + c <= 0
def isSatisfyInEquility(p, line):
    a = line.coefficients[0]
    b = line.coefficients[1]
    c = line.coefficients[2]

    val = a * p.args[0] + b * p.args[1] + c

    d = abs((a * p.args[0] + b * p.args[1] + c)) / ((a * a + b * b) ** 0.5)
    min_dist = 0.005

    if val == 0:
        return 0
    elif val < 0 and d > min_dist:
        return 1
    elif val > 0 and d > min_dist:
        return -1
        # Inequility supports
    return 0
    # Inequility not support


def pointDist(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)


# get the sign when p is put on the line L
def heightSign(p, L):
    return (L.coefficients[0] * p.args[0] + L.coefficients[1] * p.args[1] + L.coefficients[2])


# given three points, compute the triangle area
def triangle_area(a, b, c):
    return abs(
        0.5 * (a.loc[0] * (b.loc[1] - c.loc[1]) + b.loc[0] * (c.loc[1] - a.loc[1]) + c.loc[0] * (a.loc[1] - b.loc[1])))


# Update node coordinate to an optimized position
# minimize the squared error by quadratic programming
# def updateNode(nodes,i,j, values):
def updateNode(nodes_array, values_array):
    # Here the node is an internal node
    ####################################
    # (i-1,j-1)     (i-1,j)      (i-1,j+1)
    # |                |              |
    # (i,j-1)        (i,j)       (i,j+1)
    # |                |              |
    # (i+1,j-1)    (i+1,j)    (i+1,j+1)
    ####################################
    ###########################################
    # OPTIMIZATION CONSTRAINTS to feed into QUADPROG
    ###########################################

    ################# Rakib's Code ####################

    # Neighbourhood Point Construction

    p_top_left = nodes_array[0]
    p_left = nodes_array[1]
    p_bottom_left = nodes_array[2]

    p_top = nodes_array[3]
    p_middle = nodes_array[4]
    p_bottom = nodes_array[5]

    p_top_right = nodes_array[6]
    p_right = nodes_array[7]
    p_bottom_right = nodes_array[8]

    val_TL = values_array[0]
    val_BL = values_array[1]
    val_TR = values_array[2]
    val_BR = values_array[3]

    # Height Needed Calculation

    Area_TL_out = triangle_area(p_top_left, p_left, p_top)
    TL_Area_difference = val_TL - Area_TL_out
    TL_height_needed = (2 * TL_Area_difference) / Segment(p_left.loc, p_top.loc).length

    Area_TR_out = triangle_area(p_top_right, p_right, p_top)
    TR_Area_difference = val_TR - Area_TR_out
    TR_height_needed = (2 * TR_Area_difference) / Segment(p_right.loc, p_top.loc).length

    Area_BL_out = triangle_area(p_left, p_bottom_left, p_bottom)
    BL_Area_difference = val_BL - Area_BL_out
    BL_height_needed = (2 * BL_Area_difference) / Segment(p_left.loc, p_bottom.loc).length

    Area_BR_out = triangle_area(p_bottom_right, p_right, p_bottom)
    BR_Area_difference = val_BR - Area_BR_out
    BR_height_needed = (2 * BR_Area_difference) / Segment(p_right.loc, p_bottom.loc).length

    # Line Formation

    # Top Left Diagonal Line, Vertical Line and Horizontal Line
    TL_D_Line = Line(p_left.loc, p_top.loc)
    TL_V_Line = Line(p_top_left.loc, p_left.loc)
    TL_H_Line = Line(p_top_left.loc, p_top.loc)

    TR_D_Line = Line(p_right.loc, p_top.loc)
    TR_V_Line = Line(p_top_right.loc, p_right.loc)
    TR_H_Line = Line(p_top_right.loc, p_top.loc)

    BL_D_Line = Line(p_left.loc, p_bottom.loc)
    BL_V_Line = Line(p_bottom_left.loc, p_left.loc)
    BL_H_Line = Line(p_bottom_left.loc, p_bottom.loc)

    BR_D_Line = Line(p_right.loc, p_bottom.loc)
    BR_V_Line = Line(p_bottom_right.loc, p_right.loc)
    BR_H_Line = Line(p_bottom_right.loc, p_bottom.loc)

    # Constructing the in-equility for contrainst

    isOnTheDiagonalLine = False

    # if i==5 and j==7:
    #    temp = 10

    # TL lines
    checkSignVal = isSatisfyInEquility(p_middle.loc, TL_D_Line)
    if checkSignVal == 1:
        TL_D_coefficient_A = TL_D_Line.coefficients[0]
        TL_D_coefficient_B = TL_D_Line.coefficients[1]
        TL_D_coefficient_C = TL_D_Line.coefficients[2]

    elif checkSignVal == -1:
        TL_D_coefficient_A = -TL_D_Line.coefficients[0]
        TL_D_coefficient_B = -TL_D_Line.coefficients[1]
        TL_D_coefficient_C = -TL_D_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, TL_V_Line)
    if checkSignVal == 1:
        TL_V_coefficient_A = TL_V_Line.coefficients[0]
        TL_V_coefficient_B = TL_V_Line.coefficients[1]
        TL_V_coefficient_C = TL_V_Line.coefficients[2]
    elif checkSignVal == -1:
        TL_V_coefficient_A = -TL_V_Line.coefficients[0]
        TL_V_coefficient_B = -TL_V_Line.coefficients[1]
        TL_V_coefficient_C = -TL_V_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, TL_H_Line)
    if checkSignVal == 1:
        TL_H_coefficient_A = TL_H_Line.coefficients[0]
        TL_H_coefficient_B = TL_H_Line.coefficients[1]
        TL_H_coefficient_C = TL_H_Line.coefficients[2]
    elif checkSignVal == -1:
        TL_H_coefficient_A = -TL_H_Line.coefficients[0]
        TL_H_coefficient_B = -TL_H_Line.coefficients[1]
        TL_H_coefficient_C = -TL_H_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    # TR lines
    checkSignVal = isSatisfyInEquility(p_middle.loc, TR_D_Line)

    if checkSignVal == 1:
        TR_D_coefficient_A = TR_D_Line.coefficients[0]
        TR_D_coefficient_B = TR_D_Line.coefficients[1]
        TR_D_coefficient_C = TR_D_Line.coefficients[2]
    elif checkSignVal == -1:
        TR_D_coefficient_A = -TR_D_Line.coefficients[0]
        TR_D_coefficient_B = -TR_D_Line.coefficients[1]
        TR_D_coefficient_C = -TR_D_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, TR_V_Line)
    if checkSignVal == 1:
        TR_V_coefficient_A = TR_V_Line.coefficients[0]
        TR_V_coefficient_B = TR_V_Line.coefficients[1]
        TR_V_coefficient_C = TR_V_Line.coefficients[2]
    elif checkSignVal == -1:
        TR_V_coefficient_A = -TR_V_Line.coefficients[0]
        TR_V_coefficient_B = -TR_V_Line.coefficients[1]
        TR_V_coefficient_C = -TR_V_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, TR_H_Line)
    if checkSignVal == 1:
        TR_H_coefficient_A = TR_H_Line.coefficients[0]
        TR_H_coefficient_B = TR_H_Line.coefficients[1]
        TR_H_coefficient_C = TR_H_Line.coefficients[2]
    elif checkSignVal == -1:
        TR_H_coefficient_A = -TR_H_Line.coefficients[0]
        TR_H_coefficient_B = -TR_H_Line.coefficients[1]
        TR_H_coefficient_C = -TR_H_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    # BL lines
    checkSignVal = isSatisfyInEquility(p_middle.loc, BL_D_Line)

    if checkSignVal == 1:
        BL_D_coefficient_A = BL_D_Line.coefficients[0]
        BL_D_coefficient_B = BL_D_Line.coefficients[1]
        BL_D_coefficient_C = BL_D_Line.coefficients[2]
    elif checkSignVal == -1:
        BL_D_coefficient_A = -BL_D_Line.coefficients[0]
        BL_D_coefficient_B = -BL_D_Line.coefficients[1]
        BL_D_coefficient_C = -BL_D_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, BL_V_Line)
    if checkSignVal == 1:
        BL_V_coefficient_A = BL_V_Line.coefficients[0]
        BL_V_coefficient_B = BL_V_Line.coefficients[1]
        BL_V_coefficient_C = BL_V_Line.coefficients[2]
    elif checkSignVal == -1:
        BL_V_coefficient_A = -BL_V_Line.coefficients[0]
        BL_V_coefficient_B = -BL_V_Line.coefficients[1]
        BL_V_coefficient_C = -BL_V_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, BL_H_Line)
    if checkSignVal == 1:
        BL_H_coefficient_A = BL_H_Line.coefficients[0]
        BL_H_coefficient_B = BL_H_Line.coefficients[1]
        BL_H_coefficient_C = BL_H_Line.coefficients[2]
    elif checkSignVal == -1:
        BL_H_coefficient_A = -BL_H_Line.coefficients[0]
        BL_H_coefficient_B = -BL_H_Line.coefficients[1]
        BL_H_coefficient_C = -BL_H_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    # BR lines

    checkSignVal = isSatisfyInEquility(p_middle.loc, BR_D_Line)
    if checkSignVal == 1:
        BR_D_coefficient_A = BR_D_Line.coefficients[0]
        BR_D_coefficient_B = BR_D_Line.coefficients[1]
        BR_D_coefficient_C = BR_D_Line.coefficients[2]
    elif checkSignVal == -1:
        BR_D_coefficient_A = -BR_D_Line.coefficients[0]
        BR_D_coefficient_B = -BR_D_Line.coefficients[1]
        BR_D_coefficient_C = -BR_D_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, BR_V_Line)
    if checkSignVal == 1:
        BR_V_coefficient_A = BR_V_Line.coefficients[0]
        BR_V_coefficient_B = BR_V_Line.coefficients[1]
        BR_V_coefficient_C = BR_V_Line.coefficients[2]
    elif checkSignVal == -1:
        BR_V_coefficient_A = -BR_V_Line.coefficients[0]
        BR_V_coefficient_B = -BR_V_Line.coefficients[1]
        BR_V_coefficient_C = -BR_V_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    checkSignVal = isSatisfyInEquility(p_middle.loc, BR_H_Line)
    if checkSignVal == 1:
        BR_H_coefficient_A = BR_H_Line.coefficients[0]
        BR_H_coefficient_B = BR_H_Line.coefficients[1]
        BR_H_coefficient_C = BR_H_Line.coefficients[2]
    elif checkSignVal == -1:
        BR_H_coefficient_A = -BR_H_Line.coefficients[0]
        BR_H_coefficient_B = -BR_H_Line.coefficients[1]
        BR_H_coefficient_C = -BR_H_Line.coefficients[2]
    else:
        # On the line
        isOnTheDiagonalLine = True

    if isOnTheDiagonalLine is True:
        return Point(-1, -1)

    # The following code computes G and h. They just stores the
    # consraints that the coordinate of (i,j) must lie inside
    # the quadrangle defined by the four bold black lines

    G = np.array([[TL_D_coefficient_A, TL_D_coefficient_B],
                  [TR_D_coefficient_A, TR_D_coefficient_B],
                  [BL_D_coefficient_A, BL_D_coefficient_B],
                  [BR_D_coefficient_A, BR_D_coefficient_B],

                  [TL_H_coefficient_A, TL_H_coefficient_B],
                  [TL_V_coefficient_A, TL_V_coefficient_B],

                  [BL_H_coefficient_A, BL_H_coefficient_B],
                  [BL_V_coefficient_A, BL_V_coefficient_B],

                  [TR_H_coefficient_A, TR_H_coefficient_B],
                  [TR_V_coefficient_A, TR_V_coefficient_B],

                  [BR_H_coefficient_A, BR_H_coefficient_B],
                  [BR_V_coefficient_A, BR_V_coefficient_B]],
                 dtype=np.dtype('Float64'))

    h = np.array([-TL_D_coefficient_C, -TR_D_coefficient_C, -BL_D_coefficient_C, -BR_D_coefficient_C,
                  -TL_H_coefficient_C, -TL_V_coefficient_C,
                  -BL_H_coefficient_C, -BL_V_coefficient_C,
                  -TR_H_coefficient_C, -TR_V_coefficient_C,
                  -BR_H_coefficient_C, -BR_V_coefficient_C],
                 dtype=np.dtype('Float64')).reshape((12,))

    # Construction of OPTIMIZATION EQUATION to feed into QUADPROG
    ###########################################
    # we now focus on the optimization matrix
    # Assume the coefficients of a bold black line are a,b,c.
    # Then the optimization should be like [a/{a^2+b^2} b/{a^2+b^2}] [x y]' < [ - c/{a^2+b^2} \pm height_difference]
    # Therefore, we update the coefficients - See the pdf for explanation

    denominator_TL_D = ((TL_D_Line.coefficients[0] ** 2 + TL_D_Line.coefficients[1] ** 2) ** (0.5))
    denominator_TR_D = ((TR_D_Line.coefficients[0] ** 2 + TR_D_Line.coefficients[1] ** 2) ** (0.5))
    denominator_BL_D = ((BL_D_Line.coefficients[0] ** 2 + BL_D_Line.coefficients[1] ** 2) ** (0.5))
    denominator_BR_D = ((BR_D_Line.coefficients[0] ** 2 + BR_D_Line.coefficients[1] ** 2) ** (0.5))

    constraint_TLa = TL_D_Line.coefficients[0] / denominator_TL_D
    constraint_TLb = TL_D_Line.coefficients[1] / denominator_TL_D
    constraint_TLc = TL_D_Line.coefficients[2] / denominator_TL_D

    constraint_TRa = TR_D_Line.coefficients[0] / denominator_TR_D
    constraint_TRb = TR_D_Line.coefficients[1] / denominator_TR_D
    constraint_TRc = TR_D_Line.coefficients[2] / denominator_TR_D

    constraint_BLa = BL_D_Line.coefficients[0] / denominator_BL_D
    constraint_BLb = BL_D_Line.coefficients[1] / denominator_BL_D
    constraint_BLc = BL_D_Line.coefficients[2] / denominator_BL_D

    constraint_BRa = BR_D_Line.coefficients[0] / denominator_BR_D
    constraint_BRb = BR_D_Line.coefficients[1] / denominator_BR_D
    constraint_BRc = BR_D_Line.coefficients[2] / denominator_BR_D

    TL_height_needed = TL_height_needed if TL_height_needed > 0 else 0
    TR_height_needed = TR_height_needed if TR_height_needed > 0 else 0
    BL_height_needed = BL_height_needed if BL_height_needed > 0 else 0
    BR_height_needed = BR_height_needed if BR_height_needed > 0 else 0

    if (heightSign(p_middle.loc, TL_D_Line) >= 0):
        constraint_TLc = TL_height_needed - constraint_TLc
    elif (heightSign(p_middle.loc, TL_D_Line) < 0):
        TL_height_needed = - TL_height_needed
        constraint_TLc = TL_height_needed - constraint_TLc
    if (heightSign(p_middle.loc, TR_D_Line) >= 0):
        constraint_TRc = TR_height_needed - constraint_TRc
    elif (heightSign(p_middle.loc, TR_D_Line) < 0):
        TR_height_needed = - TR_height_needed
        constraint_TRc = TR_height_needed - constraint_TRc
    if (heightSign(p_middle.loc, BL_D_Line) >= 0):
        constraint_BLc = BL_height_needed - constraint_BLc
    elif (heightSign(p_middle.loc, BL_D_Line) < 0):
        BL_height_needed = - BL_height_needed
        constraint_BLc = BL_height_needed - constraint_BLc
    if (heightSign(p_middle.loc, BR_D_Line) >= 0):
        constraint_BRc = BR_height_needed - constraint_BRc
    elif (heightSign(p_middle.loc, BR_D_Line) < 0):
        BR_height_needed = - BR_height_needed
        constraint_BRc = BR_height_needed - constraint_BRc

    M = np.array([[constraint_TLa, constraint_TLb],
                  [constraint_TRa, constraint_TRb],
                  [constraint_BLa, constraint_BLb],
                  [constraint_BRa, constraint_BRb]], dtype=np.dtype('Float64'))
    P = np.dot(M.T, M)
    q = -np.dot(M.T, np.array([constraint_TLc, constraint_TRc, constraint_BLc, constraint_BRc],
                              dtype=np.dtype('Float64'))).reshape((2,))

    # solve the quadratic programming

    # output = quadprog_solve_qp(P, q, G, h)
    output = cvxopt_solve_qp(P, q, G, h)

    point_min_distance = 0.05

    if output is None:
        return Point(-1, -1)

    if pointDist(p_left.loc[0], p_left.loc[1], output[0], output[1]) < point_min_distance:
        return Point(-1, -1)

    if pointDist(p_top.loc[0], p_top.loc[1], output[0], output[1]) < point_min_distance:
        return Point(-1, -1)

    if pointDist(p_right.loc[0], p_right.loc[1], output[0], output[1]) < point_min_distance:
        return Point(-1, -1)

    if pointDist(p_bottom.loc[0], p_bottom.loc[1], output[0], output[1]) < point_min_distance:
        return Point(-1, -1)

    poly = Polygon((p_top.loc[0], p_top.loc[1]), (p_left.loc[0], p_left.loc[1]), (p_bottom.loc[0], p_bottom.loc[1]),
                   (p_right.loc[0], p_right.loc[1]))

    if poly.encloses_point(Point(p_middle.loc[0], p_middle.loc[1])) is False:
        return Point(-1, -1)

    return_point = Point(output[0], output[1])
    return return_point


def updateBoundaryNode(p, A1, A2, V1, V2, line_a, line_b, line_c):
    changed_area = A2 - (V2 * (A1 + A2) / (V1 + V2))
    shortest_distance = abs(line_a * p.args[0] + line_b * p.args[1] + line_c) / (line_a ** 2 + line_b ** 2) ** 0.5
    changed_coordinate = 2 * changed_area / shortest_distance
    return changed_coordinate


def point_in_poly(x, y, poly):
    BL_Line = Line(poly[0], poly[1])
    BR_Line = Line(poly[0], poly[3])
    TL_Line = Line(poly[1], poly[2])
    TR_Line = Line(poly[2], poly[3])
    if getSign(Point(x, y), TL_Line) > 0:
        return False
    if getSign(Point(x, y), TR_Line) > 0:
        return False
    if getSign(Point(x, y), BL_Line) < 0:
        return False
    if getSign(Point(x, y), BR_Line) < 0:
        return False
    return True


################# Error Calculation ###############################

def all_error_calc(values, nodes, grid_count_horizontal, grid_count_vertical, estimation_time, out_log_file_name,
                   iteration, stage=-1, preprocessing_time=-1):
    updated_values = np.zeros((grid_count_horizontal, grid_count_vertical))
    aspect_ratio_m1 = np.zeros((grid_count_horizontal, grid_count_vertical))
    aspect_ratio_m2 = np.zeros((grid_count_horizontal, grid_count_vertical))

    for ii in range(grid_count_horizontal):
        for jj in range(grid_count_vertical):

            try:
                p_bottom_left = nodes[ii][jj]
                p_left = nodes[ii][jj + 1]
                p_middle = nodes[ii + 1][jj + 1]
                p_bottom = nodes[ii + 1][jj]

                pol = Polygon(p_bottom.loc, p_middle.loc, p_left.loc, p_bottom_left.loc)
                area = abs(pol.area)
                updated_values[ii][jj] = round(area, 2)

                bounding_box = pol.bounds;
                height = abs(bounding_box[3] - bounding_box[1])
                width = abs(bounding_box[2] - bounding_box[0])
                aspect_ratio_m1[ii][jj] = height / width
                aspect_ratio_m2[ii][jj] = min(height, width) / max(height, width)

            except ValueError:
                # print("No solution...")
                area = 0

            except AttributeError:
                area = 0

    N = grid_count_horizontal * grid_count_vertical

    updated_diff = (abs(updated_values - values))
    error_list = updated_diff / values

    total_error = (np.sum(error_list) / N)

    fast_flow_error = (updated_values / values) - 1
    total_fast_flow_error = (np.sum(fast_flow_error) / N)

    squared_error_list = (updated_diff / values) ** 2
    total_rmse_error = (np.sum(squared_error_list) / N) ** 0.5
    weighter_rmse_error = (np.sum(values * squared_error_list)) ** 0.5

    total_mqe_error = ((np.sum(squared_error_list)) ** 0.5) / N

    updated_squared_error_list = (updated_diff / (updated_values + values)) ** 2
    updated_total_mqe_error = ((np.sum(updated_squared_error_list)) ** 0.5) / N

    average_aspect_ratio_m1 = np.average(aspect_ratio_m1)
    average_aspect_ratio_m2 = np.average(aspect_ratio_m2)

    print("Error Rate : " + str(round(total_error, 4)))
    print("Updated Error Rate : " + str(round(total_fast_flow_error, 4)))
    print("RMSE Error Rate : " + str(round(total_rmse_error, 4)))
    print("Weighted RMSE Error Rate : " + str(round(weighter_rmse_error, 4)))
    print("MQE Error Rate : " + str(round(total_mqe_error, 4)))
    print("Updated MQE Error Rate : " + str(round(updated_total_mqe_error, 6)))
    print("Average Aspect Ratio (height/width) : " + str(round(average_aspect_ratio_m1, 6)))
    print("Average Aspect Ratio min(height/width)/max(height/width) : " + str(round(average_aspect_ratio_m2, 6)))
    if preprocessing_time != -1:
        print("Pre Processing Time: " + str(round(preprocessing_time, 4)) + " sec")

    print("Processing Time: " + str(round(estimation_time, 4)) + " sec")
    print("------------------------------------------")

    out_file_name = "output/out_log_" + out_log_file_name + ".txt"
    output_txt_file = open(out_file_name, "a")
    if stage != -1:
        output_txt_file.write(str(stage) + ", ")
    output_txt_file.write(str(iteration) + ", ")
    output_txt_file.write(str(round(total_error, 4)) + ", ")
    output_txt_file.write(str(round(total_fast_flow_error, 4)) + ", ")
    output_txt_file.write(str(round(total_rmse_error, 4)) + ", ")
    output_txt_file.write(str(round(total_mqe_error, 4)) + ", ")
    output_txt_file.write(str(round(updated_total_mqe_error, 6)) + ", ")
    output_txt_file.write(str(round(average_aspect_ratio_m1, 6)) + ", ")
    output_txt_file.write(str(round(average_aspect_ratio_m2, 6)) + ", ")
    if preprocessing_time != -1:
        output_txt_file.write(str(round(preprocessing_time, 4)) + ", ")

    output_txt_file.write(str(round(estimation_time, 4)) + "\n")
    output_txt_file.close()

    return round(total_rmse_error, 4), error_list, updated_values, values


def all_error_print(values, nodes, grid_count_horizontal, grid_count_vertical, estimation_time, out_log_file_name,
                    iteration=-1, stage=-1):
    updated_values = np.zeros((grid_count_horizontal, grid_count_vertical))
    aspect_ratio_m1 = np.zeros((grid_count_horizontal, grid_count_vertical))
    aspect_ratio_m2 = np.zeros((grid_count_horizontal, grid_count_vertical))

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical):

            try:
                p = nodes[i][j]
                p_right = nodes[i + 1][j]
                p_right_bottom = nodes[i + 1][j + 1]
                p_bottom = nodes[i][j + 1]

                pol = Polygon(p.loc, p_right.loc, p_right_bottom.loc, p_bottom.loc)
                area = abs(pol.area)
                updated_values[j][i] = area

                bounding_box = pol.bounds
                height = abs(bounding_box[3] - bounding_box[1])
                width = abs(bounding_box[2] - bounding_box[0])
                aspect_ratio_m1[j][i] = height / width
                aspect_ratio_m2[j][i] = min(height, width) / max(height, width)

            except ValueError:
                # print("No solution...")
                area = 0

            except AttributeError:
                area = 0

    N = grid_count_horizontal * grid_count_vertical

    updated_diff = (abs(updated_values - values))
    error_list = updated_diff / values

    total_error = (np.sum(error_list) / N)

    fast_flow_error = (updated_values / values) - 1
    total_fast_flow_error = (np.sum(fast_flow_error) / N)

    squared_error_list = (updated_diff / values) ** 2
    total_rmse_error = (np.sum(squared_error_list) / N) ** 0.5
    weighter_rmse_error = (np.sum(values * squared_error_list)) ** 0.5

    total_mqe_error = ((np.sum(squared_error_list)) ** 0.5) / N

    updated_squared_error_list = (updated_diff / (updated_values + values)) ** 2
    updated_total_mqe_error = ((np.sum(updated_squared_error_list)) ** 0.5) / N

    average_aspect_ratio_m1 = np.average(aspect_ratio_m1)
    average_aspect_ratio_m2 = np.average(aspect_ratio_m2)

    print("Error Rate : " + str(round(total_error, 4)))
    print("Updated Error Rate : " + str(round(total_fast_flow_error, 4)))
    print("RMSE Error Rate : " + str(round(total_rmse_error, 4)))
    print("Weighted RMSE Error Rate : " + str(round(weighter_rmse_error, 4)))
    print("MQE Error Rate : " + str(round(total_mqe_error, 4)))
    print("Updated MQE Error Rate : " + str(round(updated_total_mqe_error, 6)))
    print("Average Aspect Ratio (height/width) : " + str(round(average_aspect_ratio_m1, 6)))
    print("Average Aspect Ratio min(height/width)/max(height/width) : " + str(round(average_aspect_ratio_m2, 6)))
    print("Processing Time: " + str(round(estimation_time, 4)) + " sec")
    print("------------------------------------------")

    out_file_name = "output/" + out_log_file_name + "_log.txt"
    output_txt_file = open(out_file_name, "a")
    if stage != -1:
        output_txt_file.write(str(stage) + ", ")
    if iteration != -1:
        output_txt_file.write(str(iteration) + ", ")
    output_txt_file.write(str(round(total_error, 4)) + ", ")
    output_txt_file.write(str(round(total_fast_flow_error, 4)) + ", ")
    output_txt_file.write(str(round(total_rmse_error, 4)) + ", ")
    output_txt_file.write(str(round(total_mqe_error, 4)) + ", ")
    output_txt_file.write(str(round(updated_total_mqe_error, 6)) + ", ")
    output_txt_file.write(str(round(average_aspect_ratio_m1, 6)) + ", ")
    output_txt_file.write(str(round(average_aspect_ratio_m2, 6)) + ", ")
    output_txt_file.write(str(round(estimation_time, 4)) + "\n")
    output_txt_file.close()

    return round(total_rmse_error, 4), error_list, updated_values, values


################# Error Calculation ###############################

################# Polygon Draw ####################################

def poly_draw(filename, it, im_size, nodes, grid_count_horizontal, grid_count_vertical):
    # Create an empty image

    factor_x = int(im_size[0] / grid_count_horizontal)
    factor_y = int(im_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    # Define a set of random color to color the faces
    colour = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    # this is the factor to scale up the image

    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical, 0, -1):
            pol = Polygon(Point2D(nodes[i][j].loc.x, (grid_count_vertical - nodes[i][j].loc.y)),
                          Point2D(nodes[i][j - 1].loc.x, (grid_count_vertical - nodes[i][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j - 1].loc.x, (grid_count_vertical - nodes[i + 1][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j].loc.x, (grid_count_vertical - nodes[i + 1][j].loc.y)))
            area = abs(pol.area)

            '''
            if area > imp_cell_threshold:
                colour = ["green", "yellow"]
            else:                
                colour = ["#fee0d2", "#de2d26", "#e5f5e0", "#31a354"]
                #colour = ["red", "blue"]

            d.polygon([tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                          , tuple(Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                          , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                          , tuple(Point2D(nodes[i+1][j].loc.x * factor_x, (grid_count_vertical - nodes[i+1][j].loc.y) * factor_y))]
                      , fill=colour[(0 if i % 2 == 0 else 2) + (j % 2)], outline="black")
            '''
            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                    , tuple(
                    Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                , fill="black", outline="cyan")
            # d.text(Point2D(nodes[i][j].loc.x * factor_x,
            #              (grid_count_vertical - nodes[i][j].loc.y) * factor_y), str(i)+ "_" +str(j))
    imagestring = 'output//' + filename + '_table_cartogram-' + str(grid_count_horizontal) + '_' + str(
        grid_count_vertical) \
                  + ' iterations' + str((it)) + '.png'
    img.save(imagestring)


def poly_draw_color(filename, it, im_size, nodes, maxColorListByGrid, grid_count_horizontal, grid_count_vertical):
    # Create an empty image

    factor_x = int(im_size[0] / grid_count_horizontal)
    factor_y = int(im_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    # Define a set of random color to color the faces
    colour = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    # this is the factor to scale up the image

    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical, 0, -1):
            pol = Polygon(Point2D(nodes[i][j].loc.x, (grid_count_vertical - nodes[i][j].loc.y)),
                          Point2D(nodes[i][j - 1].loc.x, (grid_count_vertical - nodes[i][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j - 1].loc.x, (grid_count_vertical - nodes[i + 1][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j].loc.x, (grid_count_vertical - nodes[i + 1][j].loc.y)))
            area = abs(pol.area)

            if area > imp_cell_threshold:
                colour = ["green", "yellow"]
            else:
                colour = ["red", "blue"]

            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                    , tuple(
                    Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                , fill=maxColorListByGrid[i][grid_count_vertical - j] + "50")
            # d.text(Point2D(nodes[i][j].loc.x * factor_x,
            #              (grid_count_vertical - nodes[i][j].loc.y) * factor_y), str(i)+ "_" +str(j))
    imagestring = 'output//' + filename + '_table_weighted_cartogram-' + str(grid_count_horizontal) + '_' + str(
        grid_count_vertical) \
                  + ' iterations' + str((it)) + '.png'
    img.save(imagestring)


def poly_draw_grid_color(filename, it, im_size, nodes, grid_count_horizontal, grid_count_vertical):
    # Create an empty image

    factor_x = int(im_size[0] / grid_count_horizontal)
    factor_y = int(im_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    # Define a set of random color to color the faces
    colour = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    # this is the factor to scale up the image

    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical, 0, -1):
            pol = Polygon(Point2D(nodes[i][j].loc.x, (grid_count_vertical - nodes[i][j].loc.y)),
                          Point2D(nodes[i][j - 1].loc.x, (grid_count_vertical - nodes[i][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j - 1].loc.x, (grid_count_vertical - nodes[i + 1][j - 1].loc.y)),
                          Point2D(nodes[i + 1][j].loc.x, (grid_count_vertical - nodes[i + 1][j].loc.y)))
            area = abs(pol.area)

            if area < imp_cell_threshold:
                colour = ["#f9f9f9", "#F2E0D3"]
            else:
                colour = ["#c42026", "#da4726"]

            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                    , tuple(
                    Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                , fill=colour[(0 if (i + j) % 2 == 0 else 1)], outline="black")

    imagestring = 'output//' + filename + '_table_cartogram-' + str(grid_count_horizontal) + '_' + str(
        grid_count_vertical) \
                  + ' iterations' + str((it)) + '.png'
    img.save(imagestring)


def poly_draw_grid_color_wtih_scale(filename, it, im_size, nodes, values, grid_count_horizontal, grid_count_vertical):
    # Create an empty image

    factor_x = int(im_size[0] / grid_count_horizontal)
    factor_y = int(im_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    # Define a set of random color to color the faces
    colour = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    # this is the factor to scale up the image

    color_count = 10
    # current_palette = sns.light_palette("#DE2D26", n_colors=color_count)
    # current_palette = ["#fee0d2", "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"]
    current_palette = ["#fee0d2", "#fee8c8", "#fdbb84", "#ef6548", "#b30000", "#7f0000", "#7f0000", "#7f0000",
                       "#7f0000", "#7f0000"]

    max_val = np.max(values)
    min_val = np.min(values)

    # diff = max_val - min_val
    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical, 0, -1):

            # color_index = int((values[i][j-1] - min_val) / diff * (color_count) - 0.0001)
            # color_hex = "#{:02x}{:02x}{:02x}".format(int(current_palette[color_index][0]*255),int(current_palette[color_index][1]*255),int(current_palette[color_index][2]*255))
            color_hex = "#000000"

            if values[i][j - 1] != min_val:
                color_val = 155 + int((values[i][j - 1] - min_val) / (max_val - min_val) * 100.00)
                color_hex = "#{:02x}{:02x}{:02x}".format(color_val,
                                                         color_val,
                                                         color_val)

            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                    , tuple(
                    Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                , fill=color_hex)
            # , fill=current_palette[color_index], outline="black")

    imagestring = 'output//' + filename + '_table_cartogram-' + str(grid_count_horizontal) + '_' \
                  + str(grid_count_vertical) + ' iterations' + str((it)) + '.png'
    img.save(imagestring)

    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=15))

    img_blurred.save('output/test_blurred_image.png')
    image_contour = np.asarray(img_blurred)

    image_contour = np.flipud(image_contour)
    Z = image_contour[:, :, 0]

    plt.contourf(Z, levels=50, cmap='RdGy')

    plt.savefig('output/test_image.png')


def poly_draw_top_to_bottom(filename, im_size, nodes, grid_count_horizontal, grid_count_vertical):
    # Create an empty image

    factor_x = int(im_size[0] / grid_count_horizontal)
    factor_y = int(im_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)

    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical):
            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, nodes[i][j].loc.y * factor_y))
                    , tuple(Point2D(nodes[i][j + 1].loc.x * factor_x, nodes[i][j + 1].loc.y * factor_y))
                    , tuple(Point2D(nodes[i + 1][j + 1].loc.x * factor_x, nodes[i + 1][j + 1].loc.y * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x, nodes[i + 1][j].loc.y * factor_y))]
                , fill="black", outline="cyan")

    actualFileName = filename + '_' + str(grid_count_horizontal) + '_' + str(
        grid_count_vertical) + '.png'
    imagestring = 'output//' + actualFileName
    img.save(imagestring)
    return actualFileName


################# Polygon Draw ####################################

################ image drawing ####################################

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def imageDraw(input_image, nodes, filename, grid_count_horizontal, grid_count_vertical):
    output_image_size = input_image.size

    img = Image.new('RGBA', (output_image_size[0], output_image_size[1]), color=(73, 109, 137, 256))

    splitted_image = []
    for i in range(grid_count_horizontal):
        im = []
        for j in range(grid_count_vertical):
            block_width = input_image.size[0] / grid_count_horizontal
            block_height = input_image.size[1] / grid_count_vertical
            upper_left_x = i * block_width
            upper_left_y = j * block_height
            lower_right_x = upper_left_x + block_width
            lower_right_y = upper_left_y + block_height

            sub_image = input_image.crop((upper_left_x, upper_left_y, lower_right_x, lower_right_y))

            # sub_image.save("output/image/input_" + str(i) + "_" + str(j) + ".png", "PNG")
            im.append(sub_image)
        splitted_image.append(im)

    factor_x = int(output_image_size[0] / grid_count_horizontal)
    factor_y = int(output_image_size[1] / grid_count_vertical)

    piece_count = 0

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical):
            # Where i and j from image array's index

            # i=0
            # j=7
            j_conv = grid_count_vertical - j

            p_tl = nodes[i][j_conv].loc
            p_bl = nodes[i][j_conv - 1].loc
            p_tr = nodes[i + 1][j_conv].loc
            p_br = nodes[i + 1][j_conv - 1].loc

            im_temp = Image.new('RGBA', (output_image_size[0], output_image_size[1]), color=(0, 0, 0, 0))
            im_temp.paste(splitted_image[i][j], (i * factor_x, j * factor_y,
                                                 (i + 1) * factor_x, (j + 1) * factor_y))

            coeffs = find_coeffs([(m.ceil(p_tl.x * factor_x), m.ceil((grid_count_vertical - p_tl.y) * factor_y)),
                                  (m.ceil(p_tr.x * factor_x), m.ceil((grid_count_vertical - p_tr.y) * factor_y)),
                                  (m.ceil(p_br.x * factor_x), m.ceil((grid_count_vertical - p_br.y) * factor_y)),
                                  (m.ceil((p_bl.x - 0.1) * factor_x),
                                   m.ceil((grid_count_vertical - p_bl.y) * factor_y))],
                                 [(i * factor_x, j * factor_y),
                                  ((i + 1) * factor_x, j * factor_y),
                                  ((i + 1) * factor_x, (j + 1) * factor_y),
                                  (i * factor_x, (j + 1) * factor_y)])

            im = im_temp.transform((output_image_size[0], output_image_size[1]), Image.PERSPECTIVE, coeffs,
                                   Image.BICUBIC)
            img.paste(im, (0, 0, output_image_size[0], output_image_size[1]), im)

            piece_count += 1
            # print("Output image assembled : " + str(piece_count) + "/"
            #      + str(grid_count_horizontal * grid_count_vertical))

    img.save('output/' + filename + '.png', 'PNG')


def newImageDraw(input_image, nodes, filename, grid_count_horizontal, grid_count_vertical, is_draw_border=False):
    output_image_size = input_image.size
    factor_x = int(output_image_size[0] / grid_count_horizontal)
    factor_y = int(output_image_size[1] / grid_count_vertical)

    dst = []
    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical, -1, -1):
            dst.append(
                [m.ceil(nodes[i][j].loc.x * factor_x), m.ceil((grid_count_vertical - nodes[i][j].loc.y) * factor_y)])

    dst = np.array(dst)

    # dst = np.array([[0, 0], [0, 400], [400, 0], [400, 400]])

    src = []

    # print(output_image_size)
    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical + 1):
            block_width = input_image.size[0] / grid_count_horizontal
            block_height = input_image.size[1] / grid_count_vertical
            upper_left_x = i * block_width
            upper_left_y = j * block_height
            src.append([m.ceil(upper_left_x), m.ceil(upper_left_y)])
    src = np.array(src)
    # src = np.array([[0, 0], [0, 400], [400, 0], [200, 400]])
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    out = warp(input_image, tform, output_shape=(output_image_size[0], output_image_size[1]))

    # fig, ax = plt.subplots()
    # ax.imshow(out)
    # ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    # ax.axis((0, out_cols, out_rows, 0))

    output_path = 'output/' + filename + '.png'

    plt.imsave(output_path, out)

    if is_draw_border:
        # Draw all the faces
        im = Image.open(output_path).convert('RGB')
        d = ImageDraw.Draw(im)
        for i in range(grid_count_horizontal):
            for j in range(grid_count_vertical, 0, -1):
                d.line(
                    [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                        , tuple(Point2D(nodes[i][j - 1].loc.x * factor_x,
                                        (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                        , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                        (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                        , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                        (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                    , fill="white", width=7)

        actualFileName = 'output/' + filename + '_with_border.png'
        im.save(actualFileName)

    return output_path


def newImageDrawTopToBottom(input_image, nodes, filename, grid_count_horizontal, grid_count_vertical):
    output_image_size = input_image.size
    factor_x = int(output_image_size[0] / grid_count_horizontal)
    factor_y = int(output_image_size[1] / grid_count_vertical)

    dst = []
    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical + 1):
            dst.append([m.ceil(nodes[i][j].loc.x * factor_x), m.ceil(nodes[i][j].loc.y * factor_y)])

    dst = np.array(dst)

    # dst = np.array([[0, 0], [0, 400], [400, 0], [400, 400]])

    src = []

    # print(output_image_size)
    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical + 1):
            block_width = input_image.size[0] / grid_count_horizontal
            block_height = input_image.size[1] / grid_count_vertical
            upper_left_x = i * block_width
            upper_left_y = j * block_height
            src.append([m.ceil(upper_left_x), m.ceil(upper_left_y)])
    src = np.array(src)
    # src = np.array([[0, 0], [0, 400], [400, 0], [200, 400]])
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    out = warp(input_image, tform, output_shape=(output_image_size[0], output_image_size[1]))

    # fig, ax = plt.subplots()
    # ax.imshow(out)
    # ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    # ax.axis((0, out_cols, out_rows, 0))
    output_path = 'output/' + filename + '.png'

    plt.imsave(output_path, out)

    return output_path


def imageDrawBrighnessChannelTopToBottom(input_img_file, nodes, weights, filename, grid_count_horizontal,
                                         grid_count_vertical):
    input_image = Image.open(input_img_file)

    output_image_size = input_image.size
    x_unit = int(output_image_size[0] / grid_count_horizontal)
    y_unit = int(output_image_size[1] / grid_count_vertical)

    min_sat_range = 0
    max_sat_range = 255
    min_sat_val = np.min(weights)
    max_sat_val = np.max(weights)

    min_light_range = 0
    max_light_range = 255
    min_light_val = np.min(weights)
    max_light_val = np.max(weights)

    out = input_image.convert("HSV")
    pixels = out.load()
    x_grid = 0
    y_grid = 0
    y_grid_prev_row = 0

    is_within_polygon = False
    is_on_top_polygon = 1000

    for j in range(input_image.size[1] - 1, -1, -1):  # for every col:
        for i in range(input_image.size[1]):  # For every row

            if i == 0:
                y_grid_prev_row = y_grid

            x_coord = i / output_image_size[0] * grid_count_horizontal
            y_coord = (input_image.size[1] - j) / output_image_size[1] * grid_count_vertical

            p_top_left = nodes[x_grid][y_grid].loc
            p_top_right = nodes[x_grid + 1][y_grid].loc
            p_bottom_right = nodes[x_grid + 1][y_grid + 1].loc
            p_bottom_left = nodes[x_grid][y_grid + 1].loc

            poly = Polygon((p_top_left[0], (p_top_left[1])),
                           (p_top_right[0], (p_top_right[1])),
                           (p_bottom_right[0], (p_bottom_right[1])),
                           (p_bottom_left[0], (p_bottom_left[1])))

            is_within_polygon = poly.encloses_point(Point(x_coord, y_coord))
            is_on_top_polygon = poly.distance(Point(x_coord, y_coord))

            if not is_within_polygon and is_on_top_polygon > 0:
                flag_in = False

                ii = x_grid + 1
                jj = y_grid

                # Right Side Grid
                if 0 <= ii < grid_count_horizontal:
                    if 0 <= jj < grid_count_vertical:
                        p_top_left = nodes[ii][jj].loc
                        p_top_right = nodes[ii + 1][jj].loc
                        p_bottom_right = nodes[ii + 1][jj + 1].loc
                        p_bottom_left = nodes[ii][jj + 1].loc
                        poly = Polygon((p_top_left[0], (p_top_left[1])),
                                       (p_top_right[0], (p_top_right[1])),
                                       (p_bottom_right[0], (p_bottom_right[1])),
                                       (p_bottom_left[0], (p_bottom_left[1])))

                        is_within_polygon = poly.encloses_point(Point(x_coord, y_coord))
                        is_on_top_polygon = poly.distance(Point(x_coord, y_coord))
                        if is_within_polygon or is_on_top_polygon == 0:
                            x_grid = ii
                            y_grid = jj
                            flag_in = True

                # Bottom Side Grid
                if not flag_in:
                    ii = x_grid
                    jj = y_grid + 1

                    if 0 <= ii < grid_count_horizontal:
                        if 0 <= jj < grid_count_vertical:
                            p_top_left = nodes[ii][jj].loc
                            p_top_right = nodes[ii + 1][jj].loc
                            p_bottom_right = nodes[ii + 1][jj + 1].loc
                            p_bottom_left = nodes[ii][jj + 1].loc
                            poly = Polygon((p_top_left[0], (p_top_left[1])),
                                           (p_top_right[0], (p_top_right[1])),
                                           (p_bottom_right[0], (p_bottom_right[1])),
                                           (p_bottom_left[0], (p_bottom_left[1])))

                            is_within_polygon = poly.encloses_point(Point(x_coord, y_coord))
                            is_on_top_polygon = poly.distance(Point(x_coord, y_coord))
                            if is_within_polygon or is_on_top_polygon == 0:
                                x_grid = ii
                                y_grid = jj
                                flag_in = True

                # Top Side Grid
                if not flag_in:
                    ii = x_grid
                    jj = y_grid + 1

                    # Bottom Side Grid
                    if 0 <= ii < grid_count_horizontal:
                        if 0 <= jj < grid_count_vertical:
                            p_top_left = nodes[ii][jj].loc
                            p_top_right = nodes[ii + 1][jj].loc
                            p_bottom_right = nodes[ii + 1][jj + 1].loc
                            p_bottom_left = nodes[ii][jj + 1].loc
                            poly = Polygon((p_top_left[0], (p_top_left[1])),
                                           (p_top_right[0], (p_top_right[1])),
                                           (p_bottom_right[0], (p_bottom_right[1])),
                                           (p_bottom_left[0], (p_bottom_left[1])))

                            is_within_polygon = poly.encloses_point(Point(x_coord, y_coord))
                            is_on_top_polygon = poly.distance(Point(x_coord, y_coord))
                            if is_within_polygon or is_on_top_polygon == 0:
                                x_grid = ii
                                y_grid = jj

            pix_val_light = min_light_range + (
                    weights[grid_count_vertical - 1 - y_grid][x_grid] - min_light_val) / (
                                    max_light_val - min_light_val) * (max_light_range - min_light_range)

            pix_val_sat = min_sat_range + (
                    weights[grid_count_vertical - 1 - y_grid][x_grid] - min_sat_val) / (
                                  max_sat_val - min_sat_val) * (max_sat_range - min_sat_range)

            pixels[i, j] = (int(pixels[i, j][0]),
                            int(pixels[i, j][1]),
                            int(pix_val_light))

        print("(i,j):(" + str(i) + "," + str(j) + ")")
        x_grid = 0
        y_grid = y_grid_prev_row
        out.show()
    # out.show()

    output_path = 'output/' + filename + '.png'
    out = out.convert("RGBA")
    out.save(output_path, 'PNG')
    return output_path


def getWeightForPixel(output_image_size, i_array, j_array, grid_count_horizontal, grid_count_vertical, nodes, weights,
                      thread_count):
    points_array = []

    for i in i_array:
        for j in j_array:

            y_coord = i / output_image_size[0] * grid_count_horizontal
            x_coord = j / output_image_size[1] * grid_count_vertical

            is_within_polygon = False
            is_on_top_polygon = 1000

            for x_grid in range(grid_count_horizontal):
                for y_grid in range(grid_count_vertical):

                    # p_top_left = nodes[x_grid][y_grid].loc
                    # p_top_right = nodes[x_grid + 1][y_grid].loc
                    # p_bottom_right = nodes[x_grid + 1][y_grid + 1].loc
                    # p_bottom_left = nodes[x_grid][y_grid + 1].loc

                    p_top_left = Point2D(nodes[x_grid][y_grid].loc.x,
                                         (grid_count_vertical - nodes[x_grid][y_grid].loc.y))
                    p_top_right = Point2D(nodes[x_grid + 1][y_grid].loc.x,
                                          (grid_count_vertical - nodes[x_grid + 1][y_grid].loc.y))
                    p_bottom_right = Point2D(nodes[x_grid + 1][y_grid + 1].loc.x,
                                             (grid_count_vertical - nodes[x_grid + 1][y_grid + 1].loc.y))
                    p_bottom_left = Point2D(nodes[x_grid][y_grid + 1].loc.x,
                                            (grid_count_vertical - nodes[x_grid][y_grid + 1].loc.y))

                    poly = Polygon((p_top_left[0], (p_top_left[1])),
                                   (p_top_right[0], (p_top_right[1])),
                                   (p_bottom_right[0], (p_bottom_right[1])),
                                   (p_bottom_left[0], (p_bottom_left[1])))

                    is_within_polygon = poly.encloses_point(Point(x_coord, y_coord))
                    is_on_top_polygon = poly.distance(Point(x_coord, y_coord))

                    if is_within_polygon or is_on_top_polygon <= 0:
                        if thread_count == 0:
                            print("(Thread,i,j):(" + str(thread_count) + "," + str(i) + "," + str(j) + ")")
                        points_array.append([i, j, weights[grid_count_horizontal - 1 - y_grid, x_grid]])

                    if is_within_polygon or is_on_top_polygon <= 0:
                        break
                if is_within_polygon or is_on_top_polygon <= 0:
                    break

    print("Thread : " + str(thread_count))
    return points_array


def imageDrawLightChannelTopToBottom_HLS(input_img_file, nodes, weights, filename, grid_count_horizontal,
                                         grid_count_vertical):
    input_image = Image.open(input_img_file)
    hls = create_hls_array(input_image)

    output_image_size = input_image.size

    cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    thread_result = []

    left_cpu_count = cpu_count
    x_pixel_left = output_image_size[0]
    x_start = 0

    for th in range(cpu_count):
        x_pixels_count = m.ceil(x_pixel_left * 1.00 / left_cpu_count)

        thread_result.append(pool.apply_async(getWeightForPixel,
                                              args=(output_image_size, range(x_start, x_start + x_pixels_count),
                                                    range(output_image_size[1]),
                                                    grid_count_horizontal, grid_count_vertical, nodes,
                                                    weights, th,)))
        left_cpu_count = left_cpu_count - 1
        x_pixel_left = x_pixel_left - x_pixels_count
        x_start = x_start + x_pixels_count

    pool.close()
    pool.join()

    for th in range(cpu_count):
        val_array = thread_result[th]._value
        for count in range(len(val_array)):
            index_x = val_array[count][0]
            index_y = val_array[count][1]
            pred_weight_val = val_array[count][2]

            # hls[index_x, index_y][1] = pred_weight_val

            percentage = 0.25
            actual_val = hls[index_x, index_y][1]
            if pred_weight_val >= actual_val:
                hls[index_x, index_y][1] = actual_val + pred_weight_val * percentage
            else:
                hls[index_x, index_y][1] = actual_val - pred_weight_val * percentage

    output_path = 'output/' + filename + '.png'
    # out = out.convert("RGBA")
    # out.save(output_path, 'PNG')

    new_image = image_from_hls_array(hls)
    new_image.save(output_path)

    return output_path


################ image drawing ####################################

################ Shape File Generation ############################
def shapeGenCarto4F(grid_count, values, nodes, output_file_name):
    w = shapefile.Writer(output_file_name, shapefile.POLYGON)
    for i in range(grid_count):
        for j in range(grid_count):
            w.poly(
                [[
                    [nodes[i][j].loc.x, nodes[i][j].loc.y],
                    [nodes[i][j + 1].loc.x, nodes[i][j + 1].loc.y],
                    [nodes[i + 1][j + 1].loc.x, nodes[i + 1][j + 1].loc.y],
                    [nodes[i + 1][j].loc.x, nodes[i + 1][j].loc.y],
                    [nodes[i][j].loc.x, nodes[i][j].loc.y]
                ]]
            )
            # w.poly([[[i, j], [i, j + 1], [i + 1, j + 1], [i + 1, j]]])

    w.field('ID', 'C', '40')
    w.field('POP', 'F', '12')

    for i in range(grid_count):
        for j in range(grid_count):
            id = str(i) + "_" + str(j)
            pop = str(values[i][j])
            # pop = 1
            w.record(id, pop)

    # w.save(output_file_name)


################ Shape File Generation ############################

################ Dat Gen - Max Flow generation ####################
def datGenMaxFlowGeneration(grid, values, nodes, output_file_name):
    file_name_boundary = output_file_name + ".gen"
    file_name_weight = output_file_name + ".dat"

    out_weight_file = open(file_name_weight, "w")
    out_boundary_file = open(file_name_boundary, "w")

    counter = 0

    for i in range(len(values)):
        for j in range(len(values[0])):
            # print(str(result_array[i][j]) + ",")
            id_name = "A_" + str(i) + "_" + str(j)
            out_weight_file.write(str(counter) + " " + str(values[i][j]) + " " + id_name)
            '''
            out_boundary_file.write(str(counter) + " " + id_name + "\n")
            out_boundary_file.write(str(i) + " " + str(j * (-1)) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str(j * (-1)) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str((j + 1) * (-1)) + "\n")
            out_boundary_file.write(str(i) + " " + str((j + 1) * (-1)) + "\n")
            out_boundary_file.write(str(i) + " " + str(j * (-1)) + "\n")
            '''

            out_boundary_file.write(str(counter) + " " + id_name + "\n")
            out_boundary_file.write(str(round(nodes[i][j].loc.x, 10)) + " " + str(round(nodes[i][j].loc.y, 10)) + "\n")
            out_boundary_file.write(
                str(round(nodes[i + 1][j].loc.x, 10)) + " " + str(round(nodes[i + 1][j].loc.y, 10)) + "\n")
            out_boundary_file.write(
                str(round(nodes[i + 1][j + 1].loc.x, 10)) + " " + str(round(nodes[i + 1][j + 1].loc.y, 10)) + "\n")
            out_boundary_file.write(
                str(round(nodes[i][j + 1].loc.x, 10)) + " " + str(round(nodes[i][j + 1].loc.y, 10)) + "\n")
            out_boundary_file.write(str(round(nodes[i][j].loc.x, 10)) + " " + str(round(nodes[i][j].loc.y, 10)) + "\n")

            out_boundary_file.write("END\n")

            counter += 1
            if counter != grid * grid:
                out_weight_file.write(str(("\n")))

    out_boundary_file.write("END\n")
    out_weight_file.close()
    out_boundary_file.close()


def create_hls_array(image):
    pixels = image.load()
    hls_array = np.empty(shape=(image.height, image.width, 3), dtype=float)

    for row in range(0, image.height):
        for column in range(0, image.width):
            rgb = pixels[column, row]
            hls = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            hls_array[row, column, 0] = hls[0]
            # hls_array[row, column, 1] = 100*(2**(2.5*(hls[1])))
            hls_array[row, column, 1] = hls[1]
            hls_array[row, column, 2] = hls[2]
    return hls_array


def image_from_hls_array(hls_array):
    new_image = Image.new("RGB", (hls_array.shape[1], hls_array.shape[0]))

    for row in range(0, new_image.height):
        for column in range(0, new_image.width):
            rgb = colorsys.hls_to_rgb(hls_array[row, column, 0],
                                      hls_array[row, column, 1],
                                      hls_array[row, column, 2])
            rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            new_image.putpixel((column, row), rgb)

    return new_image


################ Dat Gen - Max Flow generation ####################

def imageDrawLightChannelTopToBottom_HLS_new(input_img_file, nodes, values, filename, grid_count_horizontal,
                                             grid_count_vertical):
    # Create an empty image
    input_image = Image.open(input_img_file)
    output_image_size = input_image.size

    factor_x = int(output_image_size[0] / grid_count_horizontal)
    factor_y = int(output_image_size[1] / grid_count_vertical)

    img = Image.new('RGB', (grid_count_horizontal * factor_x, grid_count_vertical * factor_y), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    # Define a set of random color to color the faces
    colour = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
    # this is the factor to scale up the image

    max_val = np.max(values)
    min_val = np.min(values)
    # Draw all the faces

    for i in range(grid_count_horizontal):
        for j in range(grid_count_vertical, 0, -1):
            color_val = 0 + int((values[i][j - 1] - min_val) / (max_val - min_val) * 255.00)
            color_hex = "#{:02x}{:02x}{:02x}".format(color_val,
                                                     color_val,
                                                     color_val)

            d.polygon(
                [tuple(Point2D(nodes[i][j].loc.x * factor_x, (grid_count_vertical - nodes[i][j].loc.y) * factor_y))
                    , tuple(
                    Point2D(nodes[i][j - 1].loc.x * factor_x, (grid_count_vertical - nodes[i][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j - 1].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j - 1].loc.y) * factor_y))
                    , tuple(Point2D(nodes[i + 1][j].loc.x * factor_x,
                                    (grid_count_vertical - nodes[i + 1][j].loc.y) * factor_y))]
                , fill=color_hex)

    imagestring = 'output//' + filename + '_table_cartogram-' + str(grid_count_horizontal) + '_' \
                  + str(grid_count_vertical) + '.png'
    img.save(imagestring)

    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=10))

    img_blurred.save('output/lightness_blurred_image.png')
    image_contour = np.asarray(img_blurred)

    image_contour = np.flipud(image_contour)
    Z = image_contour[:, :, 0]

    plt.contourf(Z, levels=30, cmap='RdGy')

    plt.savefig('output/lightness_image.png')


    hls = create_hls_array(input_image)
    image_contour = np.flipud(image_contour)
    for i in range(input_image.size[0]):  # for every col:
        for j in range(input_image.size[1]):
            percentage = 0.25
            actual_val = hls[i, j][1]
            pred_weight_val = image_contour[i, j][0] / 255.00

            pred_score = actual_val + pred_weight_val * percentage
            if pred_score > 1.00:
                pred_score = 1.00

            hls[i, j][1] = pred_score

    output_path = 'output/' + filename + '.png'
    # out = out.convert("RGBA")
    # out.save(output_path, 'PNG')

    new_image = image_from_hls_array(hls)
    new_image.save(output_path)

    return output_path
