#Best you use Anaconda. If you do not have quadprog, you have to
#install it.. in anaconda you can use this '-c omnia quadprog '
import numpy as np
import random
#import quadprog
import multiprocessing as mp
from sympy.geometry import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
import math as m
from energyMinimization import *
from time import sleep
from threading import Thread
from skimage.measure import *

######################################
# Node class; each node has a name and location
######################################
#build the node class
class node:
    movable = True
    def __init__(self, name, loc):
        self.name = name
        self.loc = loc


######################################
# INPUT: A table with randomly filled positive numbers
#        The sum of the numbers is (totalnode-1)*(totalnode-1)
#        There is also a list of nodes that must not move
######################################

# the variable totalnode represents the size of the table
# right now it is a (totalnode-1)*(totalnode-1) table
# the following loop creates a bunch of nodes, each at point i,j
# the name of a node at point i,j is ij

#python PrescribedAreaDrawingParallelCodeIMG.py 64 10 "input/PBLH_grid64_64.txt" "input/weather_tsk.png" "Parallel_OPT_64_64_PBLH_"

# First param = python file name
# Second Param = square grid
# Third Param = Count of Iteration
# Forth Param = Input Data File
# Fifth Param = Input Image File
# Sixth Param = Output File Name

square_grid = int(sys.argv[1])
iteration = int(sys.argv[2])
input_data_file = sys.argv[3]
input_img_file = sys.argv[4]
output_img_filename = sys.argv[5]



#square_grid = 8
#iteration = 10
#input_data_file = 'input/weight_8_8.txt'
#output_img_filename = 'output_ParallelCode'
#input_img_file = "input/weather_tsk.png"


grid_count_horizontal = square_grid
grid_count_vertical = square_grid
cpu_count = mp.cpu_count()
#cpu_count = 2

#output_image_size = [1024, 1024]
is_max_heiristic = True
boundary_node_movement = True


total_algo_processing_time = []
min_boundary_p_dist = 0.1


#################################################
def updatedNodeParallelCode(points, nodes, values, thread_count, grid_count_horizontal, grid_count_vertical):
    points_array = []
    for p in range(len(points)):
        # if the node is in the skip nodelist then do not move those nodes
        i = points[p][0]
        j = points[p][1]
        if (nodes[i][j].movable == False):
            temp = 1
            # this is nothing
        elif (i == 0):

            p_top = nodes[i][j + 1]
            p_middle = nodes[i][j]
            p_bottom = nodes[i][j - 1]

            p_top_right = nodes[i + 1][j + 1]
            p_right = nodes[i + 1][j]
            p_bottom_right = nodes[i + 1][j - 1]

            poly1 = Polygon(p_middle.loc, p_bottom.loc, p_bottom_right.loc, p_right.loc)
            A1 = poly1.area

            poly2 = Polygon(p_middle.loc, p_right.loc, p_top_right.loc, p_top.loc)
            A2 = poly2.area

            V1 = values[i][j - 1]
            V2 = values[i][j]

            # p_right is the corresponding inside node and line is x=0
            changed_y = updateBoundaryNode(p_right.loc, A1, A2, V1, V2, 1, 0, 0)
            updated_y = nodes[i][j].loc[1] + changed_y

            if updated_y < p_bottom.loc[1] + min_boundary_p_dist:
                updated_y = p_bottom.loc[1] + min_boundary_p_dist

            if updated_y > p_top.loc[1] - min_boundary_p_dist:
                updated_y = p_top.loc[1] - min_boundary_p_dist

            BR_V_Line = Line(p_bottom_right.loc, p_right.loc)
            TR_V_Line = Line(p_right.loc, p_top_right.loc)

            val = Point2D(nodes[i][j].loc[0], updated_y)

            checkSignVal_1 = isSatisfyInEquility(val, BR_V_Line)
            checkSignVal_2 = isSatisfyInEquility(val, TR_V_Line)

            if checkSignVal_1 >= 0 and checkSignVal_2 >= 0 and boundary_node_movement:
                poly_1 = Polygon(val, p_bottom.loc, p_bottom_right.loc, p_right.loc)
                poly_2 = Polygon(val, p_right.loc, p_top_right.loc, p_top.loc)

                if poly_1.is_convex() and poly_2.is_convex():
                    nodes[i][j].loc = val
                    points_array.append([i, j, nodes[i][j].loc[0], updated_y])
                continue

        elif (i == grid_count_horizontal):
            # TODO: Replace 'continue' with your own code

            p_top = nodes[i][j + 1]
            p_middle = nodes[i][j]
            p_bottom = nodes[i][j - 1]

            p_top_left = nodes[i - 1][j + 1]
            p_left = nodes[i - 1][j]
            p_bottom_left = nodes[i - 1][j - 1]

            poly1 = Polygon(p_middle.loc, p_left.loc, p_bottom_left.loc, p_bottom.loc)
            A1 = poly1.area

            poly2 = Polygon(p_middle.loc, p_top.loc, p_top_left.loc, p_left.loc)
            A2 = poly2.area

            V1 = values[i - 1][j - 1]
            V2 = values[i - 1][j]

            # p_right is the corresponding inside node and line is x=0
            changed_y = updateBoundaryNode(p_left.loc, A1, A2, V1, V2, 1, 0, -grid_count_horizontal)
            updated_y = nodes[i][j].loc[1] + changed_y

            if updated_y < p_bottom.loc[1] + min_boundary_p_dist:
                updated_y = p_bottom.loc[1] + min_boundary_p_dist

            if updated_y > p_top.loc[1] - min_boundary_p_dist:
                updated_y = p_top.loc[1] - min_boundary_p_dist

            BL_V_Line = Line(p_left.loc, p_bottom_left.loc)
            TL_V_Line = Line(p_top_left.loc, p_left.loc)

            val = Point(nodes[i][j].loc[0], updated_y)

            checkSignVal_1 = isSatisfyInEquility(val, BL_V_Line)
            checkSignVal_2 = isSatisfyInEquility(val, TL_V_Line)

            if checkSignVal_1 <= 0 and checkSignVal_2 <= 0 and boundary_node_movement:
                poly_1 = Polygon(val, p_left.loc, p_bottom_left.loc, p_bottom.loc)
                poly_2 = Polygon(val, p_top.loc, p_top_left.loc, p_left.loc)

                if poly_1.is_convex() and poly_2.is_convex():
                    nodes[i][j].loc = val
                    points_array.append([i, j, nodes[i][j].loc[0], updated_y])
                continue

        elif (j == 0):
            # TODO: Replace 'continue' with your own code

            p_top_left = nodes[i - 1][j + 1]
            p_left = nodes[i - 1][j]

            p_top = nodes[i][j + 1]
            p_middle = nodes[i][j]

            p_top_right = nodes[i + 1][j + 1]
            p_right = nodes[i + 1][j]

            poly1 = Polygon(p_middle.loc, p_top.loc, p_top_left.loc, p_left.loc)
            A1 = poly1.area

            poly2 = Polygon(p_middle.loc, p_right.loc, p_top_right.loc, p_top.loc)
            A2 = poly2.area

            V1 = values[i - 1][j]
            V2 = values[i][j]

            changed_x = updateBoundaryNode(p_top.loc, A1, A2, V1, V2, 0, 1, 0)
            updated_x = nodes[i][j].loc[0] + changed_x

            if updated_x < p_left.loc[0] + min_boundary_p_dist:
                updated_x = p_left.loc[0] + min_boundary_p_dist

            if updated_x > p_right.loc[0] - min_boundary_p_dist:
                updated_x = p_right.loc[0] - min_boundary_p_dist

            TL_H_Line = Line(p_top_left.loc, p_top.loc)
            TR_H_Line = Line(p_top.loc, p_top_right.loc)

            val = Point(updated_x, nodes[i][j].loc[1])

            checkSignVal_1 = isSatisfyInEquility(val, TL_H_Line)
            checkSignVal_2 = isSatisfyInEquility(val, TR_H_Line)

            if checkSignVal_1 >= 0 and checkSignVal_2 >= 0 and boundary_node_movement:
                poly_1 = Polygon(val, p_top.loc, p_top_left.loc, p_left.loc)
                poly_2 = Polygon(val, p_right.loc, p_top_right.loc, p_top.loc)

                if poly_1.is_convex() and poly_2.is_convex():
                    nodes[i][j].loc = val
                    points_array.append([i, j, updated_x, nodes[i][j].loc[1]])
                continue

        elif (j == grid_count_vertical):
            # TODO: Replace 'continue' with your own code

            p_left = nodes[i - 1][j]
            p_bottom_left = nodes[i - 1][j - 1]

            p_middle = nodes[i][j]
            p_bottom = nodes[i][j - 1]

            p_right = nodes[i + 1][j]
            p_bottom_right = nodes[i + 1][j - 1]

            poly1 = Polygon(p_middle.loc, p_left.loc, p_bottom_left.loc, p_bottom.loc)
            A1 = poly1.area

            poly2 = Polygon(p_middle.loc, p_bottom.loc, p_bottom_right.loc, p_right.loc)
            A2 = poly2.area

            V1 = values[i - 1][j - 1]
            V2 = values[i][j - 1]

            changed_x = updateBoundaryNode(p_bottom.loc, A1, A2, V1, V2, 0, 1, -grid_count_vertical)
            updated_x = nodes[i][j].loc[0] + changed_x

            if updated_x < p_left.loc[0] + min_boundary_p_dist:
                updated_x = p_left.loc[0] + min_boundary_p_dist

            if updated_x > p_right.loc[0] - min_boundary_p_dist:
                updated_x = p_right.loc[0] - min_boundary_p_dist

            BL_H_Line = Line(p_bottom_left.loc, p_bottom.loc)
            BR_H_Line = Line(p_bottom.loc, p_bottom_right.loc)

            val = Point(updated_x, nodes[i][j].loc[1])

            checkSignVal_1 = isSatisfyInEquility(val, BL_H_Line)
            checkSignVal_2 = isSatisfyInEquility(val, BR_H_Line)

            if checkSignVal_1 <= 0 and checkSignVal_2 <= 0 and boundary_node_movement:
                poly_1 = Polygon(val, p_left.loc, p_bottom_left.loc, p_bottom.loc)
                poly_2 = Polygon(val, p_bottom.loc, p_bottom_right.loc, p_right.loc)

                if poly_1.is_convex() and poly_2.is_convex():
                    nodes[i][j].loc = val
                    points_array.append([i, j, updated_x, nodes[i][j].loc[1]])
                continue

        else:

            p_top_left = nodes[i - 1][j + 1]
            p_left = nodes[i - 1][j]
            p_bottom_left = nodes[i - 1][j - 1]

            p_top = nodes[i][j + 1]
            p_middle = nodes[i][j]
            p_bottom = nodes[i][j - 1]

            p_top_right = nodes[i + 1][j + 1]
            p_right = nodes[i + 1][j]
            p_bottom_right = nodes[i + 1][j - 1]

            val_TL = values[i - 1][j]
            val_BL = values[i - 1][j - 1]
            val_TR = values[i][j]
            val_BR = values[i][j - 1]

            val = updateNode([p_top_left, p_left, p_bottom_left,
                              p_top, p_middle, p_bottom,
                              p_top_right, p_right, p_bottom_right],
                             [val_TL, val_BL, val_TR, val_BR])

            if (val[0] == -1 and val[1] == -1):
                temp = 1
                # this is nothing
            else:
                poly_1 = Polygon(val, p_top.loc, p_top_left.loc, p_left.loc)
                poly_2 = Polygon(val, p_left.loc, p_bottom_left.loc, p_bottom.loc)
                poly_3 = Polygon(val, p_bottom.loc, p_bottom_right.loc, p_right.loc)
                poly_4 = Polygon(val, p_right.loc, p_top_right.loc, p_top.loc)

                if poly_1.is_convex() and poly_2.is_convex() and poly_3.is_convex() and poly_4.is_convex():
                    nodes[i][j].loc = val
                    points_array.append([i, j, val[0], val[1]])
                continue

        points_array.append([i, j, -1, -1])

    print("Thread : " + str(thread_count))
    return points_array


def pointDPT_based_on_horizontal_points(h_grid_assigned, cpu_unassigned, values, max_heuristic, grid_count_horizontal, grid_count_vertical):

    final_array_first =[]
    final_array_second = []
    grid_index = 0
    boundary_count = cpu_unassigned - 1
    total_h_grid = h_grid_assigned

    for th in range(cpu_unassigned):
        p_array_grid = []
        p_array_boundary = []
        grid_for_th = int(m.ceil((h_grid_assigned-boundary_count) / cpu_unassigned))
        boundary_index = grid_index + grid_for_th

        for i in range(grid_index, boundary_index):
            for j in range(grid_count_vertical + 1):
                # print("Thread " + str(th) + ":" + str(i) + "," + str(j))

                # Top Left
                val1 = 0 if (i == 0 or j == 0) else values[i-1][j-1]
                # Top Right
                val2 = 0 if (i == grid_count_horizontal or j == 0) else values[i][j-1]
                # Bottom left
                val3 = 0 if (i == 0 or j == grid_count_vertical) else values[i-1][j]
                # Bottom Right
                val4 = 0 if (i == grid_count_horizontal or j == grid_count_vertical) else values[i][j]
                max_neighbourhood_val = max(val1, val2, val3, val4)
                p_array_grid.append([i, j, max_neighbourhood_val])
        if max_heuristic:
            p_array_grid = sorted(p_array_grid, key=lambda x: x[2], reverse=True)
        final_array_first.append(p_array_grid)

        if boundary_index < total_h_grid:
            for k in range(grid_count_vertical + 1):
                # print("Thread " + str(th) + ":" + str(i) + "," + str(j))
                # Top Left
                val1 = 0 if (boundary_index == 0 or k == 0) else values[boundary_index - 1][k - 1]
                # Top Right
                val2 = 0 if (boundary_index == grid_count_horizontal or k == 0) else values[boundary_index][k - 1]
                # Bottom left
                val3 = 0 if (boundary_index == 0 or k == grid_count_vertical) else values[boundary_index - 1][k]
                # Bottom Right
                val4 = 0 if (boundary_index == grid_count_horizontal or k == grid_count_vertical) else values[boundary_index][k]
                max_neighbourhood_val = max(val1, val2, val3, val4)

                p_array_boundary.append([boundary_index, k, max_neighbourhood_val])
            if max_heuristic:
                p_array_boundary = sorted(p_array_boundary, key=lambda x: x[2], reverse=True)
            final_array_second.append(p_array_boundary)

        h_grid_assigned = h_grid_assigned - grid_for_th - 1
        grid_index = boundary_index + 1
        boundary_count -= 1
        cpu_unassigned -= 1

    return final_array_first, final_array_second

#################################
# Algorithm: Prescribed_Area_Drawing
#################################
# This is just like Tutte's algorithm
#for each node update its location

if __name__ == "__main__":

    #prev_rmse = 0

    print("Number of available Processors: " + str(mp.cpu_count()))
    print("Number of used Processors: " + str(cpu_count))

    print("Max Heuristic : " + str(is_max_heiristic))

    ######## Node Generation ########

    nodes = grid_node_generation(node, grid_count_horizontal, grid_count_vertical)
    ######## Node Generation ########

    ##### Reading from Input File ###########
    values_actual = read_text_file(input_data_file, grid_count_horizontal, grid_count_vertical)

    values = values_actual / np.sum(values_actual)

    # all values sum to totalarea
    values = values * grid_count_horizontal * grid_count_vertical

    # nodes that should not move
    nodes[0][0].movable = False
    nodes[0][grid_count_vertical].movable = False
    nodes[grid_count_horizontal][0].movable = False
    nodes[grid_count_horizontal][grid_count_vertical].movable = False
    # print(nodes)


    print("Algorithm Started for " + str(grid_count_horizontal) + "_by_" + str(grid_count_vertical))
    #poly_draw(output_img_filename, 0, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)

    out_file_name = "output/out_log_" + output_img_filename + ".txt"
    output_txt_file = open(out_file_name, "w")
    output_txt_file.write("Iteration, |UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N,"
                          " Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, Average Aspect Ratio (height/width),"
                          "Average Aspect Ratio min(height/width)/max(height/width), "
                          "Iteration Time(sec)\n")
    output_txt_file.close()

    input_image = Image.open(input_img_file)
    input_image = input_image.convert("RGBA")
    output_image_size = input_image.size

    preprocessing_start_time = time()

    if ((2*cpu_count) - 1) > (grid_count_horizontal + 1):
        cpu_count = int(m.ceil((grid_count_horizontal + 1)/2))

    point_to_be_changed_array, point_to_between_thread = pointDPT_based_on_horizontal_points(grid_count_horizontal + 1,
                                                                                             cpu_count, values,
                                                                                             is_max_heiristic,
                                                                                             grid_count_horizontal,
                                                                                             grid_count_vertical)
    preprocessing_end_time = time()
    preprocessing_time = preprocessing_end_time - preprocessing_start_time

    print("Pre processing time(sec): " + str(round(preprocessing_time, 4)))

    for x in range(iteration):
        print("------------------------------------------")
        print('iteration: ' + str(x+1) + '(out of ' + str(iteration) + '): ')

        iteration_start_time = time()

        pool = mp.Pool(cpu_count)
        thread_result = []
        
        for th in range(cpu_count):
            thread_result.append(
                pool.apply_async(updatedNodeParallelCode,
                                 args=(point_to_be_changed_array[th], nodes, values, th,
                                       grid_count_horizontal, grid_count_vertical,)))
        print("Parallel Code is running")
        pool.close()
        pool.join()
        print("Parallel Code has finished")

        for th in range(cpu_count):
            val_array = thread_result[th]._value
            for count in range(len(val_array)):
                updated_x = val_array[count][2]
                updated_y = val_array[count][3]
                if (updated_x == -1 or updated_y == -1):
                    temp = 1
                    # This is nothing
                else:
                    nodes[val_array[count][0]][val_array[count][1]].loc = Point2D(updated_x, updated_y)
                #print(val_array[count])


        pool = mp.Pool(cpu_count)
        thread_result = []

        for th in range(cpu_count-1):
            thread_result.append(
                pool.apply_async(updatedNodeParallelCode,
                                 args=(point_to_between_thread[th], nodes, values, th,
                                       grid_count_horizontal, grid_count_vertical,)))

        print("Parallel Code(inner boundary) is running")
        pool.close()
        pool.join()
        print("Parallel Code(inner boundary) has finished")

        for th in range(cpu_count-1):
            val_array = thread_result[th]._value
            for count in range(len(val_array)):
                updated_x = val_array[count][2]
                updated_y = val_array[count][3]
                if (updated_x == -1 or updated_y == -1):
                    temp = 1
                    # This is nothing
                else:
                    nodes[val_array[count][0]][val_array[count][1]].loc = Point2D(updated_x, updated_y)

        iteration_end_time = time()
        estimation_time = iteration_end_time - iteration_start_time

        total_algo_processing_time.append(estimation_time)

        #poly_draw(output_img_filename, x+1, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)
        current_rmse = all_error_calc(values, nodes, grid_count_horizontal, grid_count_vertical, estimation_time,
                                      output_img_filename, x+1, -1, -1)

        if current_rmse[0] < 0.005:
            break



    print("------------------------------------------")
    total_time = round(np.sum(total_algo_processing_time) + preprocessing_time, 4)
    print("Total Algorithm Processing Time (sec): " + str(total_time))

    output_txt_file = open(out_file_name, "a")
    output_txt_file.write("\n\nTotal Pre Processing Time(sec): " + str(round(preprocessing_time, 4)) + "\n")
    output_txt_file.write("Total Processing Time(sec): " + str(total_time))
    output_txt_file.close()

    print("------------------------------------------")
    print("Algorithm Finished !! ")
    print("Drawing Image ... ")

    poly_draw(output_img_filename, iteration, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)
    #imageDraw(input_image.size, splitted_image, nodes, "output", grid_count_horizontal, grid_count_vertical)
    output_image_path = newImageDraw(input_image, nodes, output_img_filename,
                                grid_count_horizontal, grid_count_vertical)

    output_image = Image.open(output_image_path)
    out_image = np.asarray(output_image.convert("RGBA"))
    in_image = np.asarray(input_image)

    mse_error = compare_mse(in_image, out_image)
    psnr_error = compare_psnr(in_image, out_image)
    ssim_error = compare_ssim(in_image, out_image, multichannel=True)

    output_txt_file = open(out_file_name, "a")
    output_txt_file.write("\n\nMSE : " + str(round(mse_error, 4)) + "\n")
    output_txt_file.write("PSNR : " + str(round(psnr_error, 4)) + "\n")
    output_txt_file.write("SSIM : " + str(round(ssim_error, 4)) + "\n")

    print("Finished")

    print("Finished")


