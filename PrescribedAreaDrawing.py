#Best you use Anaconda. If you do not have quadprog, you have to
#install it.. in anaconda you can use this '-c omnia quadprog '
import numpy as np
import random
#import quadprog
from sympy.geometry import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
import math as m
from energyMinimization import *

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


#python PrescribedAreaDrawing.py 64 5 "input/PBLH_10_new_grid64_64.txt" "SingleThread_PBLH_10_new_grid64_64"
#python PrescribedAreaDrawing.py 64 5 "input/TCarto_checker_data_8_8.txt" "SingleThread_TCarto_checker_data_8_8"
# First param = python file name
# Second Param = square grid
# Third Param = Count of Iteration
# Forth Param = Input Data File
# Fifth Param = Output File Name
'''
square_grid = int(sys.argv[1])
iteration = int(sys.argv[2])
input_data_file = sys.argv[3]
output_img_filename = sys.argv[4]
'''
'''
square_grid = 16
#iteration = int(m.log(square_grid, 2))
iteration = 5
input_data_file = "input/data_cat_16_16.txt"
output_img_filename = "SingleThread_Gaussian_16_16"
'''

square_grid = 8
iteration = 10
input_data_file = "input/Aggregation_cluster_3_grid_8_8.txt"
output_img_filename = "TCarto_Aggregation_cluster_3_grid_8_8"

grid_count_horizontal = square_grid
grid_count_vertical = square_grid


output_image_size = [1024, 1024]
is_max_heiristic = True
boundary_node_movement = True
input_img_file = "input/cats.jpg"


total_algo_processing_time = []


######## Node Generation ########
nodes = grid_node_generation(node, grid_count_horizontal, grid_count_vertical)
######## Node Generation ########


#the following creates a matrix that contains the cell values
#the cell values are currently randomly assigned




##### Reading from Input File ###########
values = read_text_file(input_data_file, grid_count_horizontal, grid_count_vertical)


#nodes that should not move
nodes[0][0].movable = False
nodes[0][grid_count_vertical].movable = False
nodes[grid_count_horizontal][0].movable = False
nodes[grid_count_horizontal][grid_count_vertical].movable = False
#print(nodes)


#####################################################################################



#################################
# Algorithm: Prescribed_Area_Drawing
#################################
# This is just like Tutte's algorithm
#for each node update its location



print("Algorithm Started for " + str(grid_count_horizontal) + "_by_" + str(grid_count_vertical))

#poly_draw(output_img_filename,0, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)
min_boundary_p_dist = 0.1


out_file_name = "output/out_log_" + output_img_filename + ".txt"
output_txt_file = open(out_file_name, "w")
output_txt_file.write("Iteration, |UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N, Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, Iteration Time (sec)\n")
output_txt_file.close()


for x in range(iteration):
    print("------------------------------------------")
    print('iteration: ' + str(x+1) + '(out of ' + str(iteration) + '): ')

    updated = 1
    iteration_start_time = time()

    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical + 1):
            #if the node is in the skip nodelist then do not move those nodes

            if(nodes[i][j].movable == False):
                continue
            elif(i == 0):

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

                V1 = values[i][j-1]
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
                    nodes[i][j].loc = val

                continue
            elif(i == grid_count_horizontal):
                #TODO: Replace 'continue' with your own code

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

                V1 = values[i-1][j-1]
                V2 = values[i-1][j]

                # p_right is the corresponding inside node and line is x=0
                changed_y = updateBoundaryNode(p_left.loc, A1, A2, V1, V2, 1, 0, -grid_count_horizontal)
                updated_y = nodes[i][j].loc[1] + changed_y

                if updated_y < p_bottom.loc[1] + min_boundary_p_dist:
                    updated_y = p_bottom.loc[1] + min_boundary_p_dist

                if updated_y > p_top.loc[1] - min_boundary_p_dist:
                    updated_y = p_top.loc[1] - min_boundary_p_dist

                BL_V_Line = Line(p_left.loc, p_bottom_left.loc)
                TL_V_Line = Line(p_top_left.loc, p_left.loc)

                val = Point2D(nodes[i][j].loc[0], updated_y)

                checkSignVal_1 = isSatisfyInEquility(val, BL_V_Line)
                checkSignVal_2 = isSatisfyInEquility(val, TL_V_Line)


                if checkSignVal_1 <= 0 and checkSignVal_2 <= 0 and boundary_node_movement:
                    nodes[i][j].loc = val
                continue
            elif(j == 0):
                #TODO: Replace 'continue' with your own code

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

                V1 = values[i-1][j]
                V2 = values[i][j]

                changed_x = updateBoundaryNode(p_top.loc, A1, A2, V1, V2, 0, 1, 0)
                updated_x = nodes[i][j].loc[0] + changed_x

                if updated_x < p_left.loc[0] + min_boundary_p_dist:
                    updated_x = p_left.loc[0] + min_boundary_p_dist

                if updated_x > p_right.loc[0] - min_boundary_p_dist:
                    updated_x = p_right.loc[0] - min_boundary_p_dist

                TL_H_Line = Line(p_top_left.loc, p_top.loc)
                TR_H_Line = Line(p_top.loc, p_top_right.loc)

                val = Point2D(updated_x, nodes[i][j].loc[1])

                checkSignVal_1 = isSatisfyInEquility(val, TL_H_Line)
                checkSignVal_2 = isSatisfyInEquility(val, TR_H_Line)

                if checkSignVal_1 >= 0 and checkSignVal_2 >= 0 and boundary_node_movement:
                    nodes[i][j].loc = val
                continue
            elif(j == grid_count_vertical):
                #TODO: Replace 'continue' with your own code

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

                V1 = values[i-1][j-1]
                V2 = values[i][j-1]

                changed_x = updateBoundaryNode(p_bottom.loc, A1, A2, V1, V2, 0, 1, -grid_count_vertical)
                updated_x = nodes[i][j].loc[0] + changed_x

                if updated_x < p_left.loc[0] + min_boundary_p_dist:
                    updated_x = p_left.loc[0] + min_boundary_p_dist

                if updated_x > p_right.loc[0] - min_boundary_p_dist:
                    updated_x = p_right.loc[0] - min_boundary_p_dist

                BL_H_Line = Line(p_bottom_left.loc, p_bottom.loc)
                BR_H_Line = Line(p_bottom.loc, p_bottom_right.loc)

                val = Point2D(updated_x, nodes[i][j].loc[1])

                checkSignVal_1 = isSatisfyInEquility(val, BL_H_Line)
                checkSignVal_2 = isSatisfyInEquility(val, BR_H_Line)


                if checkSignVal_1 <= 0 and checkSignVal_2 <= 0 and boundary_node_movement:
                    nodes[i][j].loc = val

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

                if x == 34 or x == 33:
                    temp = 10

                val = updateNode([p_top_left, p_left, p_bottom_left,
                                  p_top, p_middle, p_bottom,
                                  p_top_right, p_right, p_bottom_right],
                                 [val_TL, val_BL, val_TR, val_BR])

                if (val[0] == -1 and val[1] == -1):
                    continue

                nodes[i][j].loc = val

            #poly_draw(output_img_filename, x+1, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)
            #print("Updated val : " + str(val.x) + ", " + str(val.y))

    iteration_end_time = time()
    estimation_time = iteration_end_time - iteration_start_time

    total_algo_processing_time.append(estimation_time)
    poly_draw(output_img_filename, x+1, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)

    all_error_calc(values, nodes, grid_count_horizontal, grid_count_vertical, estimation_time, output_img_filename,
                   x+1, -1, -1)



print("------------------------------------------")
print("Total Algorithm Processing Time (sec): " + str(round(np.sum(total_algo_processing_time), 4)))

output_txt_file = open(out_file_name, "a")
output_txt_file.write("\n\nTotal Pre Processing Time(sec): 0\n")
output_txt_file.write("Total Processing Time(sec): " + str(round(np.sum(total_algo_processing_time), 4)))
output_txt_file.close()

print("------------------------------------------")
print("Algorithm Finished !! ")

print("Drawing Image ... ")

poly_draw(output_img_filename, iteration, output_image_size, nodes, grid_count_horizontal, grid_count_vertical)
#imageDraw(input_image.size, splitted_image, nodes, "output", grid_count_horizontal, grid_count_vertical)
print("Finished")


