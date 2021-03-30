import matplotlib
import numpy as np
import heapq
import pandas as pd
matplotlib.use("agg")
#from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame


null_grid_weight = 0.8
centers = [(1, 1), (0.1, 0.1), (0.5, 0.45)]
filepath = "Datasets/GeneratedData/"
# grid=[2,4,8,16,32,64,128,256,512,1024]
grid = [16]
zero_zero_val = 1.0
zero_one_val = 10.0


data = []

for g in grid:
    data = []
    for i in range(g):
        for j in range(g):
            if (i+j)%2 == 0:
                data.append(zero_zero_val)
            else:
                data.append(zero_one_val)

    '''
    file_name_boundary = filepath + "MaxFlow_checker_" + str(g) + "_" + str(
        g) + ".gen"
    file_name_weight = filepath + "MaxFlow_checker_" + str(g) + "_" + str(
        g) + ".dat"
    result_array = np.reshape(data, (g, g))
    out_weight_file = open(file_name_weight, "w")
    out_boundary_file = open(file_name_boundary, "w")

    print("Max_Flow Data is generating for " + str(g) + " by " + str(g))

    counter = 0
    for i in range(len(result_array)):
        for j in range(len(result_array[0])):
            # print(str(result_array[i][j]) + ",")
            id_name = "A_" + str(i) + "_" + str(j)
            out_weight_file.write(str(counter) + " " + str(result_array[i][j]) + " " + id_name)

            out_boundary_file.write(str(counter) + " " + id_name + "\n")
            out_boundary_file.write(str(i) + " " + str(j * (-1)) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str(j * (-1)) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str((j + 1) * (-1)) + "\n")
            out_boundary_file.write(str(i) + " " + str((j + 1) * (-1)) + "\n")
            out_boundary_file.write("END\n")

            counter += 1
            if counter != g * g:
                out_weight_file.write(str(("\n")))

    out_boundary_file.write("END\n")
    out_weight_file.close()
    out_boundary_file.close()
    '''

    print("TCarto Data is generating for " + str(g) + " by " + str(g))

    file_init_name = filepath + "TCarto_checker_data_"
    output_txt_file = file_init_name + str(g) + "_" + str(g) + ".txt"

    count = 1
    with open(output_txt_file, 'w') as f:
        for i in data:

            if count != g * g:
                f.write('{:.10f},'.format(i))
                # f.write('%s,'%i)

                if (count % g == 0):
                    f.write('\n')
            else:
                f.write('{:.10f}'.format(i))
                # f.write('%s'%i)

            count += 1
