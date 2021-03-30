import matplotlib
import numpy as np
import heapq
import pandas as pd
matplotlib.use("agg")
#from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

grid = 16
file_path = "Datasets/GeneratedData/"
#input_file_name_list = ["ALBEDO_64_64.txt", "EMISS_64_64.txt", "PBLH_64_64.txt", "PSFC_64_64.txt",
#                        "Q2_64_64.txt", "SH2O_64_64.txt", "SMOIS_64_64.txt", "TSK_64_64.txt"]

input_file_name_list = ["TCarto_checker_data_16_16.txt"]

for input_file_name in input_file_name_list:
    output_file_name = input_file_name.split('.')[0]
    input_weighted_file_name = file_path + input_file_name

    data = []

    input_file = open(input_weighted_file_name, "r")
    in_total_str = ''
    in_str = input_file.readlines()
    for i in range(len(in_str)):
        in_total_str += in_str[i].replace('\n', '').replace(' ', '')

    data = in_total_str.split(",")
    input_file.close()

    sample_val = []
    for v in data:
        sample_val.append(float(v))

    values = np.zeros((grid, grid))
    sample_val_count = 0
    for j in range(grid - 1, -1, -1):
        for i in range(grid):
            values[i][j] = sample_val[sample_val_count]
            sample_val_count += 1

    file_name_boundary = file_path + output_file_name + ".gen"
    file_name_weight = file_path + output_file_name + ".dat"
    result_array = np.reshape(values, (grid, grid))
    out_weight_file = open(file_name_weight, "w")
    out_boundary_file = open(file_name_boundary, "w")

    print("Max_Flow Data is generating for " + str(grid) + " by " + str(grid))

    counter = 0

    for i in range(len(result_array)):
        for j in range(len(result_array[0])):
            # print(str(result_array[i][j]) + ",")
            id_name = "A_" + str(i) + "_" + str(j)
            out_weight_file.write(str(counter) + " " + str(result_array[i][j]) + " " + id_name)

            out_boundary_file.write(str(counter) + " " + id_name + "\n")
            out_boundary_file.write(str(i) + " " + str(j ) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str(j) + "\n")
            out_boundary_file.write(str(i + 1) + " " + str((j + 1)) + "\n")
            out_boundary_file.write(str(i) + " " + str((j + 1)) + "\n")
            out_boundary_file.write(str(i) + " " + str(j) + "\n")
            out_boundary_file.write("END\n")

            counter += 1
            if counter != grid * grid:
                out_weight_file.write(str(("\n")))

    out_boundary_file.write("END\n")
    out_weight_file.close()
    out_boundary_file.close()
