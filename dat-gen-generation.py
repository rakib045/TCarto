import numpy as np
import pandas as pd

grid_count_vertical = 8
grid_count_horizontal = 8
filepath = "Datasets/GeneratedData/"
file_name_boundary = filepath + "MaxFlow_cluster_" + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".gen"
file_name_weight = filepath + "MaxFlow_cluster_" + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".dat"

data = pd.read_csv(filepath + "TCarto_cluster_data_8_8.csv")

data = np.array(data)

result_array = np.reshape(data, (grid_count_vertical, grid_count_horizontal))
out_weight_file = open(file_name_weight, "w")
out_boundary_file = open(file_name_boundary, "w")

print("Data is generating for " + str(grid_count_horizontal) + " by " + str(grid_count_vertical))

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
        if counter != grid_count_horizontal * grid_count_vertical:
            out_weight_file.write(str(("\n")))

out_boundary_file.write("END\n")
out_weight_file.close()
out_boundary_file.close()