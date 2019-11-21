import matplotlib
import numpy as np
matplotlib.use("agg")
import shapefile

file_path = "Datasets/GeneratedData/Carto4F/"
file_name_shp = "EMISS_grid64_64_3.shp"
file_name_dbf = "EMISS_grid64_64_3.dbf"
file_name_shx = "EMISS_grid64_64_3.shx"

output_file_name = "EMISS_grid64_64_3"


myshp = open(file_path + file_name_shp, "rb")
mydbf = open(file_path + file_name_dbf, "rb")
myshx = open(file_path + file_name_shx, "rb")

r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
print(r)

file_name_boundary = file_path + output_file_name + ".gen"
file_name_weight = file_path + output_file_name + ".dat"
out_weight_file = open(file_name_weight, "w")
out_boundary_file = open(file_name_boundary, "w")

counter = 0
print("Writing DAT file ...")
for i in range(len(r.records())):
    if i != 0:
        out_weight_file.write("\n");
    out_weight_file.write(str(counter) + " " + str(r.records()[i].POP) + " A_" + r.records()[i].ID)
    print(counter)
    counter = counter + 1

out_weight_file.close()

print("Writing GEN file ...")
counter = 0
for i in range(len(r.shapes())):
    out_boundary_file.write(str(counter) + " A_" + r.records()[i].ID + "\n")
    for j in range(len(r.shapes()[i].points)):
        out_boundary_file.write(str(r.shapes()[i].points[j][0]) + " " + str(r.shapes()[i].points[j][1]) + "\n")

    out_boundary_file.write("END\n")
    print(counter)
    counter = counter + 1



out_boundary_file.write("END\n")
out_boundary_file.close()

print("Finished ...")

'''
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
'''