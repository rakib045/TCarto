import shapefile
import numpy as np

grid_count = 64
file_path = "Datasets/GeneratedData/"
input_weighted_file_name = file_path + "PBLH_grid64_64.txt"
output_file_name = file_path + "PBLH_grid64_64"


w = shapefile.Writer(output_file_name, shapefile.POLYGON)

#this is how to add a polygon
'''
w.poly([ [[0.,0.],[0.,1.],[1.,1.],[1.,0.]] ])
w.poly([ [[1.,0.],[1.,1.],[2.,1.],[2.,0.]] ])
'''

for i in range(grid_count):
    for j in range(grid_count):
        w.poly([[[i, j], [i, j + 1], [i + 1, j + 1], [i + 1, j], [i, j]]])
        #w.poly([ [[i, j],  [i+1, j], [i+1, j+1], [i, j+1], [i, j] ] ])

'''
w.field('ID','C','40')
w.field('POP','F','12')

#this is show to add a value

w.record('01','44') # first polygon cell value 44
w.record('02','10') # 2nd polygon cell value 10
'''
w.field('ID','C', size=40)
w.field('POP','F', decimal=10)

sample_val = []
input_file = open(input_weighted_file_name, "r")
in_total_str = ''
in_str = input_file.readlines()
for i in range(len(in_str)):
    in_total_str += in_str[i].replace('\n', '').replace(' ', '')

val_str = in_total_str.split(",")
input_file.close()

for v in val_str:
    sample_val.append(float(v))

values = np.zeros((grid_count, grid_count))
sample_val_count = 0
for j in range(grid_count - 1, -1, -1):
    for i in range(grid_count):
        values[i][j] = sample_val[sample_val_count]
        sample_val_count += 1

index = 0
for i in range(grid_count):
    for j in range(grid_count):
        id = str(i) + "_" + str(j)
        #pop = str(val_str[index])
        pop = str(values[i][j])
        w.record(id, pop)
        index += 1

#w.save(output_file_name)
