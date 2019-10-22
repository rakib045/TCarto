import shapefile
import numpy as np

grid_count = 8
file_path = "Datasets/GeneratedData/"
input_weighted_file_name = file_path + "TCarto_checker_data_8_8.txt"
output_file_name = file_path + "Carto4F_checker_data_8_8"


w = shapefile.Writer(shapefile.POLYGON)

#this is how to add a polygon
'''
w.poly([ [[0.,0.],[0.,1.],[1.,1.],[1.,0.]] ])
w.poly([ [[1.,0.],[1.,1.],[2.,1.],[2.,0.]] ])
'''

for i in range(grid_count):
    for j in range(grid_count):
        w.poly([ [[i, j], [i, j+1], [i+1, j+1], [i+1, j]] ])

'''
w.field('ID','C','40')
w.field('POP','F','12')

#this is show to add a value

w.record('01','44') # first polygon cell value 44
w.record('02','10') # 2nd polygon cell value 10
'''
w.field('ID','C','40')
w.field('POP','F','12')

sample_val = []
input_file = open(input_weighted_file_name, "r")
in_total_str = ''
in_str = input_file.readlines()
for i in range(len(in_str)):
    in_total_str += in_str[i].replace('\n', '').replace(' ', '')

val_str = in_total_str.split(",")
input_file.close()

index = 0
for i in range(grid_count):
    for j in range(grid_count):
        id = str(i) + "_" + str(j)
        pop = str(val_str[index])
        w.record(id, pop)
        index += 1

w.save(output_file_name)
