from energyMinimization import *
import csv
from collections import Counter
import itertools

square_grid = 16
output_weight_filename = "Mosaic02_mosaic_randomweight_16_16"


min_weight_range = 1
max_weight_range = 1.6

#########################################################################

grid_count_horizontal = square_grid
grid_count_vertical = square_grid

multiplier = 100.00

calculated_weight_per_grid = np.random.randint(min_weight_range * multiplier, max_weight_range * multiplier,
                                                       size=(grid_count_vertical, grid_count_horizontal))

calculated_weight_per_grid = calculated_weight_per_grid/multiplier
print("Finished Calculating Weights into grids!!")

print("Writing Weights into output files ...")
calculated_weight_per_grid = calculated_weight_per_grid.transpose()
out_file_name = "input/" + output_weight_filename + ".txt"
output_txt_file = open(out_file_name, "w")

count = 1
for i in range(grid_count_horizontal):
    for j in range(grid_count_vertical):
        output_txt_file.write(str(format(calculated_weight_per_grid[i][j], '.4f')))
        if count != grid_count_horizontal * grid_count_vertical:
            output_txt_file.write(", ")
            if (count % grid_count_horizontal == 0):
                output_txt_file.write("\n")
        count += 1

output_txt_file.close()

print("Finished Writing Weights into output files!!")



