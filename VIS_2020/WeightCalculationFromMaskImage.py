from energyMinimization import *
import csv
from collections import Counter
import itertools

square_grid = 16
input_data_file = "input/weight_info_donut_apple.csv"
input_image_file = "input/donut_apple_mask.png"
output_weight_filename = "donut_apple_fat_weight_16_16"

color_column_name = "COLOR"
weight_column_name = "FAT"
label_column_name = "ITEM"

neutral_color = "#000000"
#neutral_color_weight = 30
min_weight_range = 1
max_weight_range = 5

#########################################################################

grid_count_horizontal = square_grid
grid_count_vertical = square_grid

colorOrderedList = []
weightOrderedList = []
nameOrderedList = []

maxColorListByGrid = []
splitted_image = []



print("Reading Image file ...")

input_image = Image.open(input_image_file)
input_image = input_image.convert("RGB")

pixel_count_per_grid = input_image.size[0] * input_image.size[1] / grid_count_horizontal / grid_count_vertical

print("Finished Reading Image file !!")

print("Reading Weight-CSV file ...")
with open(input_data_file, 'r') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        colorOrderedList.append(row[color_column_name].upper())
        weightOrderedList.append(float(row[weight_column_name]))
        nameOrderedList.append(row[label_column_name])
csvFile.close()
colorOrderedList.append(neutral_color)
neutral_color_weight = min(weightOrderedList) / 2
weightOrderedList.append(neutral_color_weight)
weightOrderedList = np.array(weightOrderedList)
print("Finished Weight-CSV file !!")

print("Splitting Input Image file into grids ...")
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

        #sub_image.save("input/image/input_" + str(i) + "_" + str(j) + ".png", "PNG")
        im.append(sub_image)
    splitted_image.append(im)

print("Finished Splitting Input Image file into grids !!")


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def scaleWeight(val):
    max_val = max(weightOrderedList)
    min_val = neutral_color_weight
    return min_weight_range + (val - min_val) / (max_val - min_val) * (max_weight_range - min_weight_range)

color_weight_per_grid = []
color_ordered_list_count_ratio = []
color_range = 10

print("Calculating Weights into grids ...")
for i in range(grid_count_horizontal):
    color_pixel_count_row = []
    for j in range(grid_count_vertical):
        occurence_count = Counter(splitted_image[i][j].getdata())
        pix_cols_rgb_count = occurence_count.most_common(10)

        color_pixel_count = []
        for color in colorOrderedList:
            color_weight = 0
            for col_count in pix_cols_rgb_count:
                color_rgb_tuple = hex_to_rgb(color)
                color_r = color_rgb_tuple[0]
                color_g = color_rgb_tuple[1]
                color_b = color_rgb_tuple[2]

                if (color_r-color_range) < col_count[0][0] <= (color_r+color_range) and \
                        (color_g - color_range) < col_count[0][1] <= (color_g + color_range) and \
                        (color_b - color_range) < col_count[0][2] <= (color_b + color_range):
                    color_weight = color_weight + col_count[1]

            color_pixel_count.append(color_weight)

        total_sum = sum(color_pixel_count)
        for index in range(len(color_pixel_count)):
            color_pixel_count[index] = color_pixel_count[index] / total_sum

        color_pixel_count_row.append(color_pixel_count)
    color_ordered_list_count_ratio.append(color_pixel_count_row)

total_ratio_per_color = np.zeros_like(weightOrderedList)
for array in color_ordered_list_count_ratio:
    for item_array in array:
        for index in range(len(item_array)):
            total_ratio_per_color[index] = total_ratio_per_color[index] + item_array[index]

unit_weight_per_color = weightOrderedList/total_ratio_per_color
calculated_weight_per_grid = np.zeros((grid_count_vertical, grid_count_horizontal))

for i in range(grid_count_horizontal):
    for j in range(grid_count_vertical):
        weight_grid = 0
        for col_index in range(len(colorOrderedList)):
            weight_grid = weight_grid + color_ordered_list_count_ratio[i][j][col_index] * unit_weight_per_color[col_index]

        calculated_weight_per_grid[i][j] = weight_grid

temp_ratio = (max_weight_range-min_weight_range) / (np.max(calculated_weight_per_grid) - np.min(calculated_weight_per_grid))
calculated_weight_per_grid = min_weight_range + ((calculated_weight_per_grid - np.min(calculated_weight_per_grid)) * temp_ratio)

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



