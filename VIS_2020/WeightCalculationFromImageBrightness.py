from energyMinimization import *
import colorsys
import csv
from collections import Counter
import itertools
import colorsys
import math

square_grid = 128
input_image_file = "input/LowLightImageEnhancement.png"
output_weight_filename = "input/LowLightImageEnhancement_lightness_weight_128_128.txt"



def create_hls_array(image):

    pixels = image.load()
    hls_array = np.empty(shape=(image.height, image.width, 3), dtype=float)

    for row in range(0, image.height):
        for column in range(0, image.width):
            rgb = pixels[column, row]
            hls = colorsys.rgb_to_hls(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            hls_array[row, column, 0] = hls[0]
            #hls_array[row, column, 1] = 100*(2**(2.5*(hls[1])))
            hls_array[row, column, 1] = hls[1]
            hls_array[row, column, 2] = hls[2]
    return hls_array

def image_from_hls_array(hls_array):

    new_image = Image.new("RGB", (hls_array.shape[1], hls_array.shape[0]))

    for row in range(0, new_image.height):
        for column in range(0, new_image.width):
            rgb = colorsys.hls_to_rgb(hls_array[row, column, 0],
                                      hls_array[row, column, 1],
                                      hls_array[row, column, 2])
            rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            new_image.putpixel((column, row), rgb)

    return new_image


input_image = Image.open(input_image_file)
hls = create_hls_array(input_image)
#print('Min:', hls[:, :, 1].min())
#print('Max:', hls[:, :, 1].max())

#new_image = image_from_hls_array(hls)
#new_image.save('input/hls_test3.png')

grid_count_horizontal = square_grid  
grid_count_vertical = square_grid
max = hls.shape[0]

var = hls[:, :, 1]

inc_y = int(max/grid_count_vertical)
inc_x = m.ceil(max/grid_count_horizontal)

data = []
for x in range(0, grid_count_horizontal, 1):
    for y in range(0, grid_count_vertical, 1):

        sum = 0
        n = 0
        #print('....'+str(x)+'.....'+str(y)+'.......')
        for j in range((y)*inc_y,(y+1)*inc_y,1):
            for i in range((x)*inc_x,(x+1)*inc_x,1):
                sum = var[i, j] + sum
                n = n+1
        av = sum/n
        #print(av)
        data.append(av)

#min_val = min(data)
#max_val = max(data)
#print('Min:', min_val)
#print('Max:', max_val)

count = 1
with open(output_weight_filename, 'w') as f:
    for i in data:
        if count != grid_count_horizontal * grid_count_vertical:
            f.write('{:.4f},'.format(i))
            if count % grid_count_horizontal == 0:
                f.write('\n')
        else:
            f.write('{:.4f}'.format(i))

        count += 1

print('Finished !!')

