from energyMinimization import *
import colorsys
import csv
from collections import Counter
import itertools
import colorsys
import math
square_grid = 16

input_image_file = "input/room1.jpg"
output_weight_filename = "room_weight"



def min_max_scale(arr):
    arr[:, :, 1] = (arr[:, :, 1] - arr[:, :, 1].min()) / (arr[:, :, 1].max() - arr[:, :, 1].min())
    return arr
def create_hls_array(image):

    pixels = image.load()

    hls_array = np.empty(shape=(image.height, image.width, 3), dtype=float)

    for row in range(0, image.height):

        for column in range(0, image.width):

            rgb = pixels[column, row]
            hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hls_array[row, column, 0] = hls[0]
            hls_array[row, column, 1] = 100*(2**(2.5*(hls[1])))
            hls_array[row, column, 2] =hls[2]
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
hls=create_hls_array(input_image)
print('hi',hls[:,:,1].min())
print(hls[:,:,1].max())

hlsNew=min_max_scale(hls)

print('hi',hlsNew[:,:,1].min())
print(hlsNew[:,:,1].max())
new_image=image_from_hls_array(hlsNew)

new_image.save('hls_test3.png')


grid_count_horizontal = square_grid  
grid_count_vertical = square_grid
max=hls.shape[0]
print(hls.shape)
var=hls[:,:,1]
print(var.shape)

inc_y=int(max/grid_count_vertical)
inc_x=m.ceil(max/grid_count_horizontal)
output_txt_file=output_weight_filename+'_'+str(grid_count_horizontal)+'_'+str(grid_count_vertical)+'.txt'


data=[]
for x in range(0,grid_count_horizontal,1):
    for y in range(0,grid_count_vertical,1):

        sum=0
        n=0
        print('....'+str(x)+'.....'+str(y)+'.......')
        for j in range((y)*inc_y,(y+1)*inc_y,1):
            for i in range((x)*inc_x,(x+1)*inc_x,1):
                sum=var[i,j]+sum
                n=n+1
        av=sum/n
        print(av)
        data.append((av))

count = 1
with open(output_txt_file, 'w') as f:
    for i in data:
        if count != grid_count_horizontal * grid_count_vertical:
            f.write('{:.10f},'.format(i))
            #f.write('%s,'%i)
            if(count % grid_count_horizontal == 0):
                f.write('\n')
        else:
            f.write('{:.10f}'.format(i))
            #f.write('%s'%i)

        count += 1



