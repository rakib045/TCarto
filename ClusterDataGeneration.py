import matplotlib
import numpy as np
import heapq
import pandas as pd
matplotlib.use("agg")
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame


null_grid_weight = 0.8
centers = [(1, 1), (0.1, 0.1), (0.5,0.45)]
filepath = "Datasets/GeneratedData/"
# grid=[2,4,8,16,32,64,128,256,512,1024]
grid = [8]


#####################

X, label = make_blobs(n_samples=100, centers=centers, n_features=2,cluster_std=0.05,random_state=42,)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=label))
colors = {0:'red', 1:'blue', 2:'green',3:'pink'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
pyplot.savefig(filepath + "blob_point_position.png")


########################
#TCarto data generation
print("\nTCarto data generation-")


#df = pd.read_csv(r"D:\Git\PrescribedAreaDrawing-master\data generation\Raw Data\shape data\kflame.csv", header=None)

#x = df[df.columns[0]]
#y = df[df.columns[1]]
#cluster = 16

x = X[:,0]
y = X[:,1]

N = len(x)
#print(N)

for g in grid:
    grid_count_horizontal = g
    grid_count_vertical = g
    file_init_name = filepath + "TCarto_cluster_data_"
    output_txt_file = file_init_name + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".txt"
    inc_x = float(max(x) - min(x)) / float(grid_count_horizontal)
    inc_y = float(max(y) - min(y)) / float(grid_count_vertical)

    data = []
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)

    minimum_val = 10000000000

    for j in range(grid_count_vertical):
        for i in range(grid_count_horizontal):

            weight = 0
            # print('....'+str(i)+'.....'+str(j)+'.......')
            for xi, yi in zip(x, y):
                x_low_lim = min_x + (inc_x * (i))
                x_up_lim = min_x + inc_x * (i + 1)
                y_low_lim = max(y) - (inc_y * (j + 1))
                y_up_lim = max(y) - (inc_y * (j))
                if ((x_low_lim) <= xi and xi < (x_up_lim)) and ((y_low_lim) < yi and yi <= (y_up_lim)):
                    # print('xval....'+str(xi)+'.yval....'+str(yi)+'.......')
                    weight += 1 * g * g

            #if weight == 0:
            #    weight += null_grid_weight * g * g
            #    # weight+=(1.0/(2*(float(N))))*g*g

            w = (round(float(weight) / float(N), 10))
            if w != 0:
                if w < minimum_val:
                    minimum_val = w

            # print("final weight",w)
            data.append(w)


    for i in range(grid_count_horizontal * grid_count_vertical):
        if data[i] == 0:
            data[i] = minimum_val * null_grid_weight
    #data = data
    #print(data)

    print("Data is generating for " + str(grid_count_horizontal) + " by " + str(grid_count_vertical))
    count = 1
    with open(output_txt_file, 'w') as f:
        for i in data:

            if count != grid_count_horizontal * grid_count_vertical:
                f.write('{:.10f},'.format(i))
                # f.write('%s,'%i)

                if (count % grid_count_horizontal == 0):
                    f.write('\n')
            else:
                f.write('{:.10f}'.format(i))
                # f.write('%s'%i)

            count += 1


##############################
# Max Flow Data Generation

print("\nMax Flow data generation-")
#x = df[df.columns[0]]
#y = df[df.columns[1]]
#cluster = 6

#N = len(x)
#print(N)
#grid = [64]
# grid=[256,512]
for g in grid:
    grid_count_horizontal = g
    grid_count_vertical = g
    file_name_boundary = filepath + "MaxFlow_cluster_" + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".gen"
    file_name_weight = filepath + "MaxFlow_cluster_" + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".dat"
    # file_init_name = "Data_cluster_2/data_cluster_"+str(cluster)+"_"+str(p+1)+"_grid_"
    # output_txt_file = file_init_name + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".txt"

    inc_x = (max(x) - min(x)) / grid_count_horizontal
    inc_y = (max(y) - min(y)) / grid_count_vertical
    # print(inc_y)
    data = []

    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)

    minimum_val = 10000000000

    for j in range(grid_count_vertical):
        for i in range(grid_count_horizontal):

            weight = 0
            # print('....'+str(i)+'.....'+str(j)+'.......')
            for xi, yi in zip(x, y):
                x_low_lim = min_x + (inc_x * (i))
                x_up_lim = min_x + inc_x * (i + 1)
                y_low_lim = max(y) - (inc_y * (j + 1))
                y_up_lim = max(y) - (inc_y * (j))
                if ((x_low_lim) <= xi and xi < (x_up_lim)) and ((y_low_lim) < yi and yi <= (y_up_lim)):
                    # print('xval....'+str(xi)+'.yval....'+str(yi)+'.......')
                    weight += 1

            #if weight == 0:
            #    weight += null_grid_weight * g * g
            #    # weight+=(1.0/(2*(float(N))))*g*g


            w = (round(float(weight) / float(N), 10))
            if w != 0:
                if w < minimum_val:
                    minimum_val = w
            # print("final weight",w)
            data.append(w)

    for i in range(grid_count_horizontal * grid_count_vertical):
        if data[i] == 0:
            data[i] = minimum_val * null_grid_weight

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
