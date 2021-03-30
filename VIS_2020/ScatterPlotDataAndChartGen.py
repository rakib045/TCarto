import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

filename = 'cluster_4_data_points'
grid_count = 64
output_txt_file = "input/cluster_4_grid_64_64.txt"
percentage_weight_for_empty_point = 0.75

min_x_axis = 0
max_x_axis = 1024
min_y_axis = 0
max_y_axis = 1024

scale = 100
centers1 = [(2.5, 2.5)]
centers2 = [(7.5, 7.5)]
centers3 = [(2.5, 7.5)]
centers4 = [(4, 6)]

print('Data is generating ...')
X1, y1 = make_blobs(n_samples=500, centers=centers1, n_features=2, cluster_std=0.3, random_state=42,)
X2, y2 = make_blobs(n_samples=200, centers=centers2, n_features=2, cluster_std=0.6, random_state=42,)
X3, y3 = make_blobs(n_samples=120, centers=centers3, n_features=2, cluster_std=0.55, random_state=42,)
X4, y4 = make_blobs(n_samples=120, centers=centers4, n_features=2, cluster_std=0.55, random_state=42,)

X1 = X1 * scale
X2 = X2 * scale
X3 = X3 * scale
X4 = X4 * scale

X = np.concatenate((X1, X2, X3, X4))
#print(X[:])

#fig, ax = plt.subplots()
fig = plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(np.arange(min_x_axis, max_x_axis, scale))
plt.yticks(np.arange(min_y_axis, max_y_axis, scale))
plt.show()
fig.savefig("input/" + filename + "_image.png")

#fig1, ax = plt.subplots()
fig1 = plt.figure(figsize=(5, 5))
plt.scatter(X1[:, 0], X1[:, 1], label='500 Samples')
plt.scatter(X2[:, 0], X2[:, 1], label='200 Samples')
plt.scatter(X3[:, 0], X3[:, 1], label='120 Samples')
plt.scatter(X4[:, 0], X4[:, 1], label='120 Samples')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(np.arange(min_x_axis, max_x_axis, scale))
plt.yticks(np.arange(min_y_axis, max_y_axis, scale))
plt.legend()
plt.show()
fig1.savefig("input/" + filename + "_color_image.png")


csv_filename = "input/" + filename + ".csv"
with open(csv_filename, 'w') as f:
    for i in range(len(X[:])):
        f.write("{:.4f},".format(X[:, 0][i]) + "{:.4f}\n".format(X[:, 1][i]))

print('Data generation complete !!')

df = pd.read_csv(csv_filename, header=None)

x = df[df.columns[0]]
y = df[df.columns[1]]

grid_count_horizontal = grid_count
grid_count_vertical = grid_count



data = []
min_x = min_x_axis
min_y = min_y_axis
max_x = max_x_axis
max_y = max_y_axis

inc_x = float(max_x-min_x)/float(grid_count_horizontal)
inc_y = float(max_y-min_y)/float(grid_count_vertical)

N = len(x)

for j in range(grid_count_vertical):
    for i in range(grid_count_horizontal):
        weight = 0
        #print('(i, j)=(' + str(i) + ', ' + str(j) +')')

        x_low_lim = min_x + inc_x * i
        x_up_lim = min_x + inc_x * (i + 1)
        y_low_lim = max_y - inc_y * (j + 1)
        y_up_lim = max_y - inc_y * j

        for xi, yi in zip(x, y):
            if ((x_low_lim) <= xi and xi < (x_up_lim)) and ((y_low_lim) < yi and yi <= (y_up_lim)):
                weight += 1 * grid_count * grid_count

        if weight == 0:
            weight += percentage_weight_for_empty_point * grid_count * grid_count

        w = (round(float(weight) / float(N), 10))
        data.append(w)

count = 1
with open(output_txt_file, 'w') as f:
    for i in data:
        if count != grid_count_horizontal * grid_count_vertical:
            f.write('{:.6f},'.format(i))

            if count % grid_count_horizontal == 0:
                f.write('\n')
        else:
            f.write('{:.6f}'.format(i))

        count += 1

print('Finished !!')