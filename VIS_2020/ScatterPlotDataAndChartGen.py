import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

filename = 'cluster_2'
grid_count = 64
output_txt_file = "input/cluster_2_grid_64_64.txt"
percentage_weight_for_empty_point = 0.75

min_x_axis = 0
max_x_axis = 12
min_y_axis = 0
max_y_axis = 12

centers1 = [(3, 3)]
centers2 = [(8, 8)]
centers3 = [(3, 8)]
centers4 = [(4, 7)]

print('Data is generating ...')
X1, y1 = make_blobs(n_samples=500, centers=centers1, n_features=2, cluster_std=0.3, random_state=42,)
X2, y2 = make_blobs(n_samples=200, centers=centers2, n_features=2, cluster_std=0.6, random_state=42,)
X3, y3 = make_blobs(n_samples=100, centers=centers3, n_features=2, cluster_std=0.45, random_state=42,)
X4, y4 = make_blobs(n_samples=100, centers=centers4, n_features=2, cluster_std=0.45, random_state=42,)
X = np.concatenate((X1, X2, X3, X4))
#print(X[:])

fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(np.arange(min_x_axis, max_x_axis, 1))
plt.yticks(np.arange(min_y_axis, max_y_axis, 1))
plt.show()
fig.savefig("input/" + filename + "_image.png")


csv_filename = "input/" + filename + "_datapoints.csv"
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
        print('(i, j)=(' + str(i) + ', ' + str(j) +')')

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