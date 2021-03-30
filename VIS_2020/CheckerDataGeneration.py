import matplotlib
import numpy as np
import heapq
import pandas as pd
matplotlib.use("agg")
#from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame


null_grid_weight = 0.8
filepath = "Datasets/"
# grid=[2,4,8,16,32,64,128,256,512,1024]
grid = [32]
zero_zero_val = 1.0
zero_one_val = 10.0


data = []

for g in grid:
    data = []
    for i in range(g):
        for j in range(g):
            if (i+j)%2 == 0:
                data.append(zero_zero_val)
            else:
                data.append(zero_one_val)

    print("TCarto Data is generating for " + str(g) + " by " + str(g))

    file_init_name = filepath + "TCarto_checker_data_"
    output_txt_file = file_init_name + str(g) + "_" + str(g) + ".txt"

    count = 1
    with open(output_txt_file, 'w') as f:
        for i in data:

            if count != g * g:
                f.write('{:.10f},'.format(i))
                # f.write('%s,'%i)

                if (count % g == 0):
                    f.write('\n')
            else:
                f.write('{:.10f}'.format(i))
                # f.write('%s'%i)

            count += 1
