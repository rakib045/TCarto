import numpy as np
import pandas as pd



df=pd.read_csv(r"Aggregation.csv",header=None)


x = df[df.columns[0]]
y = df[df.columns[1]]
cluster=16

N=len(x)
print(N)
grid=[2,4,8,16,32,64,128,256,512,1024]

print(grid)
for g in grid:
    grid_count_horizontal =g
    grid_count_vertical = g
    file_init_name = "input/Aggregation_cluster_3_grid_"
    output_txt_file = file_init_name + str(grid_count_horizontal) + "_" + str(grid_count_vertical) + ".txt"
    inc_x=float(max(x)-min(x))/float(grid_count_horizontal)
    inc_y=float(max(y)-min(y))/float(grid_count_vertical)

    data=[]
    min_x=min(x)
    min_y=min(y)
    max_x=max(x)
    max_y=max(y)


    for j in range(grid_count_vertical):
        for i in range(grid_count_horizontal):

            weight=0
            #print('....'+str(i)+'.....'+str(j)+'.......')
            for xi,yi in zip(x,y):
                x_low_lim=min_x+(inc_x*(i))
                x_up_lim=min_x+inc_x*(i+1)
                y_low_lim=max(y)-(inc_y*(j+1))
                y_up_lim=max(y)-(inc_y*(j))
                if ((x_low_lim)<=xi and xi<(x_up_lim))and ((y_low_lim)<yi and yi<=(y_up_lim)):
                    #print('xval....'+str(xi)+'.yval....'+str(yi)+'.......')
                    weight+=1*g*g




            if weight==0:
                weight+=(0.8)*g*g
                #weight+=(1.0/(2*(float(N))))*g*g


            w=(round(float(weight)/float(N),10))
            #print("final weight",w)
            data.append(w)

    data=data
    print(data)

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