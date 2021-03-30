from energyMinimization import *
from skimage.measure import *


# build the node class
class node:
    movable = True

    def __init__(self, name, loc):
        self.name = name
        self.loc = loc


input_node_data_file = "Datasets/GeneratedData/MaxFlow Output/PBLH_nodes_32_32.csv"
input_weight_data_file = "Datasets/GeneratedData/MaxFlow Output/PBLH_32_32.txt"

output_filename = 'output_PBLH_EMISS_32_32'
square_grid = 32
output_image_size = [512, 512]

is_image_output = True
input_img_file = "Datasets/GeneratedData/MaxFlow Output/EMISS512.png"



grid_horiz = square_grid
grid_vert = square_grid

if __name__ == "__main__":
    print("Started - Reading data from")
    nodes = []

    sample_val = []
    input_file = open(input_node_data_file, "r")
    in_total_str = ''
    in_str = input_file.readlines()
    for i in range(len(in_str)):
        in_total_str += in_str[i].replace('\n', '').replace(' ', '')

    val_str = in_total_str.split(",")
    input_file.close()

    for i in range(grid_horiz + 1):
        x = []
        for j in range(grid_vert + 1):
            x.append(node(str(i) + "_" + str(j), Point(i, j)))
        nodes.append(x)

    counter = 0
    for j in range(grid_vert, -1, -1):
        for i in range(grid_horiz + 1):
            nodes[i][j].loc = Point(float(val_str[counter].split('__')[0]), float(val_str[counter].split('__')[1]))
            counter += 1

    poly_draw_for_maxflow(output_filename + "_polygon.png", output_image_size, nodes, grid_horiz, grid_vert)
    print("Ended with polygon drawing")

    values_actual = read_text_file(input_weight_data_file, grid_horiz, grid_vert)
    values = np.zeros((grid_horiz, grid_vert))

    for x in range(grid_horiz):
        for y in range(grid_vert):
            values[x][y] = values_actual[x, y]

    values = values / np.sum(values)

    # all values sum to totalarea
    values = values * grid_horiz * grid_vert

    out_file_name = "output/out_log_" + output_filename + ".txt"
    output_txt_file = open(out_file_name, "w")
    output_txt_file.write("Nothing, |UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N,"
                          " Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, Average Aspect Ratio (height/width),"
                          "Average Aspect Ratio min(height/width)/max(height/width), Nothing\n")
    output_txt_file.close()
    all_error_calc(values, nodes, grid_horiz, grid_vert, 0, output_filename, 1)

    if is_image_output:
        input_image = Image.open(input_img_file)
        input_image = input_image.convert("RGBA")
        output_image_size = input_image.size

        output_image_path = imageDrawForMaxFlow(input_image, nodes, output_filename + "_output.png", grid_horiz, grid_vert)

        output_image = Image.open(output_image_path)
        out_image = np.asarray(output_image.convert("RGBA"))
        in_image = np.asarray(input_image)

        mse_error = compare_mse(in_image, out_image)
        psnr_error = compare_psnr(in_image, out_image)
        ssim_error = compare_ssim(in_image, out_image, multichannel=True)

        output_txt_file = open(out_file_name, "a")
        output_txt_file.write("\n\nMSE : " + str(round(mse_error, 4)) + "\n")
        output_txt_file.write("PSNR : " + str(round(psnr_error, 4)) + "\n")
        output_txt_file.write("SSIM : " + str(round(ssim_error, 4)) + "\n")

    print("Finished")
