from energyMinimization import *
from skimage.measure import *
import os, shutil


# build the node class
class node:
    movable = True

    def __init__(self, name, loc):
        self.name = name
        self.loc = loc


input_node_folder = "Datasets/GeneratedData/MaxFlowOutput/wrf_64_64"
input_weight_folder = "Datasets/GeneratedData/Weights/wrf_64_64"
input_image_folder = "Datasets/GeneratedData/wrfImages"
output_initial_folder = "Datasets/GeneratedData/MaxFlowOutputAfterCalcTCartoSys/64_64"

input_node_filenames = ["final_nodes_ALBEDO_64_64.csv", "final_nodes_EMISS_64_64.csv", "final_nodes_SMOIS_64_64.csv",
                        "final_nodes_PBLH_64_64.csv", "final_nodes_PSFC_64_64.csv", "final_nodes_ALBEDO_64_64.csv",
                        "final_nodes_SH2O_64_64.csv", "final_nodes_SMOIS_64_64.csv", "final_nodes_EMISS_64_64.csv",
                        "final_nodes_PBLH_64_64.csv"]

input_weight_filenames = ["ALBEDO_64_64.txt", "EMISS_64_64.txt", "SMOIS_64_64.txt", "PBLH_64_64.txt", "PSFC_64_64.txt",
                          "ALBEDO_64_64.txt", "SH2O_64_64.txt", "SMOIS_64_64.txt","EMISS_64_64.txt", "PBLH_64_64.txt"]

input_image_filenames = ["TSK_512.png", "SMOIS_512.png", "LWUPB_512.png","U10_512.png", "Q2_512.png",
                         "U10_512.png", "ALBEDO_512.png", "ALBEDO_512.png", "PSFC_512.png", "EMISS_512.png"]

input_weight_label = ["ALBEDO","EMISS","SMOIS","PBLH","PSFC","ALBEDO","SH2O","SMOIS","EMISS", "PBLH"]
input_image_label = ["TSK","SMOIS", "LWUPB","U10","Q2","U10","ALBEDO","ALBEDO","PSFC", "EMISS"]
square_grid = 64
output_image_size = [512, 512]


#input_node_data_file = "Datasets/GeneratedData/MaxFlow Output/PBLH_nodes_32_32.csv"
#input_weight_data_file = "Datasets/GeneratedData/MaxFlow Output/PBLH_32_32.txt"
#input_img_file = "Datasets/GeneratedData/MaxFlow Output/EMISS512.png"

#output_filename = 'output_PBLH_EMISS_32_32'


grid_horiz = square_grid
grid_vert = square_grid

if __name__ == "__main__":

    print("Everything started ...\n")
    out_result_file_name = output_initial_folder + "/result.csv"
    output_result_file_name = open(out_result_file_name, "w")
    output_result_file_name.write("Weight, Image, RMSE, MQE, AAR (height/width), AAR min/max of height, MSE, PSNR, SSIM\n")

    for index in range(len(input_node_filenames)):
        output_folder = output_initial_folder + "/" + input_weight_label[index] + "_" + input_image_label[index]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = output_folder + "/" + input_weight_label[index] + "_" + input_image_label[index] + "_" + str(grid_horiz) + "_" + str(grid_vert)

        nodes = []

        sample_val = []
        input_node_data_file = input_node_folder + "/" + input_node_filenames[index]
        print("Started - Reading data from " + input_node_data_file)

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
        print("finished polygon drawing ...")

        input_weight_data_file = input_weight_folder + "/" + input_weight_filenames[index]
        shutil.copyfile(input_weight_data_file, output_folder + "/" + input_weight_filenames[index])
        values_actual = read_text_file(input_weight_data_file, grid_horiz, grid_vert)
        values = np.zeros((grid_horiz, grid_vert))

        for x in range(grid_horiz):
            for y in range(grid_vert):
                values[x][y] = values_actual[x, y]

        values = values / np.sum(values)

        # all values sum to totalarea
        values = values * grid_horiz * grid_vert

        out_file_name = output_filename + "_log.txt"
        output_txt_file = open(out_file_name, "w")
        output_txt_file.write("|UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N,"
                              " Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, Average Aspect Ratio (height/width),"
                              "Average Aspect Ratio min(height/width)/max(height/width)\n")
        output_txt_file.close()
        rmse_error, mqe_error, average_aspect_ratio_h_by_w, average_aspect_ratio_min_by_max = \
            all_error_calc_max_flow(values, nodes, grid_horiz, grid_vert, output_filename + "_log.txt")

        input_img_file = input_image_folder + "/" + input_image_filenames[index]
        input_image = Image.open(input_img_file)
        input_image.save(output_folder + "/" + input_image_filenames[index])

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
        output_txt_file.close()

        csv_str = input_weight_label[index] + ","
        csv_str += input_image_label[index] + ","
        csv_str += str(round(rmse_error, 6)) + ","
        csv_str += str(round(mqe_error, 6)) + ","
        csv_str += str(round(average_aspect_ratio_h_by_w, 6)) + ","
        csv_str += str(round(average_aspect_ratio_min_by_max, 6)) + ","

        csv_str += str(round(mse_error, 4)) + ","
        csv_str += str(round(psnr_error, 4)) + ","
        csv_str += str(round(ssim_error, 4)) + "\n"

        output_result_file_name.write(csv_str)

    output_result_file_name.close()
    print("Finished Everything !!!")
