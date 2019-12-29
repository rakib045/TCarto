from energyMinimization import *
import math as m
from sympy import *
from PrescribedAreaDrawingDivideConq import *
import multiprocessing as mp
from skimage.measure import *



square_grid = 8
input_data_file = 'input/weight_8_8.txt'
input_img_file = 'input/weather_tsk.png'
output_img_filename = 'output_GC'


# When is_stop_with_rmse = True, rmse_threshold will work only then. Iteration stops when it crosses rmse less than threshold
is_stop_with_rmse = True
rmse_threshold = 0.18

iteration = 30
error_threshold = 0.0001
def getWeightedTrianglesBorder(triangle, weight_list):
    first_point_triangle = triangle[0]
    second_point_triangle = triangle[1]
    third_point_triangle = triangle[2]

    height = m.pow((first_point_triangle[0] - second_point_triangle[0]), 2) + m.pow(
        (first_point_triangle[1] - second_point_triangle[1]), 2)
    height = m.sqrt(height)

    triangle_points = []
    vertical_cumm_y = 0

    for i in range(len(weight_list)):
        base = 2 * weight_list[i] / height
        points = []
        points.append([second_point_triangle[0], vertical_cumm_y])
        points.append([first_point_triangle[0], 0])
        vertical_cumm_y += base
        points.append([second_point_triangle[0], vertical_cumm_y])

        triangle_points.append(points)

    return triangle_points


def getWeightedTrianglesCenterTopToBottom(triangle, weight_list_array):
    top_point = Point2D(triangle[0])
    left_base_point = Point2D(triangle[1])
    right_base_point = Point2D(triangle[2])
    base_line = Line(left_base_point, right_base_point)

    outer_triangle_height = float(base_line.distance(top_point))

    triangle_points = []

    for i in range(len(weight_list_array) - 1):

        left_line = Line(top_point, left_base_point)
        right_line = Line(top_point, right_base_point)

        area_left = weight_list_array[i, 0]
        area_right = weight_list_array[i, 1]
        area_below = np.sum(weight_list_array[i + 1:, :])

        height = 2 * area_below / left_base_point.distance(right_base_point)
        y_coordinate = outer_triangle_height - height

        parallel_line_base = base_line.parallel_line(Point([0, y_coordinate]))
        left_intersection_point = left_line.intersection(parallel_line_base)
        right_intersection_point = right_line.intersection(parallel_line_base)

        middle_point = Point2D(0, y_coordinate)

        # error_threshold = 0.0001
        min_x = float(left_intersection_point[0].x)
        max_x = float(right_intersection_point[0].x)

        while True:
            random_float_number = np.random.uniform(min_x, max_x)

            middle_point = Point2D(random_float_number, y_coordinate)
            left_calc_area = float(Polygon(top_point, middle_point, left_base_point).area)

            if abs(left_calc_area - area_left) < error_threshold:
                break
            elif left_calc_area < area_left:
                min_x = random_float_number
            elif left_calc_area > area_left:
                max_x = random_float_number

        # Top Left Area
        points = []
        points.append(triangle[1])  # Left Point
        points.append([float(top_point[0]), float(top_point[1])])  # Top Point
        points.append([random_float_number, y_coordinate])  # Middle Point
        triangle_points.append(points)

        # Top Right Area
        points = []
        points.append([random_float_number, y_coordinate])  # Middle Point
        points.append([float(top_point[0]), float(top_point[1])])  # Top Point
        points.append(triangle[2])  # Right Point
        triangle_points.append(points)
        '''
        print("Left=" + str(area_left) + ", Right=" + str(area_right) + ", Below=" + str(area_below))
        print("Area Left=" + str(float(Polygon(top_point, middle_point, left_base_point).area))
              + ", Area Right=" + str(float(Polygon(top_point, right_base_point, middle_point).area)))
        print([random_float_number, y_coordinate])
        '''
        top_point = middle_point

    bottom_two_triangles_height = float(base_line.distance(top_point))

    area_left = weight_list_array[len(weight_list_array) - 1, 0]
    middle_point_x = 2 * area_left / bottom_two_triangles_height

    middle_point = Point2D(triangle[1][0] + middle_point_x, triangle[1][1])

    points = []

    # Top Left Area
    points.append(triangle[1])  # Left Point
    points.append([float(top_point[0]), float(top_point[1])])  # Top Point
    points.append([float(middle_point[0]), float(middle_point[1])])  # Middle Point
    triangle_points.append(points)

    # Top Right Area
    points = []
    points.append([float(middle_point[0]), float(middle_point[1])])  # Middle Point
    points.append([float(top_point[0]), float(top_point[1])])  # Top Point
    points.append(triangle[2])  # Right Point
    triangle_points.append(points)

    '''
    print("Left=" + str(area_left) + ", Right=" + str(weight_list_array[len(weight_list_array)-1, 1])
          + ", Below=" + str(0))
    print("Area Left=" + str(float(Polygon(top_point, middle_point, left_base_point).area))
          + ", Area Right=" + str(float(Polygon(top_point, right_base_point, middle_point).area)))
    print([float(middle_point[0]), float(middle_point[1])])
    '''
    return triangle_points


def getWeightedTrianglesCenterBottomToTop(triangle, weight_list_array):
    bottom_point = Point2D(triangle[0])
    left_base_point = Point2D(triangle[1])
    right_base_point = Point2D(triangle[2])
    base_line = Line(left_base_point, right_base_point)

    outer_triangle_height = float(base_line.distance(bottom_point))

    triangle_points = []

    for i in range(len(weight_list_array) - 1, 0, -1):

        left_line = Line(bottom_point, left_base_point)
        right_line = Line(bottom_point, right_base_point)

        area_left = weight_list_array[i, 0]
        area_right = weight_list_array[i, 1]
        area_above = np.sum(weight_list_array[0:i, :])

        height = 2 * area_above / left_base_point.distance(right_base_point)
        y_coordinate = height

        parallel_line_base = base_line.parallel_line(Point([0, y_coordinate]))
        left_intersection_point = left_line.intersection(parallel_line_base)
        right_intersection_point = right_line.intersection(parallel_line_base)

        middle_point = Point2D(0, y_coordinate)

        # error_threshold = 0.0001
        min_x = float(left_intersection_point[0].x)
        max_x = float(right_intersection_point[0].x)

        while True:
            random_float_number = np.random.uniform(min_x, max_x)

            middle_point = Point2D(random_float_number, y_coordinate)
            left_calc_area = float(Polygon(bottom_point, left_base_point, middle_point).area)

            if abs(left_calc_area - area_left) < error_threshold:
                break
            elif left_calc_area < area_left:
                min_x = random_float_number
            elif left_calc_area > area_left:
                max_x = random_float_number

        # Top Left Area
        points = []
        points.append(triangle[1])  # Left Point
        points.append([random_float_number, y_coordinate])  # Middle Point
        points.append([float(bottom_point[0]), float(bottom_point[1])])  # Bottom Point
        triangle_points.append(points)

        # Top Right Area
        points = []
        points.append(triangle[2])  # Right Point
        points.append([float(bottom_point[0]), float(bottom_point[1])])  # Bottom Point
        points.append([random_float_number, y_coordinate])  # Middle Point

        triangle_points.append(points)
        '''
        print("Left=" + str(area_left) + ", Right=" + str(area_right) + ", Above=" + str(area_above))
        print("Area Left=" + str(float(Polygon(left_base_point, middle_point, bottom_point).area))
              + ", Area Right=" + str(float(Polygon(right_base_point, bottom_point, middle_point).area)))
        print([random_float_number, y_coordinate])
        '''
        bottom_point = middle_point

    top_two_triangles_height = float(base_line.distance(bottom_point))

    area_left = weight_list_array[0, 0]
    middle_point_x = 2 * area_left / top_two_triangles_height

    middle_point = Point2D(triangle[1][0] + middle_point_x, triangle[1][1])

    points = []

    # Top Left Area
    points.append(triangle[1])  # Left Point
    points.append([float(middle_point[0]), float(middle_point[1])])  # Middle Point
    points.append([float(bottom_point[0]), float(bottom_point[1])])  # Bottom Point

    triangle_points.append(points)

    # Top Right Area
    points = []
    points.append(triangle[2])  # Right Point
    points.append([float(bottom_point[0]), float(bottom_point[1])])  # Bottom Point
    points.append([float(middle_point[0]), float(middle_point[1])])  # Middle Point

    triangle_points.append(points)
    '''
    print("Left=" + str(area_left) + ", Right=" + str(weight_list_array[0, 1])
          + ", Above=" + str(0))
    print("Area Left=" + str(float(Polygon(left_base_point, middle_point, bottom_point).area))
          + ", Area Right=" + str(float(Polygon(right_base_point, bottom_point, middle_point).area)))
    print([float(middle_point[0]), float(middle_point[1])])
    '''
    return triangle_points


def findTargetedPointProjection(values, grid_count):
    sum_S = np.sum(values)

    # Start - Split Main Weighted Table into 2 smaller tables : top and bottom

    table_top = []
    table_bottom = []

    row_weight_cumm_count = 0
    is_first_enter_split_row = True
    for i in range(grid_count):
        row_weight_cumm_count += np.sum(values[i])
        if row_weight_cumm_count < (sum_S / 2):
            table_top.append(values[i])
        elif is_first_enter_split_row and row_weight_cumm_count > (sum_S / 2):
            is_first_enter_split_row = False

            # (S/2 - Cumulative Weight of all previous rows)
            t_lambda = ((sum_S / 2) - (row_weight_cumm_count - np.sum(values[i]))) / np.sum(values[i])
            table_top.append(values[i] * t_lambda)
            table_bottom.append(values[i] * (1 - t_lambda))

        elif not is_first_enter_split_row and row_weight_cumm_count > (sum_S / 2):
            table_bottom.append(values[i])
    table_top = np.array(table_top)
    table_bottom = np.array(table_bottom)

    # End - Split Main Weighted Table into 2 smaller tables : top and bottom

    # Start - Triangle Points Calculation
    temp_triangle_points = []

    total_vertical_dist = m.sqrt(sum_S)

    top_horizontal_cummulative_dist = 0
    bottom_horizontal_cummulative_dist = 0

    count_iteration = m.ceil(len(table_top[0]) / 2 + 1)
    for i in range(count_iteration):
        if i == 0:
            area = np.sum(table_top, axis=0)[i]
            top_horizontal_cummulative_dist = area * 2 / total_vertical_dist
            points = []
            points.append([top_horizontal_cummulative_dist, 0])
            points.append([0, 0])
            points.append([0, total_vertical_dist])
            temp_triangle_points.append(points)

        else:
            temp_index = 2 * i - 1
            bottom_area = np.sum(table_bottom, axis=0)[temp_index] + np.sum(table_bottom, axis=0)[temp_index - 1]

            points = []
            points.append([top_horizontal_cummulative_dist, 0])
            points.append([bottom_horizontal_cummulative_dist, total_vertical_dist])

            bottom_horizontal_cummulative_dist += bottom_area * 2 / total_vertical_dist

            points.append([bottom_horizontal_cummulative_dist, total_vertical_dist])
            temp_triangle_points.append(points)

            # print(bottom_temp)
            if i == (count_iteration - 1):
                top_area = np.sum(table_top, axis=0)[temp_index]
                points = []
                points.append([top_horizontal_cummulative_dist, 0])

                top_horizontal_cummulative_dist += top_area * 2 / total_vertical_dist
                points.append([top_horizontal_cummulative_dist, 0])
                points.append([top_horizontal_cummulative_dist, bottom_horizontal_cummulative_dist])
                temp_triangle_points.append(points)

            else:
                top_area = np.sum(table_top, axis=0)[temp_index] + np.sum(table_top, axis=0)[temp_index + 1]
                points = []
                points.append([bottom_horizontal_cummulative_dist, total_vertical_dist])
                points.append([top_horizontal_cummulative_dist, 0])

                top_horizontal_cummulative_dist += top_area * 2 / total_vertical_dist

                points.append([top_horizontal_cummulative_dist, 0])
                temp_triangle_points.append(points)
            # print(top_temp)

    # print(temp_triangle_points)
    # End - Triangle Points Calculation

    # Start - Weighted Triangle Calculation

    targeted_nodes = grid_node_generation(node, grid_count, grid_count)

    split_row = len(table_top)
    for i in range(len(temp_triangle_points)):
        if i == 0:
            top_left_triangles = getWeightedTrianglesBorder(temp_triangle_points[0], table_top[:, 0])
            for index in range(len(top_left_triangles)):
                targeted_nodes[i][index].loc = Point2D(top_left_triangles[index][0])
                targeted_nodes[i + 1][index].loc = Point2D(top_left_triangles[index][1])
        elif i == (len(temp_triangle_points) - 1):
            last_index = len(table_top[0])
            top_right_triangles = getWeightedTrianglesBorder(temp_triangle_points[last_index],
                                                             table_top[:, last_index - 1])
            for index in range(len(top_right_triangles)):
                targeted_nodes[i][index].loc = Point2D(top_right_triangles[index][0])
                targeted_nodes[i - 1][index].loc = Point2D(top_right_triangles[index][1])
        elif i % 2 == 1:
            middle_triangles = getWeightedTrianglesCenterTopToBottom(temp_triangle_points[i],
                                                                     table_bottom[:, (i - 1): (i - 1) + 2])

            for index in range(0, len(middle_triangles), 1):
                if index % 2 == 0:
                    targeted_nodes[i - 1][split_row + int(index / 2)].loc = Point2D(middle_triangles[index][0])
                    targeted_nodes[i][split_row + int(index / 2)].loc = Point2D(middle_triangles[index][2])
                elif index % 2 == 1:
                    targeted_nodes[i][split_row + int(index / 2)].loc = Point2D(middle_triangles[index][0])
                    targeted_nodes[i + 1][split_row + int(index / 2)].loc = Point2D(middle_triangles[index][2])

        elif i % 2 == 0:
            middle_triangles = getWeightedTrianglesCenterBottomToTop(temp_triangle_points[i],
                                                                     table_top[:, (i - 1): (i - 1) + 2])
            for index in range(0, len(middle_triangles), 1):
                if index % 2 == 0:
                    targeted_nodes[i - 1][split_row - 1 - int(index / 2)].loc = Point2D(middle_triangles[index][0])
                    targeted_nodes[i][split_row - 1 - int(index / 2)].loc = Point2D(middle_triangles[index][1])
                elif index % 2 == 1:
                    targeted_nodes[i][split_row - 1 - int(index / 2)].loc = Point2D(middle_triangles[index][2])
                    targeted_nodes[i + 1][split_row - 1 - int(index / 2)].loc = Point2D(middle_triangles[index][0])

    #poly_draw_top_to_bottom("output_generalCase_target", [800, 800], targeted_nodes, grid_count, grid_count)

    # End - Weighted Triangle Calculation
    return targeted_nodes

def updatedNodeGeneralCase(nodes, values, targeted_nodes, grid_count_horizontal, grid_count_vertical):
    for i in range(grid_count_horizontal + 1):
        for j in range(grid_count_vertical + 1):
            mid_point = nodes[i][j].loc.midpoint(targeted_nodes[i][j].loc)
            nodes[i][j].loc = mid_point

    return nodes

def run_GeneralCase(input_img_file, input_data_file, square_grid, output_img_filename):
    grid_count_horizontal_actual = square_grid
    grid_count_vertical_actual = square_grid

    input_image = Image.open(input_img_file)
    input_image = input_image.convert("RGBA")
    output_image_size = input_image.size

    pre_processing_start_time = time()
    values_actual = read_text_file_actual_order(input_data_file, grid_count_horizontal_actual,
                                                grid_count_vertical_actual)
    values = values_actual / np.sum(values_actual) * square_grid * square_grid

    # nodes = grid_node_generation(node, grid_count_horizontal_actual, grid_count_vertical_actual)
    print("Calculating Targeted Nodes ...")
    targeted_nodes = findTargetedPointProjection(values, square_grid)
    print("Finished Targeted Nodes Calculation !!!")

    pre_processing_end_time = time()
    pre_processing_time = pre_processing_end_time - pre_processing_start_time

    output_image_path = ''
    total_algo_iteration_processing_time = []

    if is_stop_with_rmse:

        out_file_name = "output/" + output_img_filename + "_log.txt"
        output_txt_file = open(out_file_name, "w")
        output_txt_file.write("Iteration, |UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N, " +
                              "Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, " +
                              "Average Aspect Ratio min(height/width)/max(height/width), " +
                              "Average Aspect Ratio (height/width), Processing Time(sec)\n")
        output_txt_file.close()
        print("Pre Processing time(sec): " + str(round(pre_processing_time, 4)))

        nodes = grid_node_generation(node, grid_count_horizontal_actual, grid_count_vertical_actual)
        nodes[0][0].movable = False
        nodes[0][grid_count_vertical_actual].movable = False
        nodes[grid_count_horizontal_actual][0].movable = False
        nodes[grid_count_horizontal_actual][grid_count_vertical_actual].movable = False



        for x in range(iteration):

            print("------------------------------------------")
            print('iteration: ' + str(x + 1) + '(out of ' + str(iteration) + '): ')
            iteration_start_time = time()
            nodes = updatedNodeGeneralCase(nodes, values, targeted_nodes,
                                           grid_count_horizontal_actual, grid_count_vertical_actual)
            iteration_end_time = time()
            estimation_time = iteration_end_time - iteration_start_time

            total_algo_iteration_processing_time.append(estimation_time)

            #poly_draw_top_to_bottom("output_generalCase_It_" + str(x + 1), [800, 800], nodes,
            #                        grid_count_horizontal_actual, grid_count_vertical_actual)

            current_rmse = all_error_print(values, nodes, grid_count_horizontal_actual, grid_count_vertical_actual,
                                           estimation_time, output_img_filename, (x + 1))

            if current_rmse[0] < rmse_threshold:
                break

        poly_draw_top_to_bottom("output_generalCase", output_image_size, nodes,
                                grid_count_horizontal_actual, grid_count_vertical_actual)

        output_image_path = newImageDrawTopToBottom(input_image, nodes, output_img_filename + '_image',
                                                    grid_count_horizontal_actual,
                                                    grid_count_vertical_actual)

        output_txt_file = open(out_file_name, "a")
        output_txt_file.write("\n\nPre-Processing Time(sec): " + str(round(pre_processing_time, 4)))
        output_txt_file.close()


    else:
        out_file_name = "output/" + output_img_filename + "_log.txt"
        output_txt_file = open(out_file_name, "w")
        output_txt_file.write("|UV-EV|/EV, UV/EV - 1, RMSE, MQE = (((|UV-EV|/EV) ** 2) ** 0.5)/N, " +
                              "Updated MQE = (((|UV-EV|/(UV+EV)) ** 2) ** 0.5)/N, " +
                              "Average Aspect Ratio min(height/width)/max(height/width), " +
                              "Average Aspect Ratio (height/width), Processing Time(sec)\n")
        output_txt_file.close()
        print("Processing time(sec): " + str(round(pre_processing_time, 4)))
        poly_draw_top_to_bottom("output_generalCase_errorless", output_image_size, targeted_nodes,
                                grid_count_horizontal_actual, grid_count_vertical_actual)
        output_image_path = newImageDrawTopToBottom(input_image, targeted_nodes, output_img_filename + '_errorless_image',
                                               grid_count_horizontal_actual,
                                               grid_count_vertical_actual)

        all_error_print(values, targeted_nodes, grid_count_horizontal_actual, grid_count_vertical_actual,
                        pre_processing_time, output_img_filename)

    output_image = Image.open(output_image_path)
    out_image = np.asarray(output_image.convert("RGBA"))
    in_image = np.asarray(input_image)

    mse_error = compare_mse(in_image, out_image)
    psnr_error = compare_psnr(in_image, out_image)
    ssim_error = compare_ssim(in_image, out_image, multichannel=True)

    processing_time = pre_processing_time + np.sum(total_algo_iteration_processing_time)

    output_txt_file = open(out_file_name, "a")
    output_txt_file.write("\n\nTotal Processing Time(sec): " + str(round(processing_time, 4)) + "\n")
    output_txt_file.write("\nMSE : " + str(round(mse_error, 4)) + "\n")
    output_txt_file.write("PSNR : " + str(round(psnr_error, 4)) + "\n")
    output_txt_file.write("SSIM : " + str(round(ssim_error, 4)) + "\n")
    output_txt_file.close()


if __name__ == '__main__':
    run_GeneralCase(input_img_file, input_data_file, square_grid, output_img_filename)
