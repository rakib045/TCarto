# TCarto
TCarto is a simple, scalable, parallel code optimization for Table Cartograms.
We present a table cartogram generator written in python. It uses local optimization based approach to construct table cartogram that gradually transforms the cells to improve the area discrepancies.

This readme explains how to set-up and use this code as well as the input data format.

# Input Data Format:
This code expect only one input data file. We suggest to unzip Datasets.zip and put it in the same directory. Please, copy the data file you are interested into 'input' folder and always generate output into 'output' folder. The only input data file is a (.txt) file that holds the weights/area values of the grids/cells of the cartogram. For example, an input data file (e.g. D31_cluster_31_grid_2_2.txt, located at '\Datasets\Synthetic Datasets\Shape Dataset' folder) is for a 2 by 2 grid cartogram. It holds data similar like below.

1.0916129032,0.9161290323,

0.8670967742,1.1225806452

It means the targeted weights or area values of the output 2x2 cartogram would be 1.0916, 0.9161, 0.8671 and 1.1226 for top left, top right, bottom left and bottom right cell/grid respectively.

# Set-up and dependencies:
1. Python 3.7 or higher version should be installed. You can download it from here: https://www.python.org/downloads/
2. MiniConda3 or Conda 4.6 or higher version should be installed.
3. Install 'CVXOPT' by using 'pip install cvxopt' or 'conda install -c conda-forge cvxopt'
4. Install 'Sympy' by using 'pip install sympy'
5. Install 'Pillow' by using 'pip install Pillow'
6. Install 'matplotlib' using 'conda install -c conda-forge matplotlib'

# Running code and generating cartogram:

1. Unzip Datasets.zip
2. Copy the interested data file inside from 'Datasets' folder to the 'input' folder. For example, copy 'PBLH_grid64_64.txt' file from 'Datasets\Real-life Dataset\' folder to 'input' folder.
3. Navigate (using terminal on Linux/Ubuntu and command prompt on Windows) to the directory with the TCarto root directory.
4. Run the executable using the following command
'python <python_file_of_expected_algorithm> <number_of_squared_grid> <input_data_file> <output_log_and_image_filename>
for example, 'python PrescribedAreaDrawingDivideConq.py 64 "input/PBLH_10_new_grid64_64.txt" "DivCon_PBLH_10_new_grid64_64"'

python_file_of_expected_algorithm -> There are four python files here to run : PrescribedAreaDrawing.py is the single threaded code, PrescribedAreaDrawingParallelCode.py is for parallel programming with two phases, PrescribedAreaDrawingDivideConq.py is for parallel programming with divide and conquer strategy, PrescribedAreaDrawingDivideConqIMG.py is similar like PrescribedAreaDrawingDivideConq.py but it takes another image file hardcodedly inside code, applies DIV-CON technique on top of that image file and provides distorted image file.

number_of_squared_grid -> Total number of grid for square shape

input_data_file -> A (.txt) file as an input data file having weights of each and every cells of the cartogram

output_log_and_image_filename -> This is just the filename for the output image and output log file

5. You can also use script file to run multiple command at the same time.
  For windows: 
    a. Write all the commands into 'run_win.cmd' file
    b. Double click to execute it
  For Linux/Ubuntu:
    a. Write all the commands into 'run_linux.sh' file
    b. Make this file as executable by using 'chmod +x run_linux.sh' command
    c. Run the file using '.\run_linux.sh'

6. The output image and log files will be generated into the 'output' folder.
