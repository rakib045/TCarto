#!/bin/bash
python PrescribedAreaDrawingDivideConq.py 8 "input/data_cat_8_8.txt" "DivideAndConq_ParallelCode_Gaussian_8_8"
python PrescribedAreaDrawingDivideConq.py 16 "input/data_cat_16_16.txt" "DivideAndConq_ParallelCode_Gaussian_16_16"

python PrescribedAreaDrawingParallelCode.py 8 5 "input/data_cat_8_8.txt" "ParallelCode_Gaussian_8_8"
python PrescribedAreaDrawingParallelCode.py 16 5 "input/data_cat_16_16.txt" "ParallelCode_Gaussian_16_16"

python PrescribedAreaDrawing.py 8 5 "input/data_cat_8_8.txt" "SingleThread_Gaussian_8_8"
python PrescribedAreaDrawing.py 16 5 "input/data_cat_16_16.txt" "SingleThread_Gaussian_16_16"

