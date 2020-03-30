import brisque

brisq = brisque.BRISQUE()

imageList = ['input/room1.jpg',
             'output/output_DivCon_room.png',
             'output/output_weather_data/ALBEDO_TSK/TSK512.png',
             'output/output_weather_data/ALBEDO_TSK/basecase/output_baseline_image.png',
             'output/output_weather_data/ALBEDO_TSK/DivCon/output_DivCon.png',
             'output/output_weather_data/ALBEDO_TSK/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/ALBEDO_U10/U10512.png',
             'output/output_weather_data/ALBEDO_U10/basecase/output_baseline_image.png',
             'output/output_weather_data/ALBEDO_U10/DivCon/output_DivCon.png',
             'output/output_weather_data/ALBEDO_U10/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/EMISS_PSFC/PSFC512.png',
             'output/output_weather_data/EMISS_PSFC/basecase/output_baseline_image.png',
             'output/output_weather_data/EMISS_PSFC/DivCon/output_DivCon.png',
             'output/output_weather_data/EMISS_PSFC/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/EMISS_SMOIS/smois512.png',
             'output/output_weather_data/EMISS_SMOIS/basecase/output_baseline_image.png',
             'output/output_weather_data/EMISS_SMOIS/DivCon/output_DivCon.png',
             'output/output_weather_data/EMISS_SMOIS/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/PBLH_EMISS/EMISS512.png',
             'output/output_weather_data/PBLH_EMISS/basecase/output_baseline_image.png',
             'output/output_weather_data/PBLH_EMISS/DivCon/output_DivCon.png',
             'output/output_weather_data/PBLH_EMISS/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/PBLH_U10/U10512.png',
             'output/output_weather_data/PBLH_U10/basecase/output_baseline_image.png',
             'output/output_weather_data/PBLH_U10/DivCon/output_DivCon.png',
             'output/output_weather_data/PBLH_U10/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/PSFC_Q2/Q2512.png',
             'output/output_weather_data/PSFC_Q2/basecase/output_baseline_image.png',
             'output/output_weather_data/PSFC_Q2/DivCon/output_DivCon.png',
             'output/output_weather_data/PSFC_Q2/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/SH20_ALBEDO/ALBEDO512.png',
             'output/output_weather_data/SH20_ALBEDO/basecase/output_baseline_image.png',
             'output/output_weather_data/SH20_ALBEDO/DivCon/output_DivCon.png',
             'output/output_weather_data/SH20_ALBEDO/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/Smois_ALBEDO/ALBEDO512.png',
             #'output/output_weather_data/Smois_ALBEDO/basecase/output_baseline_image.png',
             'output/output_weather_data/Smois_ALBEDO/DivCon/output_DivCon.png',
             #'output/output_weather_data/Smois_ALBEDO/parallel/output_Parallel.png',
             '',
             'output/output_weather_data/SMOIS_LWUPB/LWUPB512.png',
             'output/output_weather_data/SMOIS_LWUPB/basecase/output_baseline_image.png',
             'output/output_weather_data/SMOIS_LWUPB/DivCon/output_DivCon.png',
             'output/output_weather_data/SMOIS_LWUPB/parallel/output_Parallel.png',
             ]
#brisq.get_feature('/path')

for image in imageList:
    if image == '':
        print("-----------------------------------------------------------")
    else:
        print("For " + image + " BRISQUE = " + str(brisq.get_score(image)))
