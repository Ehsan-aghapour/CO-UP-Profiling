# +
from pathlib import Path
import pandas as pd
import os

Test=5

Example="Pipeline"
#Example="EarlyExit"

board="rockpi"
board="khadas"

gpionum=0
Threads_big=0
Threads_little=0
fan_mode="none"
fan_level=0
cnn_dir=""

if board == "rockpi":
    gpionum=157
    Threads_big=2
    Threads_little=4
    fan_mode="none"
    cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ARM-COUP/"

#rockpi
if board =="khadas":
    gpionum=432
    Threads_big=4
    Threads_little=2
    fan_mode="auto"
    fan_level=3
    cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ARM-COUP32bit/"

cnn={
    "Alex":"graph_alexnet_pipeline",
    "Google":"graph_googlenet_pipeline",
    "MobileV1":"graph_mobilenet_pipeline",
    "ResV1_50":"graph_resnet50_pipeline",
    "SqueezeV1":"graph_squeezenet_pipeline",
    "YOLOv3":"graph_yolov3_pipeline",
    "InceptionV3":"graph_inception_v3_pipeline",
    "InceptionV4":"graph_inception_v4_pipeline",
    "InceptionResnetV2":"graph_inception_resnet_v2_pipeline",
    "Res18EE":"graph_resnet18_earlyexit",
    "test_transfer":"graph_test_transfer_pipeline"
}


graphs=["Alex", "Google", "MobileV1", "ResV1_50", "SqueezeV1", "YOLOv3", "InceptionV3", "InceptionV4", "InceptionResnetV2", "Res18EE"]
NLayers={"Alex":8, "Google":11, "MobileV1":14, "ResV1_50":18, "SqueezeV1":10, "YOLOv3":75, "InceptionV3":17, "InceptionV4":23, "InceptionResnetV2":50, "test_transfer":2,
        "Res18EE":8}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
Num_frames=30
Num_Warm_up=3



LittleFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000]
BigFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000, 1608000, 1800000]
GPUFrequencyTable = [200000000, 300000000, 400000000, 600000000, 800000000]

# Create a dictionary to map 'comp' to the appropriate frequency table
comp_to_frequency_table = {
    'L': LittleFrequencyTable,
    'B': BigFrequencyTable,
    'N': BigFrequencyTable,
    'G': GPUFrequencyTable
}

data_path = "Data/"
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
full_data_path = script_dir / data_path
os.makedirs(full_data_path, exist_ok=True)

try:
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    full_data_path = script_dir / data_path
    Layers_csv = full_data_path / 'Layers.csv'
    Transfers_csv = full_data_path / 'Transfers.csv'
    Transfer_Freq_csv = full_data_path / 'Transfer_Freq.csv'
    Transfer_Data_Size_Min_Freq_csv = full_data_path / 'Transfer_Data_Size_Min_Freq.csv'
    Transfer_Data_Size_Max_Freq_csv = full_data_path / 'Transfer_Data_Size_Max_Freq.csv'
    Evaluations_csv = full_data_path / 'Evaluations.csv'
    Layers_logs = full_data_path / 'Layers'
    Transfers_logs = full_data_path / 'Transfers'
    Synthetic_Tranfer_logs = full_data_path / 'Synthetic_Transfers'
    Layers_Percentage_csv = full_data_path / 'Layers_Percentage.csv'
    Layers_With_Percentage_csv = full_data_path / 'Layers_With_Percentage.csv'
    #Freq_Transition_Dealy_csv = full_data_path/'..'/'DVFS-Delay'/'Perf2'/'Data'/'FreqMeasurements2_5.csv'
    Freq_Transition_Dealy_csv = full_data_path/'FreqMeasurements2_5.csv'
    GA_Results_PELSI = full_data_path / 'ga_result.csv'
    GA_Results_LW = full_data_path / 'ga_result_LW.csv'
except:
    Layers_csv=Path(data_path+'Layers.csv').resolve()
    Transfers_csv=Path(data_path+'Transfers.csv').resolve()
    Transfer_Freq_csv=Path(data_path+'Transfer_Freq.csv').resolve()
    Transfer_Data_Size_Min_Freq_csv=Path(data_path+'Transfer_Data_Size_Min_Freq.csv').resolve()
    Transfer_Data_Size_Max_Freq_csv=Path(data_path+'Transfer_Data_Size_Max_Freq.csv').resolve()
    Evaluations_csv=Path(data_path+"Evaluations.csv").resolve()
    Layers_logs=Path(data_path+"./Layers/").resolve()
    Transfers_logs=Path(data_path+"./Transfers/").resolve()
    Synthetic_Tranfer_logs=Path(data_path+"./Synthetic_Transfers/").resolve()
    Layers_Percentage_csv=Path(data_path+"Layers_Percentage.csv").resolve()
    Layers_With_Percentage_csv=Path(data_path+"Layers_With_Percentage.csv").resolve()
    #Freq_Transition_Dealy_csv = Path("../DVFS-Delay/Perf2/Data/FreqMeasurements2_5.csv").resolve()
    Freq_Transition_Dealy_csv = Path(data_path+"FreqMeasurements2_5.csv").resolve()
    GA_Results_PELSI = Path(data_path+'ga_result.csv').resolve()
    GA_Results_LW = Path(data_path+'ga_result_LW.csv').resolve()


'''print(Freq_Transition_Dealy_csv)
print(Transfers_csv)
input('dd')'''

# -


