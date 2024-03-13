# +

graphs=["YOLOv3","MobileV1"]
import sys
import time
import concurrent.futures
target_acc=66
target_graph="YOLOv3"
#target_graph="MobileV1"
#p_dir='Power_Model/'
p_dir='/home/ehsan/UvA/ARMCL/Rock-Pi/LW-ARM-CO-UP/New/Model/test'
p_dir='/home/ehsan/UvA/ARMCL/Rock-Pi/CO-UP-Profiling/Profiling'
sys.path.append(p_dir)
#import P
import predict_cost as P
import utils

'''if sys.argv[1]=="y":
    target_graph="YOLOv3"
if sys.argv[1]=="m":
    target_graph="MobileV1"
target_acc=float(sys.argv[2])'''

print(f'Running Ga for model:{target_graph} for target accuracy:{target_acc}')

# +

import numpy as np


import sys
from tensorflow.keras import layers, models
# +
#P.Load_Data()
model=None




# + endofcell="--"
Target_Acc={"YOLOv3":[66], "MobileV1":[]}

NLayers={"YOLOv3":75, "MobileV1":14}
model_names = { "MobileV1":"Mobile.h5", "YOLOv3":"YOLOv3.h5" }

# -

def decode_gene(v):
    if v==0:
        return "N",[v]
    elif v<6 :
        return "G",[v-1,7]
    elif v<14:
        return "B",[v-6]
    elif v<20:
        return "L",[v-14]
def decoder(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps
# --


model_name=model_names[target_graph]
model=models.load_model(model_name)
problem = MyProblem(target_graph,66)

if False:
    fff=[[2], [0], [0], [0], [2, 7], [1, 7], [3, 7], [1], [2], [4], [2], [7], [3, 7], [3], [0, 7], [7], [2], [5], [2, 7], [3, 7], [1], [4], [0], [7], [3], [3], [1, 7], [0, 7], [2, 7], [3], [1, 7], [2], [4], [4], [7], [0], [7], [2], [0], [0], [5], [0], [4, 7], [1], [1, 7], [3], [3], [0], [4, 7], [1, 7], [0], [7], [3, 7], [4, 7], [3], [0, 7], [5], [2, 7], [5], [7], [1, 7], [2], [7], [2], [0, 7], [4], [7], [4], [4], [5], [3], [2, 7], [2, 7], [1], [2]]
    ordd='BLNBGGGBBBBBGLGBBBGGLLBBLLGGGLGBLBBBBLNLLLGLGBLBGGNBGGBGBGBBGBBBGBBBLBLGGLB'
    P.Inference_Cost(_graph='YOLOv3',_order=ordd,_freq=fff,_debug=True)

if False:
    #inference_time,avg_power,_=P.Inference_Cost(_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x='[ 8 14  0  6  3  2  4  7  8 10  8 13  4 17]'
    x='[ 2  7 16 11  1  5  5  7 18  4  9  4  8 13 16  6  9  6  7 18  2 13 13  4\
       5  1 14 14  1  3  1 13 16  9 18 12  1 15  3  3 11  4  6  1  9 10  2  3\
      10  6  1  5 10  8  7  2  1  1  3  9  0  6 10 14 16 15  8  9 16  3 10  5\
      12 17 18]'
    # Remove the square brackets and split the string by spaces
    values = x.strip('[]').split()

    # Convert the values to integers
    x = np.array([int(value) for value in values])
    config=decoder(x)
    inference_time,avg_power,_=P.Inference_Cost(_debug=False,_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x=np.where(x==0,1,0)
    print(x)
    #model.predict(x.reshape(1,NLayers[graph]))
    print(inference_time,avg_power)
    import math
    np.isnan(inference_time)

.6*32+64

.6*116

3.4*32

632/19

x=np.zeros(75)
x[26:]=1
x_quantization=np.array([x])
x_quantization

model.predict(x_quantization).flatten()


