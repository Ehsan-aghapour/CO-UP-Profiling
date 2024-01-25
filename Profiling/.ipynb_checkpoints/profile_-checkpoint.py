import Arduino_read
from config import *
import utils
import time
import threading
import run_graph
import re
import itertools
import parse_perf
import parse_power
from data import *


# +
### This is common function to run a case
## Remember to modify ARMcL code based on your desire
def Profile(_ff=[[[0],[1],[2],[3,6],[4],[5],[6],[7]]],_Num_frames=Num_frames,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt", caching=True, kernel_c=96, _power_profie_mode='whole'):
    #caching=False
    if os.path.isfile(pwr) and os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    utils.ab()
    ff=utils.format_freqs(_ff)
    print(f'\n\nformatted freqs:\n {ff}')
    os.system(f"adb push {cnn_dir}/build/examples/Pipeline/{cnn[graph]} /data/local/ARM-CO-UP/test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(5)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    #Power_monitoring.start()
    rr=f"{cnn_dir}/Run_CO-UP model={graph} --n={_Num_frames} --order={order}  push=0 compile=0 --power_profile_mode={_power_profie_mode} --kernel_c={kernel_c}"
    print(f'run command is {rr}')
    #oo=open(tme,'w+')
    
    # if you want to set freqs with cin:
    run_graph.Run_Graph(ff,rr,tme,True,Power_monitoring)
    
    # if you want to set with run command
    #run_command=rr + f'--freqs=ff[0]'
    #p = subprocess.Popen(run_command.split(),stdout=oo,stderr=oo, stdin=subprocess.PIPE, text=True)
    time.sleep(5)
    #p.wait()
    
    Power_monitoring.do_run = False
    #oo.close()
    
#Profile(caching=False,_Num_frames=10)


# +
 
#### Run a graph for measuring trasfer times of real layers
#### As transfer time of real layers is small, it does not profile power
def Profile_Transfer_Layers(ff=["7-6-4-[3,6]-4-5-6-7"],_Num_frames=Num_frames,order='BBBGBBBB',graph="alex",tme="temp.txt",caching=False):
    if os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    #rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} 1 {_Num_frames} 0 0 100 100 {order} 1 2 4 Alex B B"
    
    rr=f"{cnn_dir}/Run_CO-UP model={graph} --n={_Num_frames} --order={order}  push=0 compile=0 --power_profile_mode=transfers"
    print(f'run command is {rr}')
    #oo=open(tme,'w+')
    run_graph.Run_Graph(ff,rr,tme,True)
    #time.sleep(2)
    #oo.close()


# -

### Run different order configuration to profile transfer time of real layers with min freqs
### It calls profile_Transfer_Layers and Parse_Transfer_Layers functions
def Profile_Transfer_Time(graph="alex"):
    utils.ab()
    os.system(f"adb push {cnn_dir}/build/examples/Pipeline/{cnn[graph]} /data/local/ARM-CO-UP/test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(5)
    global Transfers_df
    NL=NLayers[graph]
    
    C=["G","B","L"]
    combinations = list(itertools.combinations(C, 2))
    orders=[]
    
    for combination in combinations:
        order1=""
        order2=""
        for i in range(NL):
            order1=order1+combination[i%2]
            order2=order2+combination[(i+1)%2]
        orders.append(order1)
        orders.append(order2)
    print(orders)
    for _order in orders:
        print(f'graph:{graph} order:{_order} ')
        Transfers_logs.mkdir(parents=True, exist_ok=True)
        timefile=f'{Transfers_logs}/transfer_{graph}_'+_order+'.txt'
        Profile_Transfer_Layers(["min"],Num_frames,_order,graph,timefile,caching=True)
        time.sleep(2)
        trans_df=parse_perf.Parse_Transfer_Layers(timefile,graph,_order)
        print(trans_df)
        Transfers_df=pd.concat([Transfers_df,trans_df],ignore_index=True)
        #Transfers_df=Transfers_df.append(trans_df,ignore_index=True)
        Transfers_df.to_csv(Transfers_csv,index=False)


#### Run the profile_transfer_time function to profile transfer time of real layers with minimum freqs
def Run_Profile_Transfer_Time():
    global Transfers_df
    for graph in graphs:
        if Transfers_df[Transfers_df["Graph"]==graph].shape[0]==0:
            Profile_Transfer_Time(graph)
if Test==4:
    Run_Profile_Transfer_Time()


### This function is for profiling time and power of tasks in real graphs
### In ARMCL you need to sleep between tasks 
### As transfer time for most cases is less than 1.4 ms (sample interval of power measurement setup)
def Profile_Task_Time(graph):
    global Layers_df
    NL=NLayers[graph]
    orders=["B","G","L"]
    for _order in orders:
        frqss=[]
        NF=NFreqs[_order]
        if _order=="G":
            Nbig=NFreqs["B"]
            for f in range(NF):
                for fbig in range(Nbig):
                    layer_f=[f,fbig]
                    layers_f=NL*[layer_f]
                    frqss.append(layers_f)
        else:
            for f in range(NF):
                layer_f=[f]
                layers_f=NL*[layer_f]
                frqss.append(layers_f)
        print(f'graph:{graph} order:{_order} freqs:{frqss}')
        
        
        order=NL*_order
        Layers_logs.mkdir(parents=True, exist_ok=True)
        pwrfile=f'{Layers_logs}/power_{graph}_'+order+'.csv'
        timefile=f'{Layers_logs}/time_{graph}_'+order+'.txt'
        Profile(frqss,Num_frames,order,graph,pwrfile,timefile,caching=True,_power_profie_mode="layers")
        #time.sleep(10)
        time_df=parse_perf.Parse(timefile,graph,order,frqss)
        power_df=parse_power.Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        #print(time_df)
        #input()
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'],how='outer')
        Layers_df=pd.concat([Layers_df,merged_df], ignore_index=True)
        Layers_df.to_csv(Layers_csv,index=False)


# +
## ARM-COUP

def Profile_Task_Time_NPU(graph):
    global Layers_df
    NL=NLayers[graph]
    C=["N","B"]
    combinations = list(itertools.combinations(C, 2))
    orders=[]
    
    for combination in combinations:
        order1=""
        order2=""
        for i in range(NL):
            order1=order1+combination[i%2]
            order2=order2+combination[(i+1)%2]
        orders.append(order1)
        orders.append(order2)
    print(orders)
    
    for order in orders:
        NL=len(order)
        print(f'graph:{graph} order:{order} ')
        frqss=[]
        NF_Host=NFreqs["B"]
        # for test just run with max freq
        Single_Freq=0
        if Single_Freq==1:
            layer_f=[7]
            layers_f=NL*[layer_f]
            frqss.append(layers_f)
        else:
            for f in range(NF_Host):
                layer_f=[f]
                layers_f=NL*[layer_f]
                frqss.append(layers_f)
        
        
        
        print(frqss)
        
        Layers_logs.mkdir(parents=True, exist_ok=True)
        pwrfile=f'{Layers_logs}/power_{graph}_'+order+'.csv'
        timefile=f'{Layers_logs}/time_{graph}_'+order+'.txt'
        Profile(frqss,Num_frames,order,graph,pwrfile,timefile,caching=True,_power_profie_mode="layers")
        #time.sleep(10)
        time_df=parse_perf.Parse_NPU(timefile,graph,order,frqss)
        if graph=="YOLOv3":
            power_df=parse_power.Parse_Power_NPU_Yolo(pwrfile,graph,order,frqss)
        else:
            power_df=parse_power.Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        #print(time_df)
        #input()
        time_df = time_df[time_df['Component'] == 'N']
        
        power_df = power_df[power_df['Component'] == 'N']
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'],how='outer')
        
        Layers_df=pd.concat([Layers_df,merged_df], ignore_index=True)
        Layers_df.to_csv(Layers_csv,index=False)


# +
#when reading:
#test=pd.read_csv("data_df.csv",index_col=0)
#or you can use df.to_csv with index=False argument

def Profiling_Layers():
    for graph in graphs[::1]:
        if graph=="MobileV1" or graph=="YOLOv3":
            if Layers_df[Layers_df["Graph"]==graph].shape[0]==0:
                Profile_Task_Time(graph)   
            
if Test==5:
    Profiling_Layers()
    
def Profile_Layers_NPU():
    for graph in graphs[::1]:
        if graph=="MobileV1" or graph=="YOLOv3":
        #if graph=="YOLOv3":
            #if Layers_df[(Layers_df["Graph"]==graph) & (Layers_df["Component"]=="N")].shape[0]==0 :
            Profile_Task_Time_NPU(graph)
                #input("berim?")
            
if Test==5:
    Profile_Layers_NPU()
# -




