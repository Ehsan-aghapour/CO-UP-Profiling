
import re
import pandas as pd

### Parse the log results to extract timing paramters for layer times
def Parse(timefile,graph,order,frqss):
    time_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time"])
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    t={}
    ins={}
    outs={}
    trans={}
    parts={} 
    starting_layer={}
    ending_layer={}
    #Adding Graph0 target 1 PE: B Host PE: B num threads: 1 Layers: 0-7
    for l in lines:    
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+num threads: \d+ Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        for match in matches:
            graph_number, start_layer, end_layer = match
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            for layer in t:
                cmp=order[layer]
                freq=freqs[layer]
                Host_freq=-1
                if order[layer]=="G":
                    Host_freq=freq[1]
                
                #input()
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"task","Time":t[layer]}
                if layer in ins:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"in","Time":ins[layer]}
                if layer in outs:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"out","Time":outs[layer]}
                if layer in trans:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"trans","Time":trans[layer]}
                    
                
            t={}
            ins={}
            outs={}
            trans={}
            parts={}
            
        
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            t[k]=value 
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]  
            
            #print(graph_number,input_value,starting_layer)
            #input("ddd")
            #input()
            in_layer=starting_layer[graph_number]
            out_layer=ending_layer[graph_number]
            
            ins[int(in_layer)]=float(input_value)
            
            outs[int(out_layer)]=float(out_value)
            
            
    
    return time_df  


## This is like Parse but for synthetic (test_transfer) graph 
## In this graph task() is comment and just transfer time and power is explored 
def Parse_transfer_graph(timefile,graph,order,frqss):
    with open(timefile) as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    t={}
    ins={}
    outs={}
    trans={}
    parts={}
    prof_trans=[]
    transfer_df_time = pd.DataFrame(columns=['order', 'freq', 'transfer_time', 'RecFreq','SenderFreq'])
    
    for l in lines:     
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            prof_trans=trans
            transfer_df_time.loc[len(transfer_df_time)]={'order':order, 'freq': tuple(freqs), 'transfer_time':trans[1], 'RecFreq':tuple(freqs[1]),'SenderFreq':tuple(freqs[0])}
            t={}
            ins={}
            outs={}
            trans={}
            parts={}
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            t[k]=value
            
        
        match = re.search(r"input time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value =float(match.group(2))
            ins[k]=value
        match = re.search(r"output time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            outs[k]=value
        match = re.search(r"transfer_time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            trans[k]=value
            
        match = re.search(r"total(\d+)_time:(\d+\.\d+)", l)
        if match:
            k = match.group(1)
            value = float(match.group(2))
            parts[k]=value
    return prof_trans,transfer_df_time



# +
#### Function for parse the log output of armcl for extracting transfer time between components
#### Be careful to change ARMCL code so that there is no wait between layers (to be real tranfer time)
#transfer[g][layer][c_dest][c_source][t/pwr]
def Parse_Transfer_Layers(timefile,graph="alex",order="BGBGBGBG"):
    trans_df=pd.DataFrame(columns=["Graph", "Layer", "Dest", "Src", "Time"])
    with open(timefile) as ff:
        lines=ff.readlines()
    #order="BBBGBBBB"
    #freqs=[[0],[1],[2],[3,6],[4],[5],[6],[7]]
    trans={}
    starting_layer={}
    ending_layer={}
    for l in lines:         
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+Layers: (\d+)-(\d+)'
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+num threads: \d+ Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        if matches:
            graph_number, start_layer, end_layer = matches[0]
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
        
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            #print(graph_number,input_value,starting_layer[graph_number])
            #input()
            k=int(ending_layer[graph_number])
            value=float(send_value)
            if k<(len(order)-1):
                trans_df.loc[len(trans_df)]={"Graph":graph, "Layer":k, "Dest":order[k+1], "Src":order[k],"Time":value}
            
            
            
            
    print(trans)   
    return trans_df



#### ARM-COUP
### Parse the log results to extract timing paramters for layer times
def Parse_NPU(timefile,graph,order,frqss):
    time_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time"])
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
     
    T_Layer_Sub_Graph={}
    T_Layer={}
    NPU_input_time={}
    NPU_run_time={}
    NPU_run_time_profile={}
    NPU_output_time={}
    Tasks={}
    Sends={}
    Parts={}
    Ins={}
    Outs={}
        
    starting_layer={}
    ending_layer={}
    for l in lines:    
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+num threads: \d+ Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        for match in matches:
            graph_number, start_layer, end_layer = match
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{T_Layer}')
            for layer in NPU_run_time:
                cmp=order[layer]
                freq=freqs[layer]
                Host_freq=-1
                if order[layer]=="G":
                    Host_freq=freq[1]
                if order[layer]=="N":
                    Host_freq=freq[0]
                
                #input()
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"in","Time":Ins[layer]}
                '''time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"task_total","Time":T_Layer[layer]}'''
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq,
                                           "Layer":layer,"Metric":"task","Time":NPU_run_time_profile[layer]}
                
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"out","Time":Outs[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_load","Time":NPU_input_time[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_run_get","Time":NPU_run_time[layer]}
                
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_fill_tensor","Time":NPU_output_time[layer]}
                    
                
            T_Layer_Sub_Graph={}
            T_Layer={}
            NPU_input_time={} ## loading input 
            NPU_run_time={} ## start and end measuring running the npu + geting(unloading) the output
            NPU_run_time_profile={} #profiled with npu perf tool 
            NPU_output_time={} ## copying output data into output tensor
            Tasks={}
            Sends={}
            Parts={}
            Ins={}
            Outs={}
            
            
            
        
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            T_Layer_Sub_Graph[k]=value 
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)
    
        if matches:
            #print(starting_layer)
            #input()
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            #print(graph_number,input_value,starting_layer[graph_number])
            #input()
            in_layer=int(starting_layer[graph_number])
            out_layer=int(ending_layer[graph_number])
            
            Ins[int(in_layer)]=float(input_value)
            Tasks[int(graph_number)]=float(task_value)
            Sends[int(out_layer)]=float(send_value)
            Outs[int(out_layer)]=float(out_value)
            Parts[int(graph_number)]=float(process_value)
            print(f'Graph: {graph_number} \tstart_layer:{in_layer} \t end_layer:{out_layer} \
            in:{input_value} \ttask:{task_value} \tout:{out_value} \tprocess:{process_value}')
            for i,k in enumerate(range(in_layer,out_layer+1)):
                T_Layer[k]=T_Layer_Sub_Graph[i]
            print(f'T layers are:{T_Layer}')
            
        pattern = fr'NPU_{graph}_(\d+)_(\d+)\s+AVG_input_time:(\d+\.\d+)\s+AVG_run_time:(\d+\.\d+)\s+AVG_prof_run_time:(\d+\.\d+)\s+AVG_output_time:(\d+\.\d+)'
        matches = re.findall(pattern, l)
        if matches:
            values = matches[0]
            # Convert strings to appropriate numeric types (int or float)
            numeric_values = [float(value) if '.' in value else int(value) for value in values]
            first_layer=numeric_values[0]
            last_layer=numeric_values[1]
            
            
            NPU_input_time[first_layer]=numeric_values[2]
            NPU_run_time[first_layer]=numeric_values[3]
            NPU_run_time_profile[first_layer]=numeric_values[4]
            NPU_output_time[last_layer]=numeric_values[5]
            print(f'start_layer:{first_layer} \tend_layer:{last_layer} \tin:{NPU_input_time[first_layer]} \trun:{NPU_run_time[first_layer]}\
            \trun_profile:{NPU_run_time_profile[first_layer]} \tout:{NPU_output_time[last_layer]}')
            
    
    return time_df 



def Parse_total(timefile,graph,order,frqss):
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0   
    freqs=frqss[0]
    input_time=-1
    output_time=-1
    parts=[]
    df_time = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_time', 'task_time','output_time', 'total_time','pipelinetime'])
    for l in lines:        
        if "Profiling these DVFS settings finised" in l:
            print(f'Input_time:{input_time}')
            s=sum(parts)
            print(f'parts:{parts}, sum:{s}')            
            f=freqs
            if type(freqs)==str and freqs[0]=='{':
                f=freqs
            else:
                f=tuple(freqs)
            df_time.loc[len(df_time)]={'graph':graph, 'order':order, 'freq': f, 'input_time':input_time, 'task_time':s-input_time,'output_time':output_time, 'total_time':s, 'pipelinetime':max(parts)}
            input_time=-1
            output_time=-1
            parts=[]
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            print(graph_number,input_value,task_value, send_value, out_value, process_value)
            #input()
            parts.append(float(process_value))
            if int(graph_number)==0:
                input_time=float(input_value)
            output_time=float(out_value)
            
            
    if df_time.shape[0] != len(frqss):
        print(f'Parse performance error: number of runs {df_time.shape[0]} is not equals to number of freqs {len(frqss)}')
        #input()
        #return -1
    return df_time


# -

def Parse_total_pipeline(timefile,graph,order,frqss):
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_index=0   
    freqs=frqss[0]
    input_time=-1
    output_time=-1
    parts=[]
    df_time = pd.DataFrame(columns=['graph', 'order', 'freq', 'pipeline_time', 'FPS','Latency'])
    
    while freq_index < len(frqss):
        freqs=frqss[freq_index]
        t0=0
        t1=0
        t2=0
        tn=0
        f=0
        in0=0
        #L=0
        for l in lines:      
            if "Running Graph with Frequency:" in l:
                f=l.split(':')[-1].strip()
            #if "FPS:" in l:
                #ProfResultult[f]['FPS']=l.split(':')[-1]
            if "Latency:" in l:
                if f !='end' :
                    L=float(l.split(':')[-1].strip())
                    FPS=1000/max(t0,t1,t2,tn)
                    print(f'results for freq:{freqs} = {f}\n')
                    if type(freqs)==str and freqs[0]=='{':
                        freqs=freqs
                    else:
                        freqs=tuple(freqs)
                    df_time.loc[len(df_time)]={'graph':graph, 'order':order, 'freq': freqs, 'pipeline_time':1000/FPS, 'FPS':FPS, 'Latency':L}
                    t0=t1=t2=tn=0
                    freq_index+=1
                
            if "input0_time:" in l:
                in0=float(l.split(':')[-1].strip())
            if "total0_time:" in l:
                t0=float(l.split(':')[-1].strip())
            if "total1_time:" in l:
                t1=float(l.split(':')[-1].strip())
            if "total2_time:" in l:
                t2=float(l.split(':')[-1].strip())
            if "NPU subgraph: 0 --> Cost:" in l:
                tn=float(l.split(':')[-1].strip())
        
        
    #display(df_time)
    #return (L,FPS)
    return df_time


# +
#Parse_total_pipeline(timefile="temp_whole.txt",graph="Alex",order='LLLLBBBB',frqss=[ '{{2-3-4}}' ])
# -


