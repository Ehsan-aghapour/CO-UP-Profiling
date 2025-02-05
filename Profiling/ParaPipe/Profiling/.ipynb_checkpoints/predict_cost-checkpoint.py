from config import *
from utils import *
import utils
import numpy as np
Test=4


def Comp_Cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5, debug=False):
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
    if debug:
        print(f'fn is {fn}')
    
    fc=len(fn)*[None]
    for i in range(len(fc)):
        fc_l=0
        fc_b=0
        fc_g=0
        if cmps[i-1]=='G':
            fc_g=fn[i-1][0]
            fc_b=fn[i-1][1]
        if cmps[i-1]=='B':
            fc_b=fn[i-1][0]
        if cmps[i-1]=='L':
            fc_l=fn[i-1][0]
        
        f={"L":[fc_l], "B":[fc_b], "G":[fc_g,fc_b]}
        fc[i]=f[cmps[i]]
        if debug:
            print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
    if debug:
        print(f'fc is:{fc}')
        print(f'processors:{cmps}')
    tt=0
    ee=0
    tt_nodvfs=0
    ee_nodvfs=0
    
    #comp time
    tfn=Value(g,cmps[0],fn[0],0,'in','Time')
    tfc=Value(g,cmps[0],fc[0],0,'in','Time')
    t=tfc
    if tfc > dvfs_delay:
        t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay  
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
    #comp power
    pfn=Value(g,cmps[0],fn[0],0,'in','Power')
    pfc=Value(g,cmps[0],fc[0],0,'in','Power') 
    e=t*pfc
    if t > dvfs_delay:
        e=dvfs_delay*pfc + (t-dvfs_delay)*pfn
    e_nodvfs= tfn*pfn
    ee+=e
    ee_nodvfs+=e_nodvfs
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} power(next_freq):{pfn} cur_freq:{fc[0]} power(cur_freq):{pfc} energy:{e}')
        
    for i in range(0,len(fn)-1):
        tfn=Value(g,cmps[i+1],fn[i+1],i,'task','Time')
        tfc=Value(g,cmps[i+1],fc[i+1],i,'task','Time')
        t=tfc
        if tfc > dvfs_delay:
            t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
        
        pfn=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
        pfc=Value(g,cmps[i+1],fc[i+1],i,'task','Power') 
        e=t*pfc
        if t > dvfs_delay:
            e=dvfs_delay*pfc + (t-dvfs_delay)*pfn
        e_nodvfs= tfn*pfn
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} power(next_freq):{pfn} cur_freq:{fc[i+1]} power(cur_freq):{pfc} energy:{e}')
        ee+=e
        ee_nodvfs+=e_nodvfs
        
    if debug:
        print(f'time with dvfs delay: {tt}')
        print(f'time without dvfs delay: {tt_nodvfs}')
        print(f'Energy with dvfs delay: {ee/1000.0}')
        print(f'Energy without dvfs delay: {ee_nodvfs/1000.0}')
    return tt,ee/1000.0


parallel_layers = {
    56: {57:59, 58:60}, #if 56 is end point 59 in main stream runs in parallel with 57 in branch (substream) and 60 runs in parallel with 58
    57: {58: 59}, # if 57 is endpint then 59 in main branch runs in parallel with 58 in branch
    64: {65:67, 66: 68},
    65: {66: 67}
}

# Function to find the parallel branches given a partition point
def find_parallel_branches(partition_point):
    if partition_point in parallel_layers:
        return parallel_layers[partition_point]
    elif partition_point - 1 in parallel_layers:
        return parallel_layers[partition_point - 1]
    else:
        return {}


idle_power=3000
parallel_branches=True
skipping_layers={
    57:[56], 58:[56,57], 65:[64], 66:[64,65]
}
if parallel_branches==False:
    skipping_layers={}
## Scale NPU Layer timings:
# Whole NPU task time (max B freq): 337
# NPU_in for layer(0)-->17.3, NPU_out for last layer-->28.3, 
# Last layer run -run_profile (for getting output) --> 69-31=38
#Pure whole --> 337-(17.3+28.3+38)=337-83.6=253.4
# Sum of NPU_run_profile (for all separate layers) --> 1876.7 
# --> T(total)/T(sum) = 253.4/1876.7 = 0.135
## Solving this line: t' = t x (−0.0117n+1.0117)

#for mobile:
# 15.3577 - 4.736 = 10.62 for whole net pure run
# sum --> 53.12
# a= 10.62/53.12 = 0.1999
do_scale_NPU_timing=True
def Comp_Cost_variable_dvfs_delay(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B', debug=False):
    extra_power={}
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
    if debug:
        print(f'fn is {fn}')
    
    fc=len(fn)*[None]
    for i in range(len(fc)):
        fc_l=0
        fc_b=0
        fc_g=0
        if cmps[i-1]=='G':
            fc_g=fn[i-1][0]
            fc_b=fn[i-1][1]
        if cmps[i-1]=='B' or cmps[i-1]=='N':
            fc_b=fn[i-1][0]
        if cmps[i-1]=='L':
            fc_l=fn[i-1][0]
        
        
        f={"L":[fc_l], "B":[fc_b], "G":[fc_g,fc_b]}
        _PE=cmps[i]
        if _PE=='N':
            _PE='B'
        fc[i]=f[_PE]
        if debug:
            print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
    if debug:
        print(f'len fc={len(fc)} and fn={len(fn)}')
        print(f'fc is:{fc}')
        print(f'processors:{cmps}')
    tt=0
    ee=0
    tt_nodvfs=0
    ee_nodvfs=0
    
    #comp time
    tfn=Value(g,cmps[0],fn[0],0,'in','Time',debug=debug)
    tfc=Value(g,cmps[0],fc[0],0,'in','Time',debug=debug)
    t=tfc
    _PE=cmps[0]
    if _PE=='N':
        _PE='B'
    _dvfs_delay=Freq_Transition_Dealy_df[(Freq_Transition_Dealy_df["PE"]==_PE) &\
                                         (Freq_Transition_Dealy_df['Freq']==fc[0][0]) &\
                                         (Freq_Transition_Dealy_df['NextFreq']==fn[0][0])]['AVG'].mean()/1000000.0
    if debug:
        print(f'dvfs delay for inpu: {_dvfs_delay}')
        
    
    #print(g,len(cmps),cmps,cmps[-1],fc[-1],len(fn))
    t_output=Value(g,cmps[-1],fc[-1],len(fn)-2,'out','Time',debug=debug)
    if debug:
        print(f't_output: {t_output}')
    
    #in arm-co-up apply freq is at the end of the task (not after output); so the part of the input
    # that is run with current freq is _dvfs_delay-t_output so we change _dvfs_delay(of this layer) for simplicity
    if _dvfs_delay > t_output:
        _dvfs_delay=_dvfs_delay-t_output
    else:
        _dvfs_delay=0
        
        
    if tfc > _dvfs_delay:
        t=tfn - (_dvfs_delay/tfc)*tfn + _dvfs_delay 
              
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
    # we consider the input power for output power (because output power is not measureble with the current setup)
    p_output=Value(g,cmps[-1],fn[-1],0,'in','Power',debug=debug)
    e_output=t_output*p_output
    if debug:
        print(f't_output: {t_output}   p_output:{p_output}   e_output:{e_output}')
        
    ee+=e_output
    ee_nodvfs+=e_output
    tt+=t_output
    tt_nodvfs+=t_output
    
    #comp power
    pfn=Value(g,cmps[0],fn[0],0,'in','Power',debug=debug)
    pfc=Value(g,cmps[0],fc[0],0,'in','Power',debug=debug) 
    e=t*pfc
    if t > _dvfs_delay:
        e=_dvfs_delay*pfc + (t-_dvfs_delay)*pfn
    e_nodvfs= tfn*pfn
    ee+=e
    ee_nodvfs+=e_nodvfs
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} power(next_freq):{pfn} cur_freq:{fc[0]} power(cur_freq):{pfc} energy:{e}')
        print(f'total e: {ee}- nodvfs delay:{ee_nodvfs}')
    NPU_Data_t=0
    NPU_Data_e=0
    for i in range(0,len(fn)-1):
        
        _PE=cmps[i+1]
        if _PE=='N':
            _PE='B'
        _dvfs_delay=Freq_Transition_Dealy_df[(Freq_Transition_Dealy_df["PE"]==_PE) &\
                                             (Freq_Transition_Dealy_df['Freq']==fc[i+1][0]) &\
                                             (Freq_Transition_Dealy_df['NextFreq']==fn[i+1][0])]['AVG'].mean()/1000000.0
        if i in skipping_layers:
            branch_points=skipping_layers[i]
            for branch_point in branch_points:
                if cmps[branch_point+1]!=cmps[branch_point+2]:
                    #print(parallel_layers,branch_point,i)
                    parallel_layer=parallel_layers[branch_point][i]
                    #layers i+1 and parallel layers will run in parallel
                    if cmps[i+1]!='N' and cmps[parallel_layer+1]!='N':
                        pp=Value(g,cmps[i+1],fn[i+1],i,'task','Power',debug=debug)
                        pp=pp-idle_power
                        extra_power[i]=pp
                        if debug:
                            print(f'layer {i} will run in parallel with {parallel_layer} none on NPU')
                    else:
                        extra_power[i]=0
                        if debug:
                            print(f'layer {i} will run in parallel with {parallel_layer} one on NPU')
                    break
                        
            
        if cmps[i+1]=='N':
            NPU_scale_timing_factor=1
            consequitive_n=count_consecutive_N(cmps[1:],i)
            if g=='YOLOv3':
                NPU_scale_timing_factor=1.0117 - (consequitive_n * 0.0117)
            if g=='MobileV1':
                NPU_scale_timing_factor=1.0615 - (consequitive_n * 0.0615)
            if debug:
                print(f'scaling factor for timing: {NPU_scale_timing_factor}')
            pp=0
            if i in extra_power:                
                pp=extra_power[i]
                if debug:
                    print(f'Layer {i} calc power based on extra_power:{pp}')
            else:
                pp=Value(g,cmps[i+1],fn[i+1],i,'task','Power',debug=debug)
                
                
            t_run=Value(g,cmps[i+1],fn[i+1],i,'run','Time',debug=debug)
            if do_scale_NPU_timing:
                if debug:
                    print(f'Scale down NPU timing')
                t_run=t_run*NPU_scale_timing_factor
            e_run=t_run*pp
            
            #Loading into NPU
            t_load=t_unload=e_load=e_unload=0
            if i==0 or cmps[i]!='N':
                t_load=Value(g,cmps[i+1],fn[i+1],i,'load','Time',debug=debug)
                e_load=t_load*pp
            
            #Unloading from NPU
            if i==len(fn)-2 or cmps[i+2]!='N':
                t_unload=Value(g,cmps[i+1],fn[i+1],i,'unload','Time',debug=debug)
                e_unload=t_unload*pp
            
            
            NPU_Data_e+=(e_load+e_unload)
            
            if debug:
                print(f'NPU calculations\n')
                print(f'Layer:{i}\tfreq:{fn[i+1]}\tload_time{t_load}\trun time:{t_run}\tunload_time:{t_unload}\tpower:{pp}')
            
            
            
            e_npu=e_load+e_run+e_unload
            if debug:
                print(f'NPU load energy:{e_load}\trun energy:{e_run}\tunload energy:{e_unload}\t sum:{e_npu}')
            ee+=e_npu
            ee_nodvfs+=e_npu
            if i not in extra_power:
                NPU_Data_t+=(t_load+t_unload)
                tt+=(t_load+t_run+t_unload)
                tt_nodvfs+=(t_load+t_run+t_unload)
            else:
                if debug:
                    print(f'Time of layer {i}:{(t_load+t_run+t_unload)} is not considered, cause it will run in parallel')
            if debug:
                print(f'total e: {ee}- nodvfs delay:{ee_nodvfs}\ttotal_time:{tt}')
            
        else:
            tfn=Value(g,cmps[i+1],fn[i+1],i,'task','Time',debug=debug)
            tfc=Value(g,cmps[i+1],fc[i+1],i,'task','Time',debug=debug)
            t=tfc

            if debug:
                print(f'dvfs delay for layer{i}: {_dvfs_delay}')
            if tfc > _dvfs_delay:
                t=tfn - (_dvfs_delay/tfc)*tfn + _dvfs_delay
            if debug:
                print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
            if i not in extra_power:
                tt+=t
                tt_nodvfs+=tfn
            else:
                if debug:
                    print(f'Layer {i} do not consider its time {t} (t_nodvfs:{tfn})')
            pfn=pfc=0
            if i in extra_power:
                pfn=pfc=extra_power[i]
                if debug:
                    print(f'layer {i} calc based on extra power {pfc}')
            else:
                pfn=Value(g,cmps[i+1],fn[i+1],i,'task','Power',debug=debug)
                pfc=Value(g,cmps[i+1],fc[i+1],i,'task','Power',debug=debug)
                
            e=t*pfc
            if t > _dvfs_delay:
                e=_dvfs_delay*pfc + (t-_dvfs_delay)*pfn
            e_nodvfs= tfn*pfn

            ee+=e
            ee_nodvfs+=e_nodvfs
            if debug:
                print(f'layer:{i}, next_freq:{fn[i+1]} power(next_freq):{pfn} cur_freq:{fc[i+1]} power(cur_freq):{pfc} energy:{e}')
                print(f'total e: {ee}- nodvfs delay:{ee_nodvfs}\ttotal_time:{tt}')
            if np.isnan(tt):
                print(f'\n\n\n************************{cmps[i+1]}\n')
        
    if debug:
        print(f'\ntime with dvfs delay: {tt}')
        print(f'time without dvfs delay: {tt_nodvfs}')
        print(f'Energy with dvfs delay: {ee/1000.0}')
        print(f'Energy without dvfs delay: {ee_nodvfs/1000.0}')
        print(f'NPU total loading and unloading times:{NPU_Data_t}')
        print(f'NPU total loading unloading energy:{NPU_Data_e/1000.0}')
        print(f'extra_power:{extra_power}')
    #print(f'extra_power:{extra_power}')
    return tt,ee/1000.0
# -

'''import data
data.Load_Data()
#Value('google','L',[0],[3],'task','Time')
Value('Alex','B',[7],7,'task','Time')'''

if Test==3:
    _g='YOLOv3'

    Comp_Cost_variable_dvfs_delay(g=_g,cmps='N'*NLayers[_g],fn=[[7]]*NLayers[_g],debug=False)
    Comp_Cost_variable_dvfs_delay(g=_g,cmps='N'*57+'BNBB'+'BBBB'+'NNBBB'+'N'*5,fn=[[7]]*NLayers[_g],debug=True)


def Transfer_Info(p1='B',p2='G',f1=[4],f2=[3,4],_debug=False):
    global Transfer_Freq_df
    f1=[int(i) for i in f1]
    f2=[int(i) for i in f2]
    if p1=='N':
        p1='B'
    if p2=='N':
        p2='B'
        
    order=p1+p2
    if order=='GL':
        order='GB'
        f2[0]=f1[1]
        p2='B'
    if p1=='G':
        f1[0]=0
    else:
        f1=[0]
    if p2=='G':
        f2[0]=0
    if order=='BG':
        f1[0]=f2[1]
    if order=='GB':
        f2[0]=f1[1]
    freqs=tuple([tuple(f1),tuple(f2)])
    row=Transfer_Freq_df[ (Transfer_Freq_df['freq']==str(freqs)) & (Transfer_Freq_df['order']==order)]
    if _debug:
        print(freqs)
        print(row)
    power=row['transfer_power'].iloc[0]
    coef_t=row['time_ratio'].iloc[0]  
    return power,coef_t
if Test==2:
    a,b=Transfer_Info('G','B',[2.0, 7.0],[7.0])
    Transfer_Freq_df


def Comm_Cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B', debug=False):
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
    if debug:
        print(f'fn is {fn}')
    
    fc=len(fn)*[None]
    for i in range(len(fc)):
        fc_l=0
        fc_b=0
        fc_g=0
        if cmps[i-1]=='G':
            fc_g=fn[i-1][0]
            fc_b=fn[i-1][1]
        if cmps[i-1]=='B' or cmps[i-1]=='N':
            fc_b=fn[i-1][0]
        if cmps[i-1]=='L':
            fc_l=fn[i-1][0]
        
        f={"L":[fc_l], "B":[fc_b], "G":[fc_g,fc_b], "N":[fc_b]}
        fc[i]=f[cmps[i]]
        if debug:
            print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
    if debug:
        print(f'fc is:{fc}')
        print(f'processors:{cmps}')
        
    transfer_t=0
    transfer_e=0
    # Layers are indexed from 1 (because first index in cmps, fn, and fc is for input)
    # We start from layer=2 because comparing with previous layer
    if debug:
        print(f'cmps: {cmps}')
    #make it compatibel with dac profiled data
    # that layers are from 0 to N-1 not from 1 to N
    #for i in range(2,len(fn)):
    for i in range(1,len(fn)-1):
        if cmps[i]!=cmps[i-1]: 
            if debug:
                print(f'transfer happen between {cmps[i-1]} and {cmps[i]}')
            #transfer_time=transfer_times[g][i][cmps[i]][cmps[i-1]]
            src=cmps[i-1]
            dst=cmps[i]
            f_src=fc[i-1]
            f_dst=fc[i]
            if src=='N':
                src='B'
                if dst=='B':
                    dst='L'
                    if f_dst[0] > (NFreqs['L']-1):
                        f_dst[0]=(NFreqs['L']-1)
                
            if dst=='N':
                if src=='B':
                    dst='L'
                    if f_dst[0] > (NFreqs['L']-1):
                        f_dst[0] = (NFreqs['L']-1)
                else:
                    dst='B'
            if debug:
                print(f'now the src and dst are:{src}->{dst} for layer {i-1} in graph {g}')
                
            if debug:
                print(Transfers_df[(Transfers_df["Graph"]==g) &
                                       (Transfers_df["Layer"]==i-1) &
                                       (Transfers_df["Dest"]==dst) &
                                       (Transfers_df["Src"]==src)]["Time"])
                
                
            transfer_time=Transfers_df[(Transfers_df["Graph"]==g) &
                                       (Transfers_df["Layer"]==i-1) &
                                       (Transfers_df["Dest"]==dst) &
                                       (Transfers_df["Src"]==src)]["Time"].iloc[0]
            if debug:
                print(f'{fc[i-1]}--{fc[i]}')
            transfer_power,time_ratio=Transfer_Info(p1=src,p2=dst,f1=fc[i-1],f2=fc[i],_debug=debug)
        
            scaled_time=transfer_time * time_ratio
            transfer_energy=scaled_time * transfer_power
            
            transfer_t+=scaled_time
            transfer_e+=transfer_energy
            if debug:
                print(f"Transfer between layer {i-1} and {i} (inexed start with 1)")
                print(f'transfer_time: {transfer_time}, time_ratio:{time_ratio}, scaled_time:{scaled_time}')
                print(f'transfer_power:{transfer_power}, transfer_energy:{transfer_energy}')
                print(f'total time:{transfer_t}')
                print(f'total energy:{transfer_e/1000.0}')
    return transfer_t, transfer_e/1000.0


# +
#Transfers_df.query("Graph=='YOLOv3' and Src=='B' and Dest=='G' and Layer==4")

# +
def Inference_Cost(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False):
    #print(_graph,_freq,_order)
    fff=[]
    if _freq=="min" or _freq=="{{min}}" or _freq[0]=="{{min}}":
        for c in _order:
            if c=='G' or c=='N':
                fff.append([0,0])
            else:
                fff.append([0])
        _freq=fff
    if _freq=="max" or _freq=="{{max}}" or _freq[0]=="{{max}}":
        for c in _order:
            if c=='G':
                fff.append([4,7])
            if c=='N':
                fff.append([7,7])
            if c=='B':
                fff.append([7])
            if c=='L':
                fff.append([5])
        _freq=fff
    #print(_freq)
    
    _dvfs_delay='variable' # DAC is implemented in variable version of comp_cost_...
    total_time=0
    total_energy=0
    if _dvfs_delay=="variable":
        t_cmp,e_cmp=Comp_Cost_variable_dvfs_delay(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    else:
        t_cmp,e_cmp=Comp_Cost(g=_graph,fn=_freq,cmps=_order,dvfs_delay=_dvfs_delay, debug=_debug)
    t_cmu,e_cmu=Comm_Cost(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    total_time=t_cmp + t_cmu
    total_energy=e_cmp + e_cmu
    average_power=1000*(total_energy/total_time)
    return total_time,average_power,total_energy
def Inference_Cost_0(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False):
    _dvfs_delay='variable' # DAC is implemented in variable version of comp_cost_...
    total_time=0
    total_energy=0
    if _dvfs_delay=="variable":
        t_cmp,e_cmp=Comp_Cost_variable_dvfs_delay(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    else:
        t_cmp,e_cmp=Comp_Cost(g=_graph,fn=_freq,cmps=_order,dvfs_delay=_dvfs_delay, debug=_debug)
    t_cmu,e_cmu=Comm_Cost(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    total_time=t_cmp + t_cmu
    total_energy=e_cmp + e_cmu
    average_power=1000*(total_energy/total_time)
    return total_time,average_power,total_energy
if Test==2:
    print(Inference_Cost(_dvfs_delay=0))
    print(Inference_Cost(_dvfs_delay=3.5))
    print(Inference_Cost(_dvfs_delay='variable'))
    
if Test==3:
    _g='YOLOv3'

    print(Inference_Cost(_graph=_g,_order='N'*NLayers[_g],_freq=[[7]]*NLayers[_g],_debug=True))
    print(Inference_Cost(_graph=_g,_order='N'*57+'BNBB'+'BBBB'+'NNBBB'+'N'*5,_freq=[[7]]*NLayers[_g],_debug=False))
    
    _g='MobileV1'
    print(Inference_Cost(_graph=_g,_order='N'*NLayers[_g],_freq=[[7]]*NLayers[_g],_debug=True))
    
if False:
    fff=[[2], [0], [0], [0], [2, 7], [1, 7], [3, 7], [1], [2], [4], [2], [7], [3, 7], [3], [0, 7], [7], [2], [5], [2, 7], [3, 7], [1], [4], [0], [7], [3], [3], [1, 7], [0, 7], [2, 7], [3], [1, 7], [2], [4], [4], [7], [0], [7], [2], [0], [0], [5], [0], [4, 7], [1], [1, 7], [3], [3], [0], [4, 7], [1, 7], [0], [7], [3, 7], [4, 7], [3], [0, 7], [5], [2, 7], [5], [7], [1, 7], [2], [7], [2], [0, 7], [4], [7], [4], [4], [5], [3], [2, 7], [2, 7], [1], [2]]
    ordd='BLNBGGGBBBBBGLGBBBGGLLBBLLGGGLGBLBBBBLNLLLGLGBLBGGNBGGBGBGBBGBBBGBBBLBLGGLB'
    r=Inference_Cost(_graph='YOLOv3',_order=ordd,_freq=fff,_debug=False)
    print(r)
    
    ordd='BLLLBLGLLLGNLLLBLLLLBLLBLBLLLLBLLBLLLBLBLLGLLLBBGLLBBLBLNNLLLBBBLGLLBLLLBLN'
    fff=[[5], [5], [5], [0], [1], [3], [3, 7], [0], [1], [4], [2, 7], [7], [5], [5], [5], [0], [1], [3], [5], [0], [0], [2], [1], [4], [3], [2], [5], [1], [0], [5], [5], [2], [3], [7], [4], [4], [1], [1], [4], [5], [4], [3], [1, 7], [2], [0], [5], [4], [2], [0, 7], [2], [2], [0], [0], [5], [2], [3], [7], [7], [0], [5], [3], [2], [0], [6], [1], [3, 7], [3], [2], [2], [2], [4], [4], [6], [2], [7]]
    r=Inference_Cost(_graph='YOLOv3',_order=ordd,_freq=fff,_debug=False)
    print(r)
#Value('YOLOv3', 'N', [7], 57, 'task', 'Power', True)
#[57] in [57,58] --> this gives false in python
#layer 59 running with NPU with min freq time is empty fill it manually

dbg=False
ordd="NNNNNNNNNNNNNN"
fff=[[5], [5], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
r=Inference_Cost(_graph="MobileV1",_order=ordd,_freq=fff,_debug=dbg)
print(r)

ordd="LLNNNNNNNNNNNN"
fff=[[5], [5], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
r=Inference_Cost(_graph="MobileV1",_order=ordd,_freq=fff,_debug=dbg)
print(r)

ordd="GLLLLNNNNNNNNN"
fff=[[4,7],[5],[5],[5],[5],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
r=Inference_Cost(_graph="MobileV1",_order=ordd,_freq=fff,_debug=dbg)
print(r)

ordd="GLLLLLLNNNBNNN"
fff=[[4,7],[5],[5],[5],[5],[5],[5],[7],[7],[7],[7],[7],[7],[7]]
r=Inference_Cost(_graph="MobileV1",_order=ordd,_freq=fff,_debug=dbg)
print(r)

ordd="LLLLLLLLLLLLLL"
fff=[[5], [5], [5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]]
r=Inference_Cost(_graph="MobileV1",_order=ordd,_freq=fff,_debug=dbg)
print(r)
# -


if False:
    k=49
    j=0
    ordd = 'L' * 75
    if k:
        ordd = ordd[:-k] + 'N' * k
    ordd = 'B'*j + ordd[j:]
    print(ordd)
    #ordd=ordd[:57]+'LL'+ordd[59:]
    #print(ordd)
    r=Inference_Cost(_graph='YOLOv3',_order=ordd,_freq='min',_debug=False)
    print(r)


# +
import concurrent.futures
import multiprocessing
import time

def Inference_Cost_wrapper(params):
    _graph, _order, _freq, _debug = params
    result = Inference_Cost(_graph=_graph, _order=_order, _freq=_freq, _debug=_debug)
    return result

if False:
    # Define your parameters for the function calls
    parameters = [('YOLOv3', 'N'*57+'BNBB'+'BBBB'+'NNBBB'+'N'*5, [[7]]*NLayers['YOLOv3'], False)]

    # Measure the execution time of one call
    start_time = time.time()
    result_single = Inference_Cost_wrapper(parameters[0])
    end_time = time.time()
    print(result_single)
    print("Execution time of one call:", end_time - start_time)
    
    
    # Run 10 calls in parallel
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the function to the parameters and execute in parallel
        results = list(executor.map(Inference_Cost_wrapper, parameters * 200))
    end_time = time.time()
    print("Execution time of one call:", end_time - start_time)
    print(results)
    
    
    # Run 10 calls in parallel
    start_time = time.time()
    # Create a multiprocessing Pool with the number of available CPU cores
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Use the Pool.map function to map the compute_cost function to each config
    results = pool.map(Inference_Cost_wrapper, parameters * 200)
    
    # Close the pool to release resources
    pool.close()
    pool.join()
    end_time = time.time()
    print("Execution time of one call:", end_time - start_time)
    print(results)

# +

if Test==3:
    for g in graphs:
        #g='Alex'
        for cmp in NFreqs:
        #cmp='G'
            #print(Inference_Cost(_dvfs_delay=0,_debug=True))
            fs=[[NFreqs[cmp]-1]]*NLayers[g]
            if cmp=='G':
                fs=[[NFreqs[cmp]-1,NFreqs['B']-1]]*NLayers[g]
            t,p,e=Inference_Cost(_graph=g,_freq=fs,_order=NLayers[g]*cmp,_dvfs_delay='variable',_debug=False)
            print(f'graph:{g:<12} comp:{cmp}   time:{t:<8.2f}   PE:{1000/e:<8.3f}   energy:{e:.2f}')

#considering idle energy
if Test==3:
    idle_power=2750
    target_latency=200
    for fi in range(NFreqs[cmp]):
        g='Alex'
        cmp='L'
        fs=((5,), (5,), (1,), (2,), (3,), (2,), (2,), (5,))
        fs=[[NFreqs[cmp]-1]]*NLayers[g]
        fs=[[fi]]*NLayers[g]
        print(f'freq:{fs}')
        t,p,e=Inference_Cost(_graph=g,_freq=fs,_order=NLayers[g]*cmp,_dvfs_delay='variable',_debug=False)
        idle_energy=(target_latency-t)*idle_power/1000
        interval_energy=idle_energy+e
        print(f'graph:{g:<8} comp:{cmp}  time:{t:<8.1f}   PE:{1000/e:<5.3f}   energy:{e:.0f}    idle_energy={idle_energy:.1f}   interval_energy={interval_energy:.0f}')
# -

#exhaustive search for best freq combination considering idle energy
if Test==3:
    idle_power=2750
    target_latency=200
    g='Alex'
    cmp='L'


    import itertools
    # Define the possible values for each layer
    layers_values = [(0, 1, 2, 3, 4, 5)] * 8
    # Generate all possible combinations
    all_combinations = [tuple((value,) for value in combination) for combination in itertools.product(*layers_values)]
    #print(all_combinations[0])

    min_e=100000
    f_min=None
    for ii,fs in enumerate(all_combinations):
        t,p,e=Inference_Cost(_graph=g,_freq=fs,_order=NLayers[g]*cmp,_dvfs_delay='variable',_debug=False)
        idle_energy=(target_latency-t)*idle_power/1000
        interval_energy=idle_energy+e
        #print(f'graph:{g:<8} comp:{cmp}  time:{t:<8.1f}   PE:{1000/e:<5.3f}   energy:{e:.0f}    idle_energy={idle_energy:.1f}   interval_energy={interval_energy:.0f}')
        if interval_energy<min_e:
            min_e=interval_energy
            f_min=fs
        if(ii%100==0):
            print(ii)

# +
idle_power=2750
target_latency=200
g='Alex'
cmp='L'

# Example evaluation function
def eval_function(member):
    #print(member)
    #input()
    # Example: Sum of all values in the member tuple
    #return sum(member)
    t,p,e=Inference_Cost(_graph=g,_freq=member,_order=NLayers[g]*cmp,_dvfs_delay='variable',_debug=False)
    idle_energy=(target_latency-t)*idle_power/1000
    interval_energy=idle_energy+e
    return -interval_energy
# +
def Fill_prediction(_FileName, dvfs_delay):
    if _FileName.exists():
        Evals_df=pd.read_csv(_FileName).drop_duplicates()
    else:
        print("Ga result file is not existed")
        return
    
    if 'input_e' in Evals_df:
        Evals_df['total_e']=Evals_df['input_e']+Evals_df['task_e']+Evals_df['output_e']
    cases=Evals_df.shape[0]
    print(f'There are {cases}')
    
    Regenerate_Prediction=False
    Regenerate_Errors=False
    
    def prediction(row):
        #print(row)
        graph=row['graph']
        print([row['freq']])
        freq=utils.format_to_list([row['freq']])[0]
        #freq=utils.format_freqs([row['freq']])
        order=row['order']
        #print(graph,freq,order,dvfs_delay)
        return Inference_Cost(_graph=graph,_freq=freq,_order=order,_dvfs_delay=dvfs_delay, _debug=False)
    
    
    '''
    if 'Predicted_Time' not in Evals_df:
        Evals_df[['Predicted_Time','Predicted_Power','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    if 'Predicted_Time' in Evals_df:
        if pd.isna(Evals_df['Predicted_Time']).any() or Regenerate_Prediction:
            Evals_df[['Predicted_Time','Predicted_Power','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    '''
    if 'Predicted_Time' not in Evals_df:
        Evals_df['Predicted_Time']=None
        Evals_df['Predicted_Power']=None
        Evals_df['Predicted_Energy']=None
    if 'Power' not in Evals_df:
        Evals_df['Power']=None
    try:
        for index, row in Evals_df.iterrows():
            try:
                if pd.isna(row['Predicted_Time']) or Regenerate_Prediction:
                    row[['Predicted_Time', 'Predicted_Power', 'Predicted_Energy']] = prediction(row)
                if pd.isna(row['total_e']):
                    row['total_e']=row['input_e']+row['task_e']+row['output_e']
                if pd.isna(row['Power']):
                    row['Power']=row['total_e']/row['total_time']
                Evals_df.iloc[index]=row
            except Exception as e:
                print(f"Error in row {index}: {e}")
                continue
    except Exception as e:
        print(f"Error: {e}")
    #display(Evals_df)
    Evals_df['Predicted_Power']=Evals_df['Predicted_Power']/1000
    def calc_EE(row):
        Measured=1000.0/row['total_e']
        Pred=1000.0/row['Predicted_Energy']
        Err=(Pred-Measured)/Measured
        return 100.0*Err
    
    def calc_Power(row):
        measured=row['total_e']/row['total_time']
        pred=row['Predicted_Energy']/row['Predicted_Time']
        Err=100*abs(pred-measured)/measured
        return Err
    
    def calc_Power_MAE(row):
        measured=row['total_e']/row['total_time']
        pred=row['Predicted_Energy']/row['Predicted_Time']
        Err=abs(pred-measured)
        return Err
    
    def calc_FPS(row):
        measured=1000/row['total_time']
        pred=1000/row['Predicted_Time']
        Err=100.0*(pred-measured)/measured
        return Err
    
    Evals_df = Evals_df.loc[Evals_df['total_e'].notna()]
    new_file=_FileName.with_name(_FileName.name.replace(".csv", "_prediction.csv"))
    Evals_df.to_csv(new_file)
    #display(Evals_df)
    if 'Error_Time' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Time']=Evals_df.apply(lambda x:abs(100*(x['Predicted_Time']-x['total_time'])/x['total_time']),axis=1)
    if 'Error_Energy' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Energy']=Evals_df.apply(lambda x:abs(100*(x['Predicted_Energy']-x['total_e'])/x['total_e']),axis=1)
    if 'Error_EE' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_EE']=Evals_df.apply(calc_EE,axis=1)
    #Evals_df['Error_Power']=Evals_df.apply(lambda x:100*abs( (x['Predicted_Energy']/x['Predicted_Time']) - (x['total_e']/x['total_time']) /(x['total_e']/x['total_time']) ),axis=1)
    if 'Error_Power' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Power']=Evals_df.apply(calc_Power,axis=1)
    if 'Error_FPS' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_FPS']=Evals_df.apply(calc_FPS,axis=1)
        
    Evals_df['MAE_Time']=Evals_df.apply(lambda x:abs(x['Predicted_Time']-x['total_time']),axis=1)
    Evals_df['MAE_Power']=Evals_df.apply(calc_Power_MAE,axis=1)
    
    new_file=_FileName.with_name(_FileName.name.replace(".csv", "_prediction.csv"))
    Evals_df.to_csv(new_file)

if Test==2:
    for g in graphs:
        fname=Path('Evaluations_'+g+'.csv')
        Fill_prediction(fname, 'variable')

#Fill_prediction(Path("yolo.csv"),'variable')
#Fill_prediction(Path("Yolov3_analyze_layers.csv"),'variable')
if Test==3:
    Fill_prediction(Path('Test_Model_Evaluations_YOLOv3.csv'),'variable')


# -

def produce_desing_points(g='YOLOv3'):
    #EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
    EvalFile=Path("Yolov3_analyze_layers.csv").resolve()
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','output_time','total_time', 'input_power','task_power'])    
    cases=Evaluations_df[Evaluations_df['graph']==g].shape[0]
    print(f'There are {cases} existed for graph {g}')
    #num_evals=max(0,num_evals-cases)
    #num_orders=math.ceil(num_evals/num_freqs)
    
    
    _n=NLayers[g]
    base_string='B'*_n
    
    
    #For Figure6
    '''orders=[]
    for i in range(len(base_string)):
        order=base_string[:i]+'N'+base_string[i+1:]
        orders.append(order)'''
        
    # For Figure 7
    
    orders=[]
    '''for i in range(76, 0, -1):
        binary_sequence = np.zeros(75, dtype=str)
        if i < 76:
            binary_sequence[i-1:] = 'N'

        # Convert 'L' for remaining elements
        binary_sequence[:i-1] = 'L'
        binary_sequence = ''.join(binary_sequence)
        print(binary_sequence)
        orders.append(binary_sequence)'''
        
        
    for i in range(0, 76, 1):
        binary_sequence = np.zeros(75, dtype=str)
        binary_sequence[0:]='L'
        if i > 0:
            binary_sequence[0:i] = 'N'

        
        binary_sequence = ''.join(binary_sequence)
        print(binary_sequence)
        orders.append(binary_sequence)
    
    
        
        
    print(orders)
    print(len(orders))
    
    
               
    #_fs=str(tuple([7]*_n))
    fs={}
    for order in orders:        
        fs[order]=['{{max}}']
            
    
    for order in fs:
        for f in fs[order]:
            row=Evaluations_df[(Evaluations_df['order']==order) & (Evaluations_df['freq']==str(f)) & (Evaluations_df['graph']==g)]
            if row.shape[0]==0:
                Evaluations_df.loc[len(Evaluations_df)]={"graph":g,"order":order,"freq":f}
            
    Evaluations_df.to_csv(EvalFile,index=False)
#produce_desing_points(g='YOLOv3')

#def Anlze_Error():
#if True:
if Test==1:
    for g in graphs:
        print(f'Graph: {g}')
        if not Path('Evaluations_'+g+'_prediction.csv').exists():
            continue
        Evals_df=pd.read_csv('Evaluations_'+g+'_prediction.csv')
        #error_time = abs(100.0*(Evals_df['Predicted_Time'] - Evals_df['total_time'])/Evals_df['total_time'])
        #print(abs(error_time).describe())
        #error_energy = abs(100.0*((1000.0/Evals_df['Predicted_Energy']) - (1000.0/Evals_df['total_e']))/(1000.0/Evals_df['total_e']))
        #error_energy=Evals_df['Error_Time']
        error_energy=abs(Evals_df['Error_Energy'])
        #error_energy=Evals_df['Error_EE']
        #plt.hist(error_energy, bins=50, density=True)
        print(error_energy.describe())
        # Add normal curve
        mu, std = norm.fit(error_energy)
        x = np.linspace(-30, 30, 200)

        y = norm.pdf(x, mu, std)
        plt.plot(x, y, label=g)
        plt.xlabel('Error%')
        plt.ylabel('Pdf')
        
    plt.legend()
    plt.show()


# +
def prediction(File,row_num,dvfs_delay):
        _FileName=Path(File)
        if _FileName.exists():
            Evals_df=pd.read_csv(_FileName).drop_duplicates()
        else:
            print("Ga result file is not existed")
            return

        cases=Evals_df.shape[0]
        print(f'There are {cases}')
        #print(row)
        #Evals_df=Evals_df.sort_values('Error_Time',ascending=False)
        row=Evals_df.iloc[row_num]
        graph=row['graph']
        freq=utils.format_to_list([row['freq']])
        order=row['order']
        #print(graph,freq,order,dvfs_delay)
        t,e=Inference_Cost(_graph=graph,_freq=freq[0],_order=order,_dvfs_delay=dvfs_delay, _debug=True)
        print(f'total_time:{t}, total_e:{e}')
        run=False
        if run:
            Real_Evaluation(g=graph,_ord=order,_fs=freq)
            
        return t,e
    

if Test==2:
    g='google'
    prediction('Evaluations_'+g+'_prediction.csv',0,'variable')

# +
import random

# Define the possible values for each layer
layers_values = [(0, 1, 2, 3, 4, 5)] * 8

# Generate initial population
def generate_population(population_size):
    #population = [tuple((value,) for value in random.choice(layers_values)) for _ in range(population_size)]
    population = random.choices(all_combinations, k=population_size)
    return population

# Evaluate the goodness of each member in the population
def evaluate_population(population):
    fitness_scores = [eval_function(member) for member in population]
    return fitness_scores


# Select parents for mating using tournament selection
def select_parents(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected_parents.append(winner)
    return selected_parents

# Crossover operator: Single point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation operator: Random mutation
def mutate(member, mutation_rate):
    mutated_member = list(member)
    for i in range(len(mutated_member)):
        if random.random() < mutation_rate:
            mutated_member[i] = tuple(random.choice(layers_values[i]) for _ in range(1))
    return tuple(mutated_member)

# Genetic Algorithm
def genetic_algorithm(eval_function, population_size, num_generations, tournament_size, mutation_rate):
    # Generate initial population
    population = generate_population(population_size)
    
    for generation in range(num_generations):
        # Evaluate the population
        fitness_scores = evaluate_population(population)
        
        # Select parents for mating
        parents = select_parents(population, fitness_scores, tournament_size)
        #print(f'selected parents:{parents}')
        # Create next generation through crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            #print(f'p1:{parent1}, p2:{parent2}')
            child1, child2 = crossover(parent1, parent2)
            #print(f'c1:{child1}, c2:{child2}')
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            #print(f'mc1:{child1}, mc2:{child2}')
            next_generation.extend([child1, child2])
            #input()
        
        # Replace the current population with the next generation
        population = next_generation
        
        # Evaluate the final population
        fitness_scores = evaluate_population(population)
        
        # Find the best member and its fitness
        best_fitness = max(fitness_scores)
        best_member = population[fitness_scores.index(best_fitness)]
        
        # Print progress
        percentage_complete = (generation + 1) * 100 / num_generations
        print(f"Generation {generation+1}/{num_generations} complete. Best member: {best_member} --> Best fitness so far: {best_fitness}. Progress: {percentage_complete}%")
    
    # Evaluate the final population
    fitness_scores = evaluate_population(population)
    
    # Find the best member
    best_member = population[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)
    
    return best_member, best_fitness

if Test==5:
    import itertools
    # Define the possible values for each layer
    layers_values = [(0, 1, 2, 3, 4, 5)] * 8
    # Generate all possible combinations
    all_combinations = [tuple((value,) for value in combination) for combination in itertools.product(*layers_values)]
    # Example usage
    best_member, best_fitness = genetic_algorithm(eval_function, population_size=200, num_generations=200, tournament_size=5, mutation_rate=0.2)

    print("Best member:", best_member)
    print("Best fitness:", best_fitness)


# -

def Transfer_Cost(_order,fs,_kernel_c=96*100):   
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    
    Synthetic_Tranfer_logs.mkdir(parents=True, exist_ok=True)
    pwrfile=f'{Synthetic_Tranfer_logs}/power_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.csv'
    timefile=f'{Synthetic_Tranfer_logs}/time_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.txt'
    
    Profile(_ff=fs, _Num_frames=Num_frames, order=_order, graph=g, pwr=pwrfile, tme=timefile,caching=False,kernel_c=_kernel_c)
    
    trans,transfer_df_time=Parse_transfer_graph(timefile=timefile, graph=g, order=_order, frqss=fs)

    trans_pwr,trans_pwr_df=Parse_Power_Transfer_graph(file_name=pwrfile,graph=g,order=_order,frqss=fs)
    
    return trans,trans_pwr,trans_pwr_df,transfer_df_time


