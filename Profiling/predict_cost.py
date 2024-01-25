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
    tfn=Value(g,cmps[0],fn[0],0,'in','Time')
    tfc=Value(g,cmps[0],fc[0],0,'in','Time')
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
    t_output=Value(g,cmps[-1],fc[-1],len(fn)-2,'out','Time')
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
    p_output=Value(g,cmps[-1],fn[-1],0,'in','Power')
    e_output=t_output*p_output
    if debug:
        print(f't_output: {t_output}   p_output:{p_output}   e_output:{e_output}')
        
    ee+=e_output
    ee_nodvfs+=e_output
    tt+=t_output
    tt_nodvfs+=t_output
    
    #comp power
    pfn=Value(g,cmps[0],fn[0],0,'in','Power')
    pfc=Value(g,cmps[0],fc[0],0,'in','Power') 
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
                        pp=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
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
                
            pp=0
            if i in extra_power:                
                pp=extra_power[i]
                if debug:
                    print(f'Layer {i} calc power based on extra_power:{pp}')
            else:
                pp=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
                
                
            t_run=Value(g,cmps[i+1],fn[i+1],i,'run','Time')
            if do_scale_NPU_timing:
                if debug:
                    print(f'Scale down NPU timing')
                t_run=t_run*NPU_scale_timing_factor
            e_run=t_run*pp
            
            #Loading into NPU
            t_load=t_unload=e_load=e_unload=0
            if i==0 or cmps[i]!='N':
                t_load=Value(g,cmps[i+1],fn[i+1],i,'load','Time')
                e_load=t_load*pp
            
            #Unloading from NPU
            if i==len(fn)-2 or cmps[i+2]!='N':
                t_unload=Value(g,cmps[i+1],fn[i+1],i,'unload','Time')
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
            tfn=Value(g,cmps[i+1],fn[i+1],i,'task','Time')
            tfc=Value(g,cmps[i+1],fc[i+1],i,'task','Time')
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
                pfn=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
                pfc=Value(g,cmps[i+1],fc[i+1],i,'task','Power')
                
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

if Test==4:
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
def Inference_Cost(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False):
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
    
if Test==4:
    _g='YOLOv3'

    print(Inference_Cost(_graph=_g,_order='N'*NLayers[_g],_freq=[[7]]*NLayers[_g],_debug=True))
    print(Inference_Cost(_graph=_g,_order='N'*57+'BNBB'+'BBBB'+'NNBBB'+'N'*5,_freq=[[7]]*NLayers[_g],_debug=False))
    
    _g='MobileV1'
    print(Inference_Cost(_graph=_g,_order='N'*NLayers[_g],_freq=[[7]]*NLayers[_g],_debug=True))


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