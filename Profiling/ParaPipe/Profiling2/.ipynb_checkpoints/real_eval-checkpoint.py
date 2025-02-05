from config import *
from profile_ import *
from parse_perf import *
from parse_power import *
import traceback
import math
import utils
import random
import numpy as np
import traceback

# +
Test=6
def Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ],suffix='',gpu_host='B', npu_host='B'):
    pf="pwr_whole.csv"
    tf="temp_whole.txt"
    pipeline=False
    if len(_ord)==1:
        _ord=NLayers[g]*_ord
    global Evaluations_df
    if suffix=='':
        suffix=g
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + suffix + ".csv"))
    #EvalFile=Evaluations_csv.split(".")[0]+'_'+g+Evaluations_csv.split(".")[0]
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        if pipeline:
            Evaluations_df=pd.DataFrame(columns=['graph','order','host','freq','pipeline_time','FPS','Latency', 'pipeline_power'])
        else:
            Evaluations_df=pd.DataFrame(columns=['graph','order','host','freq','input_time','task_time','output_time','total_time', 'input_power','task_power'])
    
    
    cols_to_add = ['output_time','input_e','task_e','output_e','total_e', 'power_efficiency']
    if pipeline:
        cols_to_add=[]
    # Loop through the columns to add
    
    for col in cols_to_add:
        # Check if the column is not already in the DataFrame
        if col not in Evaluations_df.columns:
            # Add the column with default values (in this case, NaN)
            Evaluations_df[col] = pd.Series([float('nan')] * len(Evaluations_df))
            

    
    host=(gpu_host+npu_host).upper()
    new_fs=[]
    repeat=False
    #print(evaluations)
    for ff in _fs:
        tpl_f=ff
        if type(ff)==list:
            tpl_f=(tuple(tuple(i) for i in ff))
        
        row=Evaluations_df[(Evaluations_df['order']==_ord) & (Evaluations_df['freq']==str(tpl_f)) & (Evaluations_df['graph']==g) & (Evaluations_df['host'] == host)]
        
        #if repeat==False and row.shape[0]==0:
        if repeat==True or row.shape[0]==0:
            new_fs.append(ff)
        else:
            print(f'{_ord}, Freq:{ff} already evaluated:')
            try:
                display(row)
            except:
                pprint.pprint(row)
            if pd.isna(row.reset_index().loc[0,'task_time']):
                new_fs.append(ff)

    if len(new_fs)==0:
        return Evaluations_df
    global n
    Profile(_ff=new_fs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pf, tme=tf,caching=False,kernel_c=96*50,gpu_host=gpu_host,npu_host=npu_host)
    
    time_df=None
    power_df=None
    if pipeline:
        time_df=Parse_total_pipeline(timefile=tf, graph=g, order=_ord, frqss=new_fs)
        power_df=Parse_Power_total_pipeline(file_name=pf,graph=g,order=_ord,frqss=new_fs)
    else:
        time_df=Parse_total(timefile=tf, graph=g, order=_ord, frqss=new_fs)
        power_df=Parse_Power_total(file_name=pf,graph=g,order=_ord,frqss=new_fs)
    
    
    if type(_fs[0])==list:
        if _fs[0][0][0]!='{' and type(_fs[0][0])!=str:
            power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
            time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
            
    display(time_df)
    display(power_df)
    merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])
    print(merged_df)
    #input()

    '''input_time=time_df['input_time'].iloc[0]
    task_time=time_df['task_time'].iloc[0]
    input_power=power_df['input_power'].iloc[0]
    task_power=power_df['task_power'].iloc[0]
    input_e=input_power*input_time
    task_e=task_power*task_time
    total_e=input_e+task_e
    merged_df['input_e']=input_e/1000.0
    merged_df['task_e']=task_e/1000.0
    merged_df['total_e']=total_e/1000.0'''
    if pipeline:
        merged_df['total_e']=Evaluations_df['pipeline_power']*Evaluations_df['pipeline_time']/1000.0
        merged_df['power_efficiency']=1000.0/merged_df['total_e']
    if not pipeline:
        merged_df['input_e']=merged_df['input_power']*merged_df['input_time']/1000.0
        merged_df['task_e']=merged_df['task_power']*merged_df['task_time']/1000.0
        merged_df['output_e']=merged_df['input_power']*merged_df['output_time']/1000.0
        merged_df['total_e']=merged_df['input_e']+merged_df['task_e']+merged_df['output_e']
        merged_df['power_efficiency']=1000.0/merged_df['total_e']
        merged_df['power_efficiency']=1000.0/merged_df['total_e']
    
    merged_df['host']=host
    try:
        display(merged_df)
    except:
        pprint.pprint(merged_df)
    #merged_df=merged_df.reset_index(drop=True,inplace=True)
    
    '''for i,k in merged_df.iterrows(): 
        r=Evaluations_df[(Evaluations_df['graph']==k['graph']) & (Evaluations_df['order']==k['order']) & (Evaluations_df['freq']==str(k['freq']))].index
        if(len(r)):
            r=r[0]
            for j,col in enumerate(Evaluations_df):
                Evaluations_df.iloc[r,j]=k[col]
        else:
            Evaluations_df=pd.concat([Evaluations_df,merged_df], ignore_index=True)
    '''        
            
    for i, k in merged_df.iterrows(): 
        # Filter rows based on graph, order, and freq columns
        r = Evaluations_df[(Evaluations_df['graph'] == k['graph']) & 
                     (Evaluations_df['order'] == k['order']) & 
                     (Evaluations_df['freq'] == str(k['freq']))].index

        # Check if 'host' column exists in Evals_df
        if 'host' in Evaluations_df.columns:
            # Add 'host' column to the filtering condition
            r = Evaluations_df[(Evaluations_df['graph'] == k['graph']) & 
                         (Evaluations_df['order'] == k['order']) & 
                         (Evaluations_df['freq'] == str(k['freq'])) & 
                         (Evaluations_df['host'] == str(k['host']))].index

        # Check if any rows were found
        if len(r):
            # Get the index of the first row
            r = r[0]
            # Update the columns in the row
            for col in Evaluations_df.columns:
                if col != 'score':
                    Evaluations_df.at[r, col] = k[col]
        else:
            # Concatenate the row from merged_df to Evals_df
            Evaluations_df = pd.concat([Evaluations_df, k.to_frame().T], ignore_index=True)
        

    print(Evaluations_df.columns)
    if pd.isna(Evaluations_df['power_efficiency']).any():
        if pipeline:
            Evaluations_df['total_e']=Evaluations_df['pipeline_power']*Evaluations_df['pipeline_time']/1000.0
            Evaluations_df['power_efficiency']=1000.0/Evaluations_df['total_e']
        else:
            Evaluations_df['input_e']=Evaluations_df['input_power']*Evaluations_df['input_time']/1000.0
            Evaluations_df['task_e']=Evaluations_df['task_power']*Evaluations_df['task_time']/1000.0
            Evaluations_df['output_e']=Evaluations_df['input_power']*Evaluations_df['output_time']/1000.0
            Evaluations_df['total_e']=Evaluations_df['input_e']+Evaluations_df['task_e']+merged_df['output_e']
            Evaluations_df['power_efficiency']=1000.0/Evaluations_df['total_e']
        
        
    
        
    Evaluations_df.to_csv(EvalFile,index=False)
    return Evaluations_df

    

if Test==4:
    ordd="NNNNNNNNNNNNNN"
    fff=[[7], [7], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="GGNNNNNNNNNNNN"
    fff=[[4,7], [4,7], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="BBNNNNNNNNNNNN"
    fff=[[7], [7], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    
    ordd="GLLLLNNNNNNNNN"
    fff=((4, 7), (5,), (5,), (5,), (5,), (7,), (2,), (3,), (2,), (3,), (2,), (5,), (7,), (7,))
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    
    ordd="BBLLLLLNNNBNNN"
    fff=((7,), (7,), (5,), (5,), (5,), (5,), (5,), (7,), (2,), (7,), (7,), (7,), (7,), (7,))
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="GLLLLLLNNNBNNN"
    fff=((4, 7), (5,), (5,), (5,), (5,), (5,), (5,), (7,), (2,), (7,), (7,), (7,), (7,), (7,))
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="BBNNNNNNNNNNNN"
    fff=((7,), (7,), (7,), (1,), (2,), (0,), (2,), (3,), (2,), (3,), (2,), (5,), (7,), (7,))
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="NNNNNNNNNNNNNN"
    fff=((7,), (4,), (2,), (1,), (2,), (0,), (2,), (3,), (2,), (3,), (2,), (5,), (7,), (7,))
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
    ordd="GGNNNNNNNNNNNN"
    fff=[[4,7], [4,7], [7],[7],[1],[1],[1],[1],[1],[1],[1],[1],[7],[7]]
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    
#fig 2 of dac paper
def eval_single_pe():
    for _g in graphs:
        for pe in ['L','B','G','N']:
            Real_Evaluation(g=_g,_ord=pe*NLayers[_g],_fs=[[['max']]])
            Real_Evaluation(g=_g,_ord=pe*NLayers[_g],_fs=[[['min']]])
            
if Test==3:
    eval_single_pe()
            
#fig 4 of dac paper
def eval_single_pe_f():
    for _g in graphs:
        if _g=="YOLOv3" or True:
            for pe in ['L','B','G']:
                fs=[]
                for f in range(NFreqs[pe]):
                    if pe=='G':
                        fs.append([[f,7]]*NLayers[_g])
                    else:
                        fs.append([[f]]*NLayers[_g])
                print(fs)
                Real_Evaluation(g=_g,_ord=pe*NLayers[_g],_fs=fs)
    
if Test==3:
    eval_single_pe_f()

def AOA():
    for _g in graphs:
        Real_Evaluation(g=_g,_ord='G',_fs=[[["min"]]],suffix="AOA")
        
if Test==2:
    AOA()

#print(Test)
#Real_Evaluation(g="Alex",_ord='LLLLBBBB',_fs=[ '{{2-3-4}}' ],suffix='npipe_pipeline',gpu_host='B', npu_host='B')
if Test==5:
    ordd="BBBBBBGB"
    fff="{{max}}"
    #fff=[[7], [7], [7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]]
    #Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="Alex",_ord=ordd,_fs=[ fff ])
    ordd="GGBBBGGG"
    Real_Evaluation(g="Alex",_ord=ordd,_fs=[ fff ])
    ordd="GGBBBGGB"
    Real_Evaluation(g="Alex",_ord=ordd,_fs=[ fff ])
    Real_Evaluation(g="Alex",_ord='B',_fs=[ fff ])
    Real_Evaluation(g="Alex",_ord='G',_fs=[ fff ])
    Real_Evaluation(g="Alex",_ord='L',_fs=[ fff ])
    ordd="GGGGGGBBBBBBBB"
    Real_Evaluation(g="MobileV1",_ord=ordd,_fs=[ fff ])
    Real_Evaluation(g="MobileV1",_ord='B',_fs=[ fff ])
    Real_Evaluation(g="MobileV1",_ord='G',_fs=[ fff ])
    Real_Evaluation(g="MobileV1",_ord='L',_fs=[ fff ])
    ordd="GGGGGGGBBBBBBBBBBB"
    Real_Evaluation(g="ResV1_50",_ord=ordd,_fs=[ fff ])
    Real_Evaluation(g="ResV1_50",_ord='B',_fs=[ fff ])
    Real_Evaluation(g="ResV1_50",_ord='G',_fs=[ fff ])
    Real_Evaluation(g="ResV1_50",_ord='L',_fs=[ fff ])

# +
#Yujie Parapipe collaboration
if Test==5:
    graph_name = 'Google'
    mappings = ['GGGGBBBBBBG', 'GGGGBBBBBBB', 'B'*11, 'G'*11] 
    t_sum,e_sum=[],[]
    for mapping in mappings:
        Real_Evaluation(g=graph_name, _ord=mapping, _fs=[['{{max}}']])
        time.sleep(60)
        #input("continue?")

    graph_name = 'InceptionResnetV2'
    mappings = ['GGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', 'B'*50, 'G'*50, 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBG', 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBB'] 
    for mapping in mappings:
        Real_Evaluation(g=graph_name, _ord=mapping, _fs=[['{{max}}']])
        time.sleep(60)
        #input("continue?")

    graph_name='InceptionV3'
    mappings = ['GGGGGGGGBBBBBBBBG', 'GGGGGGGGBBBBBBBBB', 'B'*17, 'G'*17]
    for mapping in mappings:
        Real_Evaluation(g=graph_name, _ord=mapping, _fs=[['{{max}}']])
        time.sleep(60)
        #input("continue?")

    graph_name='InceptionV4'
    mappings = ['GGGBGBGGGGBBBBBBBBBBBBB', 'GGGGGBGGGGBBBBBBBBBBBBB', 'GGGBGGGGGGBBBBBBBBBBBBB', 'GGGGGGGGGGBBBBBBBBBBBBB', 'B'*23, 'G'*23, 'GGGGGGGGGGGGGGGGGGGGGGB']
    for mapping in mappings:
        Real_Evaluation(g=graph_name, _ord=mapping, _fs=[['{{max}}']])
        time.sleep(60)
        #input("continue?")

        
def run_for_memory_profile(graph_name,mappings):
    for mapping in mappings:
        input("press to start...")
        Real_Evaluation(g=graph_name, _ord=mapping, _fs=[['{{max}}']])
        input("finished press to next...")
        #time.sleep(60)
    
#Yujie Reviews
if Test==6:
    graph_name = 'Google'
    mappings = ['GGGGBBBBBBB', 'B'*11, 'G'*11]
    #mappings = ['GGGGBBBBBBB']
    
    run_for_memory_profile(graph_name,mappings)
    

    graph_name = 'InceptionResnetV2'
    mappings = ['GGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', 'B'*50, 'G'*50] 
    run_for_memory_profile(graph_name,mappings)

    graph_name='InceptionV3'
    mappings = ['GGGGGGGGBBBBBBBBB', 'B'*17, 'G'*17]
    run_for_memory_profile(graph_name,mappings)

    graph_name='InceptionV4'
    mappings = ['GGGGGGGGGGBBBBBBBBBBBBB', 'B'*23, 'G'*23]
    run_for_memory_profile(graph_name,mappings)
# -

'''Google= {'switching':216,'B':165,'G':285}
InceptionResnetV2= {'switching':708,'B':677,'G':742}
InceptionV3= {'switching':415,'B':360,'G':470}
InceptionV4= {'switching':607,'B':567,'G':614}'''

if Test==4:
    for _g in graphs:
        _cmps=['N']
        for cmp in _cmps:
            fs=[[NFreqs['B']-1]]*NLayers[_g]
            if cmp=='G':
                fs=[[NFreqs[cmp]-1,NFreqs['B']-1]]*NLayers[_g]
            Real_Evaluation(g=_g,_ord=NLayers[_g]*cmp,_fs=[ [['max']] ],suffix='',npu_host='L')
            Real_Evaluation(g=_g,_ord=NLayers[_g]*cmp,_fs=[ [['max']] ],suffix='',npu_host='B')

# +
#Real_Evaluation(g="Alex",_ord='GNNNNNNL',_fs=[ [['max']] ],suffix='')
#Real_Evaluation(g="Alex",_ord='GNNNNNNL',_fs=[ [['max']] ],suffix='',npu_host='B',gpu_host='L')
#Real_Evaluation(g="Alex",_ord='NNNNNNNN',_fs=[ [['max']] ],suffix='',npu_host='L')
#Real_Evaluation(g="Alex",_ord='NNNNNNNN',_fs=[ [['max']] ],suffix='',npu_host='B')

# +
#Fixed freq
Motivation_Fig2=False
#def Motivation_Fig2():
if Motivation_Fig2:
    _g='mobile'
    N=NLayers[_g]
    Real_Evaluation(g=_g,_ord='L',_fs=[[[5]]*N,[[4]]*N,[[3]]*N,[[2]]*N,[[1]]*N,[[0]]*N],suffix="Motivation_Figure")
    Real_Evaluation(g=_g,_ord='B',_fs=[[[0]]*N,[[1]]*N,[[2]]*N,[[3]]*N,[[4]]*N,[[5]]*N,[[6]]*N,[[7]]*N],suffix="Motivation_Figure")
    Real_Evaluation(g=_g,_ord='G',_fs=[[[0,0]]*N,[[1,1]]*N,[[2,2]]*N,[[3,3]]*N,[[4,4]]*N],suffix="Motivation_Figure")
    
#Real_Evaluation(g="google",_ord='LLLLLLLLLLL',_fs=[[[5]]*11],suffix="ttt")
# -

# Todaes number of threads (cores):
threads_df=None
time_file='threadstime.txt'
pwr_file='threadspower.csv'
_freqs=['{{max}}']

# Todaes thread exploration
if False:
    for g in graphs:
        #g='Alex'
        _ord='B'*NLayers[g]
        for _threads_big in range(1,7):
            #_threads_big=1
            _threads_little=4


            Profile(_ff=_freqs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pwr_file, tme=time_file,caching=False,threads_big=_threads_big, threads_little=_threads_little)

            time_df=None
            power_df=None
            pipeline=False
            if pipeline:
                time_df=Parse_total_pipeline(timefile=time_file, graph=g, order=_ord, frqss=_freqs)
                power_df=Parse_Power_total_pipeline(file_name=pwr_file,graph=g,order=_ord,frqss=_freqs)
            else:
                time_df=Parse_total(timefile=time_file, graph=g, order=_ord, frqss=_freqs)
                power_df=Parse_Power_total(file_name=pwr_file,graph=g,order=_ord,frqss=_freqs)


            if type(_freqs[0])==list:
                if _fs[0][0][0]!='{':
                    power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
                    time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )

            display(time_df)
            display(power_df)
            merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])
            merged_df['threads_little']=_threads_big
            merged_df['threads_big']=_threads_little
            threads_df=pd.concat([threads_df,merged_df])
            display(merged_df)
            display(threads_df)

        _ord='L'*NLayers[g]
        for _threads_little in range(1,7):

            _threads_big=2
            #_threads_little=4


            Profile(_ff=_freqs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pwr_file, tme=time_file,caching=False,threads_big=_threads_big, threads_little=_threads_little)

            time_df=None
            power_df=None
            pipeline=False
            if pipeline:
                time_df=Parse_total_pipeline(timefile=time_file, graph=g, order=_ord, frqss=_freqs)
                power_df=Parse_Power_total_pipeline(file_name=pwr_file,graph=g,order=_ord,frqss=_freqs)
            else:
                time_df=Parse_total(timefile=time_file, graph=g, order=_ord, frqss=_freqs)
                power_df=Parse_Power_total(file_name=pwr_file,graph=g,order=_ord,frqss=_freqs)


            if type(_freqs[0])==list:
                if _fs[0][0][0]!='{':
                    power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
                    time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )

            display(time_df)
            display(power_df)
            merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])
            merged_df['threads_little']=_threads_big
            merged_df['threads_big']=_threads_little
            threads_df=pd.concat([threads_df,merged_df])
            display(merged_df)
            display(threads_df)
            threads_df.to_csv('threadsss.csv')


# +
#This version of Real_Evalutaion is for evaluating GA results, so it get the df instead of using global one
def Real_Evaluation_For_GA(g="alex",_ord='GBBBBBBB',_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ],Evals_df=None,FileName="",npu_host='B', gpu_host='B'):
    pf="pwr_whole.csv"
    tf="temp_whole.txt"
    
    if len(_ord)==1:
        _ord=NLayers[g]*_ord
    
    EvalFile = FileName
    if EvalFile.exists():
        Evals_df=pd.read_csv(EvalFile)
    #else:
    #    Evals_df=pd.DataFrame(columns=['graph','order','freq','socre','input_time','task_time','total_time', 'input_power','task_power'])
    cols_to_add = ['input_time','task_time','total_time', 'input_power','task_power','input_e','task_e','total_e']
    # Loop through the columns to add
    
    for col in cols_to_add:
        # Check if the column is not already in the DataFrame
        if col not in Evals_df.columns:
            # Add the column with default values (in this case, NaN)
            Evals_df[col] = pd.Series([float('nan')] * len(Evals_df))
    
    
    
    new_fs=[]
    repeat=False
    #print(evaluations)
    for ff in _fs:
        tpl_f=ff
        if type(ff)==list:
            tpl_f=(tuple(tuple(i) for i in ff))
        
        row=Evals_df[(Evals_df['order']==_ord) & (Evals_df['freq']==str(tpl_f)) & (Evals_df['graph']==g)]
        
        if repeat==True or row.shape[0]==0:
            print(f'new freq adding freq {ff}')
            new_fs.append(ff)
        else:
            print(f'{_ord}, Freq:{ff} already evaluated:')
            try:
                display(row)
            except:
                pprint.pprint(row)
            
            if pd.isna(row.reset_index()['task_time']).all():
                print(f'task_time is none adding freq {ff}')
                new_fs.append(ff)

    if len(new_fs)==0:
        return Evals_df
    
    
    Profile(_ff=new_fs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pf, tme=tf,caching=False,kernel_c=96*50, npu_host=npu_host, gpu_host=gpu_host)
    time_df=Parse_total(timefile=tf, graph=g, order=_ord, frqss=new_fs)
    power_df=Parse_Power_total(file_name=pf,graph=g,order=_ord,frqss=new_fs)
    
    if type(_fs[0])==list:
        if _fs[0][0][0]!='{':
            power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
            time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
    merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])


    '''input_time=time_df['input_time'].iloc[0]
    task_time=time_df['task_time'].iloc[0]
    input_power=power_df['input_power'].iloc[0]
    task_power=power_df['task_power'].iloc[0]
    input_e=input_power*input_time
    task_e=task_power*task_time
    total_e=input_e+task_e
    merged_df['input_e']=input_e/1000.0
    merged_df['task_e']=task_e/1000.0
    merged_df['total_e']=total_e/1000.0'''
    
    merged_df['input_e']=merged_df['input_power']*merged_df['input_time']/1000.0
    merged_df['task_e']=merged_df['task_power']*merged_df['task_time']/1000.0
    merged_df['total_e']=merged_df['input_e']+merged_df['task_e']
    if 'host' in Evals_df.columns:
        merged_df['host']=gpu_host+npu_host
    #merged_df['score']=Evals_df['score']
    try:
        display(merged_df)
    except:
        pprint.pprint(merged_df)
    #merged_df=merged_df.reset_index(drop=True,inplace=True)
    '''for i,k in merged_df.iterrows(): 
        r=Evals_df[(Evals_df['graph']==k['graph']) & (Evals_df['order']==k['order']) & (Evals_df['freq']==str(k['freq']))].index
        if 'host' in Evals_df.columns:
            r=Evals_df[(Evals_df['graph']==k['graph']) & (Evals_df['order']==k['order']) & (Evals_df['freq']==str(k['freq']) & (Evals_df['host']==str(k['host']))].index
        if(len(r)):
            r=r[0]
            for j,col in enumerate(Evals_df):
                if col!='score':
                    #Evals_df.iloc[r,j]=k[col]
                    Evals_df.iloc[r][col]=k[col]
        else:
            Evals_df=pd.concat([Evals_df,merged_df], ignore_index=True)'''
    for i, k in merged_df.iterrows(): 
        # Filter rows based on graph, order, and freq columns
        r = Evals_df[(Evals_df['graph'] == k['graph']) & 
                     (Evals_df['order'] == k['order']) & 
                     (Evals_df['freq'] == str(k['freq']))].index

        # Check if 'host' column exists in Evals_df
        if 'host' in Evals_df.columns:
            # Add 'host' column to the filtering condition
            r = Evals_df[(Evals_df['graph'] == k['graph']) & 
                         (Evals_df['order'] == k['order']) & 
                         (Evals_df['freq'] == str(k['freq'])) & 
                         (Evals_df['host'] == str(k['host']))].index

        # Check if any rows were found
        if len(r):
            # Get the index of the first row
            r = r[0]
            # Update the columns in the row
            for col in Evals_df.columns:
                if col != 'score' and col in k:
                    Evals_df.at[r, col] = k[col]
        else:
            # Concatenate the row from merged_df to Evals_df
            Evals_df = pd.concat([Evals_df, k.to_frame().T], ignore_index=True)

        

    Evals_df.to_csv(EvalFile,index=False)
    return Evals_df




# +
#This is for reading GA result file and run the the real evalation for GA
def Run_Eval_For_GA(_FileName):
    
    
    if _FileName.exists():
        Evals_df=pd.read_csv(_FileName).drop_duplicates()
    else:
        print("Ga result file is not existed")
        return
    
    cases=Evals_df.shape[0]
    print(f'There are {cases}')
    

    for g in  graphs:
        
        if 'host' in Evals_df.columns:
            grouped = Evals_df[Evals_df['graph'] == g].groupby(['order', 'host'])
            for (order, host), group in grouped:
                print(f"Order: {order}, Host: {host}")
                # Get the values in column 'freq' for the current group
                column_freq_values = group['freq'].values
                # Print the value in column 'order' and 'host' and the corresponding values in column 'freq'
                print(f"Values in column 'freq': {column_freq_values}")
                # Process the frequency values and do further operations as needed
                processed_freqs = utils.process_freqs(column_freq_values, mode='per_processor')
                print(f'processed freqs:{processed_freqs}')
                print("----")
                # Perform additional operations with the processed frequencies if needed
                Real_Evaluation_For_GA(g, _ord=order, npu_host=host[1].upper(), gpu_host=host[0].upper(), _fs=processed_freqs, Evals_df=Evals_df, FileName=_FileName)
        else:
        
            grouped = Evals_df[Evals_df['graph']==g].groupby('order')
            unique_values_order = Evals_df[Evals_df['graph']==g]['order'].unique()
            #print(grouped.unique())
            print(unique_values_order)
            # Loop through the unique values in column 'order'
            for value in unique_values_order:
                # Get the group corresponding to the current value in column 'order'
                group = grouped.get_group(value)
                # Get the values in column 'freq' for the current group
                column_freq_values = group['freq'].values
                # Print the value in column 'A' and the corresponding values in column 'freq'
                print(f"Value in column 'order': {value}")
                #processed_freqs=utils.process_freqs(column_freq_values, mode='per_processor')
                processed_freqs=utils.format_to_list(column_freq_values)
                #processed_freqs=utils.format_freqs(column_freq_values)
                print(f'processed freqs:{processed_freqs}')
                print("----")
                try:
                    Real_Evaluation_For_GA(g,_ord=value,_fs=processed_freqs,Evals_df=Evals_df,FileName=_FileName)
                except Exception as e:
                    print(f'Error {e}')
                    traceback.print_exc()
                    time.sleep(3)
                    input()
                    continue
if Test==3:
    Run_Eval_For_GA(_FileName=GA_Results_PELSI)
    Run_Eval_For_GA(_FileName=GA_Results_LW)
    Run_Eval_For_GA(Path("Evaluations_YOLOv3_test.csv").resolve())
 


#TODAES
#Run_Eval_For_GA(_FileName=Path('Alex_Xs.csv').resolve())

#DAC:
#Run_Eval_For_GA(_FileName=Path('YOLOv3-66.csv').resolve())

#Run_Eval_For_GA(_FileName=Path('Yolo66.csv').resolve())



#Run_Eval_For_GA(_FileName=Path('TestYolo.csv').resolve())

# +
def Fill_prediction(_FileName, dvfs_delay):
    if _FileName.exists():
        Evals_df=pd.read_csv(_FileName).drop_duplicates()
    else:
        print("Ga result file is not existed")
        return
    
    Evals_df['total_e']=Evals_df['input_e']+Evals_df['task_e']+Evals_df['output_e']
    cases=Evals_df.shape[0]
    print(f'There are {cases}')
    
    Regenerate_Predeiction=False
    Regenerate_Errors=False
    
    def prediction(row):
        #print(row)
        graph=row['graph']
        freq=utils.format_to_list([row['freq']])[0]
        order=row['order']
        #print(graph,freq,order,dvfs_delay)
        return Inference_Cost(_graph=graph,_freq=freq,_order=order,_dvfs_delay=dvfs_delay, _debug=True)
    if 'Predicted_Time' not in Evals_df:
        Evals_df[['Predicted_Time','Predicted_Power','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    if 'Predicted_Time' in Evals_df:
        if pd.isna(Evals_df['Predicted_Time']).any() or Regenerate_Predeiction:
            Evals_df[['Predicted_Time','Predicted_Power','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    
    #display(Evals_df)
    def calc_EE(row):
        Measured=1000.0/row['total_e']
        Pred=1000.0/row['Predicted_Energy']
        Err=(Pred-Measured)/Measured
        return 100.0*Err
    
    def calc_Power(row):
        measured=row['total_e']/row['total_time']
        pred=row['Predicted_Energy']/row['Predicted_Time']
        Err=100*(pred-measured)/measured
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
    
    if 'Error_Time' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Time']=Evals_df.apply(lambda x:100*(x['Predicted_Time']-x['total_time'])/x['total_time'],axis=1)
    if 'Error_Energy' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Energy']=Evals_df.apply(lambda x:100*(x['Predicted_Energy']-x['total_e'])/x['total_e'],axis=1)
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
# -

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
#Layers_df[(Layers_df['Graph']=='google') & (Layers_df['Layer']==4) & (Layers_df['Component']=='B') &(Layers_df['Freq']==0)]

# +
#Value('google','L',[0],[3],'task','Time')
# -

def _Test():
    _fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ],
         [ [7],[6],[5],[4],[3],[2],[1],[0] ] ]
    _order='BBBBBBBB'
    _g="alex"
    for fs in _fs:
        Real_Evaluation(g="alex",_ord=_order,_fs=[fs])
        ''' Profile(_ff=[fs], _Num_frames=Num_frames, order=_order, graph=_g, pwr="pwr.csv", tme="temp.txt",caching=False)
        time=Parse(timefile="temp.txt", graph=_g, order=_order, frqss=[fs])
        power=Parse_Power(pwrfile="pwr.csv", graph=_g, order=_order, frqss=[fs])
        print(time)
        print(power)'''
        _dvfs_delay=3.5
        #np.reshape(fs,-1)
        cmp=Comp_Cost(g=_g,fn=fs,cmps=_order,dvfs_delay=_dvfs_delay, debug=False)
        cmm=Comm_Cost(g=_g,fn=fs,cmps=_order,dvfs_delay=_dvfs_delay, debug=False)
        print(cmp)
        print(cmm)





'''def Run_Eval(g='alex',num_evals=1000,num_freqs=10):
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])    
    cases=Evaluations_df[Evaluations_df['graph']==g].shape[0]
    print(f'There are {cases} existed for graph {g}')
    num_evals=max(0,num_evals-cases)
    num_orders=math.ceil(num_evals/num_freqs)
    
    _n=NLayers[g]
    orders=utils.generate_random_strings(_n,num_orders)
               
    fs={}
    for order in orders:
        fs[order]=[]
        for k in range(num_freqs):
            f=[]
            for i,comp in enumerate(order):
                v=[]
                if comp=='N':
                    comp='B'
                v.append(random.randint(0, NFreqs[comp]-1))
                if comp=='G':
                    v.append(random.randint(0, NFreqs['B']-1))
                f.append(tuple(v))
                
            fs[order].append(str(tuple(f)))
            
    
    for order in fs:
        for f in fs[order]:
            row=Evaluations_df[(Evaluations_df['order']==order) & (Evaluations_df['freq']==str(f)) & (Evaluations_df['graph']==g)]
            if row.shape[0]==0:
                Evaluations_df.loc[len(Evaluations_df)]={"graph":g,"order":order,"freq":f}
            
    Evaluations_df.to_csv(EvalFile,index=False)
    
    grouped = Evaluations_df.groupby('order')
    unique_values_order = Evaluations_df['order'].unique()

    # Loop through the unique values in column 'order'
    for value in unique_values_order:
        # Get the group corresponding to the current value in column 'order'
        group = grouped.get_group(value)
        # Get the values in column 'freq' for the current group
        column_freq_values = group['freq'].values
        # Print the value in column 'A' and the corresponding values in column 'freq'
        print(f"Value in column 'order': {value}")
        print(f"Values in column 'freq': {column_freq_values}")
        print("----")
        list_fs=utils.format_to_list(column_freq_values)
        Real_Evaluation(g,_ord=value,_fs=list_fs)
if Test==3:
    Run_Eval(g='Alex')'''


def Run_Eval_no_dvfs(g='alex',num_evals=1000,num_freqs=1):
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','host','input_time','task_time','total_time', 'input_power','task_power'])    
    cases=Evaluations_df[Evaluations_df['graph']==g].shape[0]
    print(f'There are {cases} existed for graph {g}')
    num_evals=max(0,num_evals-cases)
    num_orders=math.ceil(num_evals/num_freqs)
    
    _n=NLayers[g]
    orders=utils.generate_random_strings(_n,num_orders)
               
    fs={}
    for order in orders:
        fs[order]=[]
        for k in range(num_freqs):
            f=[]
            for i,comp in enumerate(order):
                v=[]
                if comp=='N':
                    comp='B'
                v.append(NFreqs[comp]-1)
                #v.append(random.randint(0, NFreqs[comp]-1))
                if comp=='G':
                    v.append(NFreqs['B']-1)
                    #v.append(random.randint(0, NFreqs['B']-1))
                if comp=='N':
                    v.append(NFreqs['B']-1)
                    #v.append(random.randint(0, NFreqs['B']-1))
                f.append(tuple(v))
                
            fs[order].append(str(tuple(f)))
            
    '''print(orders)
    print(fs)
    input()'''
    for order in fs:
        for f in fs[order]:
            row=Evaluations_df[(Evaluations_df['order']==order) & (Evaluations_df['freq']==str(f)) & (Evaluations_df['graph']==g) & (Evaluations_df['host']=='BB')]
            if row.shape[0]==0:
                Evaluations_df.loc[len(Evaluations_df)]={"graph":g,"order":order,"freq":f, 'host':'BB'}
            
    Evaluations_df.to_csv(EvalFile,index=False)
    
    grouped = Evaluations_df.groupby('order')
    unique_values_order = Evaluations_df['order'].unique()

    # Loop through the unique values in column 'order'
    for value in unique_values_order:
        # Get the group corresponding to the current value in column 'order'
        group = grouped.get_group(value)
        # Get the values in column 'freq' for the current group
        column_freq_values = group['freq'].values
        # Print the value in column 'A' and the corresponding values in column 'freq'
        print(f"Value in column 'order': {value}")
        print(f"Values in column 'freq': {column_freq_values}")
        print("----")
        list_fs=utils.format_to_list(column_freq_values)
        try:
            Real_Evaluation(g,_ord=value,_fs=list_fs)
        except:
            print(f"error in running {value}")
            #os.system('khadasOff')
            time.sleep(2)
            continue

# +




def Run_Eval_DAC(g='YOLOv3',num_changes=10,step=10):
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
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
    
    orders=generate_strings_with_changes_and_step(base_string, num_changes=num_changes, step=step)
    
    
        
        
    print(orders)
    print(len(orders))
    
    
               
    #_fs=str(tuple([7]*_n))
    fs={}
    for order in orders:        
        fs[order]=[str((((7,),)*_n))]
            
    
    for order in fs:
        for f in fs[order]:
            row=Evaluations_df[(Evaluations_df['order']==order) & (Evaluations_df['freq']==str(f)) & (Evaluations_df['graph']==g)]
            if row.shape[0]==0:
                Evaluations_df.loc[len(Evaluations_df)]={"graph":g,"order":order,"freq":f}
            
    Evaluations_df.to_csv(EvalFile,index=False)
    
    grouped = Evaluations_df.groupby('order')
    unique_values_order = Evaluations_df['order'].unique()

    # Loop through the unique values in column 'order'
    for value in unique_values_order:
        # Get the group corresponding to the current value in column 'order'
        group = grouped.get_group(value)
        # Get the values in column 'freq' for the current group
        column_freq_values = group['freq'].values
        # Print the value in column 'A' and the corresponding values in column 'freq'
        print(f"Value in column 'order': {value}")
        print(f"Values in column 'freq': {column_freq_values}")
        print("----")
        list_fs=utils.format_to_list(column_freq_values)
        Real_Evaluation(g,_ord=value,_fs=list_fs)
if Test==4:
    #Run_Eval_DAC(g='YOLOv3',step=1,num_changes=1)
    Run_Eval_DAC(g='YOLOv3',step=5,num_changes=2)
    #Run_Eval_DAC(g='MobileV1',step=1)


# +

    
    
def Run_Eval_ISLPED_test(g='YOLOv3',num_changes=10,step=10):
    #EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
    EvalFile=Path("TestYolo2.csv").resolve()
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
    
    Run_Eval_For_GA(_FileName=EvalFile)
    
    '''grouped = Evaluations_df.groupby('order')
    unique_values_order = Evaluations_df['order'].unique()

    # Loop through the unique values in column 'order'
    for value in unique_values_order:
        # Get the group corresponding to the current value in column 'order'
        group = grouped.get_group(value)
        # Get the values in column 'freq' for the current group
        column_freq_values = group['freq'].values
        # Print the value in column 'A' and the corresponding values in column 'freq'
        print(f"Value in column 'order': {value}")
        print(f"Values in column 'freq': {column_freq_values}")
        print("----")
        #list_fs=utils.format_freqs(column_freq_values)
        list_fs=column_freq_values
        Real_Evaluation(g,_ord=value,_fs=list_fs)
        #Run_Eval_For_GA(g,_ord=value,_fs=list_fs,EvalFile=EvalFile)'''
if Test==5:
    #Run_Eval_DAC(g='YOLOv3',step=1,num_changes=1)
    Run_Eval_ISLPED_test(g='YOLOv3',step=5,num_changes=2)
    #Run_Eval_DAC(g='MobileV1',step=1)
    
# -

def Gather_real_profile(_g,_num_evals):
    Finished=False
    while not Finished:
        try:
            Run_Eval_no_dvfs(g=_g,num_evals=_num_evals)
            Finished=True
        except Exception as e:
            print("Error occurred:", e)
            print("Traceback:")
            traceback.print_exc()
            # #!sudo apt install sox
            os.system('play -nq -t alsa synth {} sine {}'.format(5, 440))
            #input("Continue?")
            utils.ab()
            time.sleep(5)
#3
if Test==3:
    for g in graphs:
        if g=='YOLOv3':
            Gather_real_profile(g,1000)

import serial
serPort = "/dev/ttyACM0"
baudRate = 115200

# +
#ser = serial.Serial(serPort, baudRate)
# -


