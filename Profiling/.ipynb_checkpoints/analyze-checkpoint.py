from config import *
from data import *
import matplotlib.pyplot as plt
import predict_cost
import real_eval

# +
t=Layers_df[(Layers_df['Graph']=='Alex') & (Layers_df['Component']=='L')]
t['Energy']=t['Time']*t['Power']/1000
# Group by 'Layer' and 'Freq' and calculate the mean 'Energy' for each group
effect_of_freq_per_layer = t.groupby(['Layer', 'Freq'])['Time'].max().reset_index()
#display(effect_of_freq_per_layer)
#input()
# Iterate over each layer and plot the 'Freq' vs 'Energy'
for layer in effect_of_freq_per_layer['Layer'].unique():
    layer_data = effect_of_freq_per_layer[effect_of_freq_per_layer['Layer'] == layer]
    plt.plot(layer_data['Freq'], layer_data['Time'], label=f'Layer {layer}')

plt.xlabel('Frequency (Freq)')
plt.ylabel('Energy')
plt.title('Effect of Freq on Energy for each Layer')
plt.legend()
plt.show()
# -



# +

#exploring dvfs for layers for example
def Analyze(graph_name=graphs,metric=['task','in','out','trans'],comp=['G','B','L'],
            freq_h=[-1],f=range(10),layers=range(40),index=['Layer'],columns=['Freq'],parameter='Time'):

    # This was for old parser that save time in NPU_run_profile metric, but new parser save time in task meteric same as cpu and gpu
    # the old is that I use for dac (first version before revision)
    if 'N' in comp and False:
        # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
        grouped_df1 = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                    (Layers_df['Metric'].isin(metric)) & 
                    (Layers_df['Component'].isin(comp)) & 
                    (Layers_df['Freq_Host'].isin(freq_h))&
                    (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Power'].sum().reset_index()
        # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
        metric=["NPU_run_profile"]
        grouped_df2 = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                    (Layers_df['Metric'].isin(metric)) & 
                    (Layers_df['Component'].isin(comp)) & 
                    (Layers_df['Freq_Host'].isin(freq_h))&
                    (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Time'].sum().reset_index()
        grouped_df=pd.merge(grouped_df1,grouped_df2)
    else:
        # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
        grouped_df = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                        (Layers_df['Metric'].isin(metric)) & 
                        (Layers_df['Component'].isin(comp)) & 
                        (Layers_df['Freq_Host'].isin(freq_h))&
                        (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Time','Power'].sum().reset_index()
        
    #print(grouped_df)
    #input()
    grouped_df['Energy']=grouped_df['Power']*grouped_df['Time']/1000.0
    grouped_df['Power-Efficiency']=1000.0/(grouped_df['Energy'])
    if "N" in comp:
        grouped_df.to_csv("nn.csv")
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df, values=parameter, index=index, columns=columns)
    
        
    frequency_table = comp_to_frequency_table[comp[0]]
    
    # Map column headers to their corresponding values
    column_mapping = {column: value/1000000 if comp[0]=='G' else value/1000 for column, value in zip(pivot_table.columns, frequency_table)}

    # Rename columns using the mapping
    pivot_table.rename(columns=column_mapping, inplace=True)
    try:
        display(pivot_table)
    except:
        pprint.pprint(pivot_table)
    pivot_table.to_csv(f'Todaes_power_efficiency_vs_freq_{comp}.csv')
    #pivot_table.plot(kind='bar', stacked=False, figsize=(30, 14))
    #plt.figure(figsize=(6, 3))
    font_size = 16
    _figsize=(6,3)
    bar_width = 0.75
    pivot_table.plot(kind='bar',width=bar_width, stacked=False, figsize=_figsize)#_figsize=(10, 5.625)
    
    #pivot_table.plot(kind='bar', stacked=False, figsize=(12, 12))
    
    #This part is wrote for getting graph for presentation if you see error later
    # consider removing it and uncommenting next lines that are similar
    
    #plt.title(f'{parameter} vs {columns[0]} for {graph_name[0]} {index[0]}s', fontsize=font_size)
    plt.xlabel(f'{index[0]}', fontsize=font_size-2)
    plt.ylabel(f'{parameter}', fontsize=font_size-2)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    legend=[fr"$F_{{{i}}}$" for i in pivot_table.columns.astype(int).tolist()]
    
    #plt.legend(legend,fontsize=font_size-4,loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(legend))
    plt.legend(legend,fontsize=font_size-4,loc='upper right', ncol=1)
    _legend = plt.legend()
    _legend.get_frame().set_linewidth(0.7)  # Adjusting the border width
    _legend.get_frame().set_edgecolor('black')  # Setting the border color
    plt.xlim(-1, 16)
    plt.tight_layout()
    '''
    plt.title(f'{metric} {parameter} vs {columns} for {graph_name} ')
    plt.xlabel(f'{index}')
    plt.ylabel(f'{metric} {parameter}')
    plt.show()'''
    plt.savefig(f'Layers_Freq_{comp[0]}.pdf', format='pdf', dpi=3000)
    return pivot_table

#Figure of TODAES (Power efficiency vs. Frequency for MobileNetV1 layers.)
if Test==4:
    g='MobileV1'
    Analyze(graph_name=[g],metric=['task'],comp=['L'],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
    Analyze(graph_name=[g],metric=['task'],comp=['B'],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
    Analyze(graph_name=[g],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
    Analyze(graph_name=[g],metric=['task'],comp=['N'],freq_h=list(range(10)),index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')

# +
if False:
    graph_name=['MobileV1']
    metric=['NPU_run_profile']
    comp=['N']
    freq_h=range(10)
    index=['Layer']
    columns=['Freq']
    parameter='Power-Efficiency'
    #parameter='Time'
    f=range(10)
    layers=range(100)


    d1=Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                        (Layers_df['Metric'].isin(metric)) & 
                        (Layers_df['Component'].isin(comp)) & 
                        (Layers_df['Freq_Host'].isin(freq_h))&
                        (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Time'].sum().reset_index()

    display(d1)
    metric=['task']
    d2=Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                        (Layers_df['Metric'].isin(metric)) & 
                        (Layers_df['Component'].isin(comp)) & 
                        (Layers_df['Freq_Host'].isin(freq_h))&
                        (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Power'].sum().reset_index()
    d2
    pd.merge(d1,d2)





# -

#Extract data of mobile and yolo for dac paper (after running this func remember to fill the power value of layers:
# 57,58 based on layers 59,60 (I filled it with 5707mw)
# and layers 65,66 based on layers 67,68 (again 5707)
# so do not repeat this function otherwise you need to fill this values again
if Test==4:
    graph_name=['YOLOv3','MobileV1']
    metric=['task']
    comp=['L','B','G','N']
    freq_h=[-1,7]
    #freq=[7]
    freq=list(range(8))
    layers=range(80)
    index=['Layer']
    columns=[]
    parameter='Time'
    grouped_df = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                        #(Layers_df['Metric'].isin(metric)) & 
                        (((Layers_df['Component'].isin( ['B','L','G'])) & (Layers_df['Metric'] == 'task')) |((Layers_df['Component'] == 'N') & (Layers_df['Metric'].isin(['task','NPU_run_profile'])))) &
                        (Layers_df['Component'].isin(comp)) & 
                        (Layers_df['Freq_Host'].isin(freq_h))&
                        (Layers_df['Freq'].isin(freq)) &
                        (Layers_df['Layer'].isin(layers)) ]
    grouped_df.reset_index().drop('index',axis=1).to_csv('DAC.csv') #.reset_index(ignore_index=True) newer python versions


# Compare NPU with B
if Test==4:
    df = pd.read_csv('DAC.csv',index_col=0)
    '''task_df = df[df['Metric'] == 'task'][['Graph', 'Component', 'Freq', 'Freq_Host', 'Layer', 'Power']]
    npu_run_profile_df = df[df['Metric'] == 'NPU_run_profile']

    # Merge the 'Power' values from 'task_df' into 'npu_run_profile_df' based on the matching columns
    npu_run_profile_with_power = pd.merge(npu_run_profile_df, task_df, on=['Graph', 'Component', 'Freq', 'Freq_Host', 'Layer'])'''

                            
    task_df = df[df['Metric'] == 'task'][['Graph', 'Component', 'Freq', 'Freq_Host', 'Layer', 'Power']]
    npu_run_profile_df = df[df['Metric'] == 'NPU_run_profile']

    # Merge the 'Power' values from 'task_df' into 'npu_run_profile_df' based on the matching columns
    npu_run_profile_with_power = pd.merge(npu_run_profile_df, task_df, on=['Graph', 'Component', 'Freq', 'Freq_Host', 'Layer'])
                            
    
    # Drop the old 'Power' column (which contains NaNs) from 'npu_run_profile_with_power'
    npu_run_profile_with_power.drop(columns=['Power_x'], inplace=True)

    # Rename the merged 'Power' column to just 'Power'
    npu_run_profile_with_power.rename(columns={'Power_y': 'Power'}, inplace=True)
    

    '''
    # Now, you want to update these rows in the original 'df'
    # First, drop the 'NPU_run_profile' rows from 'Layers_df'
    df = Layers_df[df['Metric'] != 'NPU_run_profile']

    # Then, append the updated 'npu_run_profile_with_power' rows back into 'df'
    df = df.append(npu_run_profile_with_power, ignore_index=True)
    '''
    npu_run_profile_with_power['Metric']='task'
    npu_run_profile_with_power=npu_run_profile_with_power[npu_run_profile_with_power['Graph']=='YOLOv3']




    df=df[(df['Component']=='B') & (df['Freq']==7) & (df['Graph']=='YOLOv3')]

    combined_df = pd.concat([df, npu_run_profile_with_power], ignore_index=True)
    if 'Energy' not in combined_df:
        combined_df['Energy']=combined_df['Power']*combined_df['Time']/1000.0
    if 'Power-Efficiency' not in combined_df:
        combined_df['Power-Efficiency']=1000.0/(combined_df['Energy'])


    combined_df.drop(['Freq','Freq_Host','Metric'],axis=1)
    combined_df.to_csv('t.csv')

    pivot = combined_df.pivot_table(index='Layer', columns='Component', values='Power-Efficiency', aggfunc='mean')

    # Ensure that only the components 'B' and 'G' are considered
    #pivot = pivot[['B', 'G']]

    # Plotting
    pivot.plot(kind='line', marker='o')
    pivot.plot(kind='bar', stacked=False, figsize=(10, 5.625))
    plt.xlabel('Layer')
    plt.ylabel('Power-Efficiency')
    plt.title(f'Power-Efficiency')
    plt.legend(title='Component')
    plt.show()
    pivot['Power-Efficiency NPU (Normalized)']=pivot['N']/pivot['B']
    #pivot.to_csv('NPU_Layers.csv')

if Test==4:
    graph_name=['YOLOv3']
    metric=['task']
    comp=['B','N']
    freq_h=[-1,7]
    freq=[7]
    #freq=list(range(8))
    layers=range(80)
    index=['Layer']
    columns=[]
    parameter='Time'
    '''
    .groupby(index+columns)['Time','Power'].sum().reset_index()

        pivot_table = pd.pivot_table(grouped_df, values=parameter, index=index, columns=columns)
        try:
            display(pivot_table)
        except:
            pprint.pprint(pivot_table)
        #pivot_table.plot(kind='bar', stacked=False, figsize=(30, 14))
        pivot_table.plot(kind='bar', stacked=False, figsize=(10, 5.625))'''
    filtered_df = grouped_df[(grouped_df['Graph'].isin(graph_name)) & 
                            (grouped_df['Metric'].isin(metric)) & 
                            (grouped_df['Component'].isin(comp)) & 
                            (grouped_df['Freq_Host'].isin(freq_h))&
                            (grouped_df['Freq'].isin(freq)) &
                            (grouped_df['Layer'].isin(layers)) ].reset_index().drop('index',axis=1)
    #grouped_df[['Component','Power-Efficiency']]

    #t=grouped_df.groupby('Layer').sum()
    pivot = filtered_df.pivot_table(index='Layer', columns='Component', values='Power-Efficiency', aggfunc='mean')

    # Ensure that only the components 'B' and 'G' are considered
    #pivot = pivot[['B', 'G']]
    pivot


def Analyze2(graph_name = 'alex'):
    graph_df = Layers_df[Layers_df['Graph'] == graph_name]
    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    #grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Freq', 'Layer'])['Time'].sum()
    grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Layer', 'Freq'])['Time'].sum().reset_index()
    #print(grouped_df)
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df,index=['Graph', 'Component', 'Layer'], columns='Freq', values='Time')
    # Generate a line plot to visualize the effect of frequency on task timing for different layers
    pivot_table.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title(f'Task Timing vs Frequency for {graph_name}')
    plt.xlabel('Layer')
    plt.ylabel('Task Timing (ms)')
    plt.show()
    return pivot_table
if Test==2:
    Analyze2()


def Explore_Freq_on_Transfering():

    _fs={'BL':[[[0],[i]] for i in range(NFreqs['L'])],
            'LB':[[[0],[i]] for i in range(NFreqs['B'])],
            'GB':[[[0,i],[i]] for i in range(NFreqs['B'])],
            'LG':[[[0],[0,i]] for i in range(NFreqs['B'])],
            'BG':[[[i],[0,i]] for i in range(NFreqs['B'])],
        }
    global Transfer_Freq_df
    if Transfer_Freq_csv.exists():
        Transfer_Freq_df=pd.read_csv(Transfer_Freq_csv)
    else:
        Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq', 'transfer_time', 'transfer_power'])
    kernel_cs=[150]
    kernel_cs=[96*i for i in kernel_cs]
    
    #global trans,trans_pwr,trans_pwr_df,transfer_df_time
    for kernel_c in kernel_cs:
        for order in _fs:
            print(f'order:{order}, kernels:{kernel_c}, shape: {Transfer_Freq_df[(Transfer_Freq_df["order"]==order) & (Transfer_Freq_df["kernels"]==kernel_c)].shape[0]}')
            if Transfer_Freq_df[(Transfer_Freq_df['order']==order) & (Transfer_Freq_df['kernels']==kernel_c)].shape[0]==0:
                try:
                #if True:
                    trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=_fs[order],_kernel_c=kernel_c)
                    #transfer_df_freq.loc[len(transfer_df_freq)] = {"kernels":kernel_c, "c":orders[_c], "transfer_time":transfer_df_time, "transfer_power":trans_pwr_df}
                    trans_pwr_df['freq'] = trans_pwr_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    transfer_df_time['freq'] = transfer_df_time['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    merged_df = pd.merge(trans_pwr_df, transfer_df_time, on=['order', 'freq','SenderFreq','RecFreq'])
                    merged_df['kernels']=kernel_c
                    Transfer_Freq_df=pd.concat([Transfer_Freq_df,merged_df], ignore_index=True)
                    print(f'merged is:\n{merged_df}')
                    print(f'accumulated result is:\n{Transfer_Freq_df}')
                    Transfer_Freq_df.to_csv(Transfer_Freq_csv,index=False)
                    time.sleep(5)
                    #input()
                except Exception as e:
                    print("Error occurred:", e)
                    print("Traceback:")
                    traceback.print_exc()
                    ab()
    first_transfer_time = Transfer_Freq_df.groupby('order')['transfer_time'].first()
    first_transfer_power = Transfer_Freq_df.groupby('order')['transfer_power'].first()
    Transfer_Freq_df['time_ratio'] = Transfer_Freq_df['transfer_time'] / Transfer_Freq_df['order'].map(first_transfer_time)
    Transfer_Freq_df['power_ratio'] = Transfer_Freq_df['transfer_power'] / Transfer_Freq_df['order'].map(first_transfer_power)   
    Transfer_Freq_df.to_csv(Transfer_Freq_csv,index=False)
    plt.plot(Transfer_Freq_df['freq'],Transfer_Freq_df['time_ratio'])
if Test==2:
    Explore_Freq_on_Transfering()


# +
def Plot_Transfer_VS_Data_size(order,freq_mode):
    if freq_mode=="max":
        trans_df=Transfer_Data_Size_Max_Freq_df
    if freq_mode=="min":
        trans_df=Transfer_Data_Size_Min_Freq_df
    
    trans_df['Data-Size']=trans_df['kernels']*729/pow(2,20)
    t=trans_df[trans_df['order']==order].groupby(['kernels']).sum(['transfer_time', 'transfer_power'])
    #p=trans_df[trans_df['c']==order].groupby(['kernels'])['trans_power'].sum()
    print(f'results for {order}:\n{t}')
    #print(p)
    pivot_table_time = pd.pivot_table(t,index=['Data-Size'], values=['transfer_time'])
    pivot_table_time.plot(figsize=(8,6))
    plt.title(f'{order} time vs Data size')
    plt.xlabel('Data-size (MB)')
    plt.ylabel('time (ms)')
    plt.savefig(f'transfertime_vs_datasize_{order}.pdf', format='pdf', dpi=3000)
    plt.show()
    pivot_table_power = pd.pivot_table(t,index=['Data-Size'], values=['transfer_power'])
    pivot_table_power.plot(ylim=[0, 6000],figsize=(5, 3))
    plt.title(f'{order} Power vs Data size')
    plt.xlabel('Data-size (MB)')
    plt.ylabel('Power (mW)')
    plt.show()
    
if Test==4:
    Plot_Transfer_VS_Data_size("BG","min")
# -

# Todaes transfer time
if False or True:
    import seaborn as sns

    label_mapping = {'L': 'Little', 'B': 'Big', 'G': 'GPU'}
    
    # Assuming 'Transfers_df' is your DataFrame
    Transfers_df2=pd.read_csv("Transfers.csv")
    # Filter the DataFrame for the specific graph (MobileV1)
    mobilev1_data = Transfers_df2[Transfers_df2['Graph'] == 'MobileV1']
    # Replace labels in the DataFrame
    mobilev1_data['Src'] = mobilev1_data['Src'].map(label_mapping)
    mobilev1_data['Dest'] = mobilev1_data['Dest'].map(label_mapping)
    mobilev1_data.to_csv('TODAES_TransferMobilenet.csv', index=False)
    # Set font parameters
    sns.set(font_scale=1.5, style='whitegrid', rc={'font.family': 'serif'})

    # Create a Seaborn grouped bar plot with 'Dest' as hue and 'Src' as column
    plt.figure(figsize=(12, 8))
    plot = sns.catplot(x='Layer', y='Time', hue='Dest', col='Src', kind='bar', data=mobilev1_data, alpha=1)


    # Customize the plot
    plt.subplots_adjust(top=0.85)

    # Customize the y-axis label
    plot.set_axis_labels('Layer', 'Time (ms)')

    # We replaced the names in data instead of in plot
    '''# Get the legend object
    legend = plot._legend
    # Replace labels in the legend
    label_mapping = {'L': 'Little', 'B': 'Big', 'G': 'GPU'}
    for text in legend.texts:
        text.set_text(label_mapping[text.get_text()])

        
    # Get the axes and update their titles
    for ax, col in zip(plot.axes.flat, mobilev1_data['Src'].unique()):
        ax.set_title(label_mapping[col])'''


    plt.savefig(f'transfertime_layer.pdf', format='pdf', dpi=3000)
    # Show the plot
    plt.show()

if False or True:
    label_mapping = {'L': 'Little', 'B': 'Big', 'G': 'GPU'}
    # Assuming 'Transfers_df' is your DataFrame
    #Transfers_df2=pd.read_csv("Transfers_1.csv")
    # Filter the DataFrame for the specific graph (MobileV1)
    mobilev1_data = Transfers_df[Transfers_df['Graph'] == 'MobileV1']
    # Replace labels in the DataFrame
    mobilev1_data['Src'] = mobilev1_data['Src'].map(label_mapping)
    mobilev1_data['Dest'] = mobilev1_data['Dest'].map(label_mapping)
    t1=mobilev1_data.pivot_table(index=["Src","Layer"],columns='Dest',values='Time').loc['Little'].dropna(axis=1)
    t2=mobilev1_data.pivot_table(index=["Src",'Dest'],columns=["Layer"],values='Time')
    t3=mobilev1_data.pivot_table(index=["Src",'Dest'],columns=["Layer"],values='Time').transpose()
    pivot_df=mobilev1_data.pivot_table(index="Layer",columns=['Src','Dest'],values='Time')

    display(t1.columns,t1,t2.columns,t2,t3.columns,t3,pivot_df.columns,pivot_df)


    # Reset index to convert 'Layer' to a column

    pivot_df.columns = ['_'.join(map(str, col)) for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    display(pivot_df)
    pivot_df.to_csv('data.csv',index=False)


# set sleep time between tasks to 0 in ARMCL src/graph/detail/ExecuionHelpers.cpp 
#(check graphmanager.cpp for sure that there is no sleep )
def Explore_Data_Size_on_Transfering(freq_mode="max"):
    global Transfer_Data_Size_Max_Freq_df, Transfer_Data_Size_Min_Freq_df
    g="test_transfer"
    _fs={"max":{"BL":[ [ [0],[5] ] ],
                "LB":[ [ [0],[7] ] ],
                "GB":[ [ [0,7],[7] ] ],
                "LG":[ [ [0],[0,7] ] ],
                "BG":[ [ [7],[0,7] ] ]},
         "min":{"BL":[ [ [0],[0] ] ],
                "LB":[ [ [0],[0] ] ],
                "GB":[ [ [0,0],[0] ] ],
                "LG":[ [ [0],[0,0] ] ],
                "BG":[ [ [0],[0,0] ] ]}
        }
    if freq_mode=="max":
        if Transfer_Data_Size_Max_Freq_csv.exists():
            Transfer_Data_Size_Max_Freq_df=pd.read_csv(Transfer_Data_Size_Max_Freq_csv)
            print(f'max freq trans data:\n{Transfer_Data_Size_Max_Freq_df}')
        else:
            Transfer_Data_Size_Max_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
        kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
        kernel_cs=[96*i for i in kernel_cs]

        for kernel_c in kernel_cs:
            for order in _fs["max"]:
                if Transfer_Data_Size_Max_Freq_df[(Transfer_Data_Size_Max_Freq_df['order']==order) & 
                                                  (Transfer_Data_Size_Max_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if order[1]=='G' and kernel_c > 150*96:
                            continue
                                                    
                        ff=_fs["max"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=ff,_kernel_c=kernel_c)
                        Transfer_Data_Size_Max_Freq_df.loc[len(Transfer_Data_Size_Max_Freq_df)] = {"kernels":kernel_c, "order":order, "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                        Transfer_Data_Size_Max_Freq_df.to_csv(Transfer_Data_Size_Max_Freq_csv,index=False)
                        time.sleep(10)
                    except:
                        ab()               

        return Transfer_Data_Size_Max_Freq_df
    
    if freq_mode=="min":
        if os.path.isfile(Transfer_Data_Size_Min_Freq_csv):
            Transfer_Data_Size_Min_Freq_df=pd.read_csv(Transfer_Data_Size_Min_Freq_csv)
            print(f'min freq trans data:\n{Transfer_Data_Size_Min_Freq_df}')
            #return transfer_df_min
        else:
            Transfer_Data_Size_Min_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
        kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
        kernel_cs=[96*i for i in kernel_cs]


        for kernel_c in kernel_cs:
            for order in _fs["min"]:
                if Transfer_Data_Size_Min_Freq_df[(Transfer_Data_Size_Min_Freq_df['order']==order) & 
                                                  (Transfer_Data_Size_Min_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if order[1]=='G' and kernel_c > 150*96:
                            continue
                        ff=_fs["min"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=ff,_kernel_c=kernel_c)
                        Transfer_Data_Size_Min_Freq_df.loc[len(Transfer_Data_Size_Min_Freq_df)] = {"kernels":kernel_c, "order":order, "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                        Transfer_Data_Size_Min_Freq_df.to_csv(Transfer_Data_Size_Min_Freq_csv,index=False)
                        time.sleep(10)
                    except:
                        ab()               

        return Transfer_Data_Size_Min_Freq_df


# +
def Run_Explore_Data_Size_on_Transfering(_freq_mode="max"):
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    Explore_Data_Size_on_Transfering(freq_mode=_freq_mode)
    for i in orders:
        Plot_Transfer_VS_Data_size(order=orders[i],freq_mode=_freq_mode)
        
if Test==4:
    #Run_Explore_Data_Size_on_Transfering(_freq_mode="max")
    Run_Explore_Data_Size_on_Transfering(_freq_mode="min")


# -

def Compute_Layer_Percentage():
#if True:
    sum_time_per_graph_component = Layers_df.groupby(['Graph', 'Component', 'Freq', 'Freq_Host'])['Time'].sum().reset_index()
    pd.set_option('display.max_rows', 1000)
    Layers_With_Percentage_df=Layers_df.merge(sum_time_per_graph_component, on=['Graph', 'Component', 'Freq', 'Freq_Host'], suffixes=('', '_sum'))
    Layers_With_Percentage_df['Time_Percentage'] = Layers_With_Percentage_df['Time'] / Layers_With_Percentage_df['Time_sum'] * 100
    #print(Layers_With_Percentage_df[(Layers_With_Percentage_df["Graph"]=="alex") & (Layers_With_Percentage_df["Freq"]==0) & (Layers_With_Percentage_df["Component"]=="G") & (Layers_With_Percentage_df["Freq_Host"]==0)]["Time_Percentage"].sum())
    Layers_With_Percentage_df.to_csv(Layers_With_Percentage_csv, index=False)
    Layers_Percentage_df=Layers_With_Percentage_df.groupby(['Graph', 'Component','Layer','Metric'])['Time_Percentage'].mean().reset_index()
    #print(Layers_Percentage_df)
    #Layers_Percentage_df.to_csv(Layers_Percentage_csv, index=False)
    pivot_df = Layers_Percentage_df.pivot_table(index=['Graph', 'Layer', 'Metric'], columns='Component', values='Time_Percentage')
    pivot_df.columns = ['Time_Percentage_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    pivot_df['Time_Percentage_Average'] = pivot_df[['Time_Percentage_B', 'Time_Percentage_G', 'Time_Percentage_L']].mean(axis=1)
    
    pivot_df = pivot_df.groupby(['Graph', 'Layer']).sum().reset_index()
    pivot_df.to_csv(Layers_Percentage_csv, index=False)
    try:
        display(pivot_df)
    except:
        pprint.pprint(pivot_df)


# +
## plot energy of layers running with different components with freq min
def _Analyze_Components(g=['alex']):
    Layers_df['Energy']=Layers_df['Time']*Layers_df['Power']/1000.0
    grouped_df = Layers_df[(Layers_df['Graph'].isin(g)) & 
                        (Layers_df['Metric'].isin(['in','task'])) & 
                        (Layers_df['Freq'].isin(range(10))) & 
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(100))) ].groupby(['Component','Layer','Metric'])\
                        ['Time','Power','Energy'].mean().reset_index()

    
    display(grouped_df)
    #display(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Component', 'Layer']).agg(aggregations).reset_index()
    #display(grouped_df)
    grouped_df['Power-Efficiency']=1000.0/grouped_df['Energy']

    pivot_df = grouped_df.pivot_table(index=['Layer'], columns='Component', values=['Time', 'Energy', 'Power-Efficiency'])
    pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    try:
        display(pivot_df)
    except:
        pprint.pprint(pivot_df)
        
    pivot_df.to_csv('TODAES_Layers_Processors.csv',index=False)
        
    _fontsize = 18
    _fontsize2 = 14
    _figsize=(16,3.5)
    
    legend=['NPU','GPU','big CPU', 'Little CPU']
    PE_cols = ['Power-Efficiency_N','Power-Efficiency_G', 'Power-Efficiency_B', 'Power-Efficiency_L']
    #pivot_table.plot(kind='bar', stacked=False, figsize=(10, 5.625))
    
    energy_plot = pivot_df.plot(x='Layer', y=PE_cols, kind='bar',figsize=_figsize)#,figsize=(10, 5.625)
    energy_plot.set_xlabel('Layer',fontsize=_fontsize2)#, fontsize=font_size-2
    energy_plot.set_ylabel('Power-Efficiency FPS/Watt',fontsize=_fontsize2)#, fontsize=font_size-2
    # Set the y-axis limit to make space for the legend
    plt.ylim(0, 205)  # Adjust the upper limit as needed
    energy_plot.legend(legend,ncol=len(legend),fontsize=_fontsize2)
    plt.xticks(fontsize=_fontsize2-2)
    plt.yticks(fontsize=_fontsize2-2)
    #plt.title('Power-Efficiency of {}net layers for Average Freqs'.format(g[0]))
    '''energy_plot.title.set_fontsize(font_size)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.legend(fontsize=font_size-2)'''
    
    '''energy_plot = pivot_df.plot(x='Layer', y=PE_cols, kind='bar', title='Energy-Efficiency of {}net layers for Average Freqs'.format(g[0]))
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy-Efficiency FPS/Watt')
    legend = energy_plot.legend()
    for label in legend.get_texts():
        label.set_text(label.get_text().replace('Power-Efficiency', 'Energy-Efficiency'))'''
    plt.tight_layout()
    plt.savefig(f'Layers_P_PE.pdf', format='pdf', dpi=3000)
    plt.show()
        
    energy_cols = ['Energy_N','Energy_G', 'Energy_B', 'Energy_L']
    energy_plot = pivot_df.plot(x='Layer', y=energy_cols, kind='bar',figsize=_figsize)
    #plt.title(title='{} Energy for Average Freqs'.format(g[0]))
    energy_plot.set_xlabel('Layer',fontsize=_fontsize2)
    energy_plot.set_ylabel('Energy (mj)',fontsize=_fontsize2)
    energy_plot.legend(legend,fontsize=_fontsize2)
    plt.xticks(fontsize=_fontsize2-2)
    plt.yticks(fontsize=_fontsize2-2)
    plt.tight_layout()
    plt.savefig(f'Layers_P_Energy.pdf', format='pdf', dpi=3000)
    plt.show()

    # Plot Time columns
    time_cols = ['Time_N','Time_G', 'Time_B', 'Time_L']
    time_plot = pivot_df.plot(x='Layer', y=time_cols, kind='bar',figsize=_figsize)
    #plt.title(title='{} Time for Average Freqs'.format(g[0]))
    time_plot.set_xlabel('Layer',fontsize=_fontsize2)
    time_plot.set_ylabel('Time (mw)',fontsize=_fontsize2)
    time_plot.legend(legend,fontsize=_fontsize2)
    plt.xticks(fontsize=_fontsize2-2)
    plt.yticks(fontsize=_fontsize2-2)
    plt.tight_layout()
    plt.savefig(f'Layers_P_Time.pdf', format='pdf', dpi=3000)
    plt.show()

#figure of TODAES (Execution time (a), energy consumption (b), and power efficiency (c) of MobileNetV1 layers on NPU, GPU, big CPU and Little CPU.)
if Test==4:
    _Analyze_Components(g=['MobileV1'])
    
# -

## plot (and extract and save result to csv files) energy of layers running with different components with freq min
def Analyze_Components(g=['alex']):

    grouped_df = Layers_df[(Layers_df['Graph'].isin(g)) & 
                        (Layers_df['Metric'].isin(['in','task'])) &  
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(10))) ].groupby(['Freq','Component','Layer','Metric'])\
                        ['Time','Power'].sum().reset_index()

    grouped_df['Energy']=grouped_df['Time']*grouped_df['Power']/1000.0
    
    #print(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Freq','Component', 'Layer']).agg(aggregations).reset_index()
    #print(grouped_df)

    grouped_df['Power-Efficiency']=1000.0/grouped_df['Energy']
    
    pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values=['Time', 'Energy','Power-Efficiency'])
    pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
    #pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values='Energy')
    #pivot_df.columns = ['Energy_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    #print(pivot_df)
    
    
   # Group by 'Layer' and get the maximum valid frequency for each parameter
    max_freq_B = pivot_df[pivot_df['Energy_B'].notna()].groupby('Layer')['Freq'].max()
    max_freq_G = pivot_df[pivot_df['Energy_G'].notna()].groupby('Layer')['Freq'].max()
    max_freq_L = pivot_df[pivot_df['Energy_L'].notna()].groupby('Layer')['Freq'].max()

    # Extract the values at the maximum valid frequency for each parameter
    freq_df = pd.DataFrame({
        'Layer': max_freq_B.index,
        'Energy_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Energy_B'].values,
        'Time_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Time_B'].values,
        'Energy_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Energy_G'].values,
        'Time_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Time_G'].values,
        'Energy_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Energy_L'].values,
        'Time_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Time_L'].values,
        'Power-Efficiency_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Power-Efficiency_L'].values,
        'Power-Efficiency_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Power-Efficiency_B'].values,
        'Power-Efficiency_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Power-Efficiency_G'].values,
    })
    try:
        display(freq_df)
    except:
        pprint.pprint(freq_df)
    energy_cols = ['Energy_G_MaxFreq', 'Energy_B_MaxFreq', 'Energy_L_MaxFreq']
    energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Energy for Freq Max')
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy')
    plt.show()
    
    energy_cols = ['Power-Efficiency_G_MaxFreq', 'Power-Efficiency_B_MaxFreq', 'Power-Efficiency_L_MaxFreq']
    energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Power-Efficiency for Freq Max')
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy')
    plt.show()

    # Plot Time columns
    time_cols = ['Time_G_MaxFreq', 'Time_B_MaxFreq', 'Time_L_MaxFreq']
    time_plot = freq_df.plot(x='Layer', y=time_cols, kind='bar', title='Time for Freq Max')
    time_plot.set_xlabel('Layer')
    time_plot.set_ylabel('Time')
    plt.show()
    
    for freq in pivot_df['Freq'].unique():
        # Filter dataframe for the current Freq value
        freq_df = pivot_df[pivot_df['Freq'] == freq]
        try:
            display(freq_df)
        except:
            pprint.pprint(freq_df)
        
        # Plot Energy-Efficiency columns
        energy_efficiency_cols = ['Power-Efficiency_G', 'Power-Efficiency_B', 'Power-Efficiency_L']
        energy_plot = freq_df.plot(x='Layer', y=energy_efficiency_cols, kind='bar', title='Power-Efficiency for Freq {}'.format(freq))
        energy_plot.set_xlabel('Layer')
        energy_plot.set_ylabel('Power-Efficiency FPS/Watt')
        plt.show()

        # Plot Energy columns
        energy_cols = ['Energy_G', 'Energy_B', 'Energy_L']
        energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Energy for Freq {}'.format(freq))
        energy_plot.set_xlabel('Layer')
        energy_plot.set_ylabel('Energy')
        plt.show()

        # Plot Time columns
        time_cols = ['Time_G', 'Time_B', 'Time_L']
        time_plot = freq_df.plot(x='Layer', y=time_cols, kind='bar', title='Time for Freq {}'.format(freq))
        time_plot.set_xlabel('Layer')
        time_plot.set_ylabel('Time')
        plt.show()
if Test==4:
    Analyze_Components(g=['Alex'])

# +
# Todaes DSD CPU-GPU layer switch:


## Extract time data of just max freq
def extract_max_freqs():
    #remove metric (and also extract just time (skip power and energy))
    #When you want to print a goupby:
    #df=Layers_df.groupby(['Graph', 'Component','Freq','Freq_Host','Layer'])['Time'].apply(print)
    df=Layers_df.groupby(['Graph', 'Component','Freq','Freq_Host','Layer'])['Time'].sum().reset_index()
    # Group by 'Graph', 'Component', 'Layer', and 'Metric' and get the row indices with maximum 'Freq'
    max_freq_indices = df.groupby(['Graph', 'Component', 'Layer','Freq_Host'])['Freq'].idxmax()
    # Select the rows with maximum 'Freq'
    max_freq_rows = df.loc[max_freq_indices]
    # Group by 'Graph', 'Component', 'Layer' again on the subset of rows with maximum 'Freq'
    # Then get the row indices with maximum 'Freq_Host' within each group
    max_freq_host_indices = max_freq_rows.groupby(['Graph', 'Component', 'Layer'])['Freq_Host'].idxmax()

    # Select the rows with maximum 'Freq_Host' among the rows with maximum 'Freq'
    max_rows = max_freq_rows.loc[max_freq_host_indices]
    #now drop the Freq and Freq_Host --> the columns are Graph, Component, Layer, Time
    max_rows.drop(columns=['Freq','Freq_Host'],inplace=True)
    return max_rows

## Best component for each layer
def min_config(df=None):
    if df is None:
        #remove metric (and also extract just time (skip power and energy))
        df=Layers_df.groupby(['Graph', 'Component','Freq','Freq_Host','Layer'])['Time'].sum().reset_index()
    # Group by 'Graph', 'Metric', and 'Layer', then find the minimum 'Time' value
    min_time_per_layer = df.groupby(['Graph', 'Layer'])['Time'].min()
    #display(min_time_per_layer)
    #display(min_time_per_layer.groupby('Graph').sum())
    min_time_per_layer_idx = df.groupby(['Graph', 'Layer'])['Time'].idxmin()
    best_config=df.loc[min_time_per_layer_idx]
    #display(best_config.query("Graph=='Alex'"))
    return best_config

# Table of components and min for each layer
def table_min_component(df=None):
    if df is None:
        df=extract_max_freqs()
    reshape_df=df.pivot_table(index=['Graph','Layer'],columns=['Component'],values=['Time'])
    # Compute the minimum 'Time' value across all components for each group
    reshape_df[('Time', 'Min')] = reshape_df['Time'].min(axis=1)
    reshape_df.dropna(axis=1,inplace=True)
    #display(reshape_df)
    return reshape_df
    
# min sum for each (graph,comp)
def min_sum():
    #remove metric (and also extract just time (skip power and energy))
    df=Layers_df.groupby(['Graph', 'Component','Freq','Freq_Host','Layer'])['Time'].sum().reset_index()
    #Sum of layers for each group of (graph,comp,Freq,Freq_Host)
    sum_time_per_group = df.groupby(['Graph', 'Component', 'Freq', 'Freq_Host'])['Time'].sum().reset_index()
    #Min of sum for each component (which means the best freq for each component)
    min_sum_time_per_group=sum_time_per_group.groupby(['Graph', 'Component'])['Time'].min().reset_index()
    return min_sum_time_per_group


# sum of mins for each (graph,comp)
def sum_min():
    #remove metric (and also extract just time (skip power and energy))
    df=Layers_df.groupby(['Graph', 'Component','Freq','Freq_Host','Layer'])['Time'].sum().reset_index()
    #min of each layer for each (group,comp,layer) which means the best freq setting for it
    min_time_per_group=df.groupby(['Graph', 'Component','Layer'])['Time'].min().reset_index()
    # sum of layers for these mins for each (graph,comp)
    sum_min_time_per_group=min_time_per_group.groupby(['Graph', 'Component'])['Time'].sum().reset_index()
    return sum_min_time_per_group


# +
if Test==5:
    df_max_freq=extract_max_freqs()
    df_min_config=min_config(df_max_freq)
    df_min_table=table_min_component(df_max_freq)
    df_min_sum=min_sum()
    df_sum_min=sum_min()
    target_graph="SqueezeV1"
    #display("max_freq:",df_max_freq.query(f"Graph=='{target_graph}'"))
    display("min config:",df_min_config.query(f"Graph=='{target_graph}'"),df_min_config.query(f"Graph=='{target_graph}'")[['Component','Time']].sum())
    display("table:",df_min_table.query(f"Graph=='{target_graph}'").reset_index())
    "table:",df_min_table.query(f"Graph=='{target_graph}'").reset_index().to_csv('f.csv')
    display("min sum:",df_min_sum.query(f"Graph=='{target_graph}'"))
    display("sum min:",df_sum_min.query(f"Graph=='{target_graph}'"))
    _order=df_min_config.query(f"Graph=='{target_graph}'")['Component'].sum()
    _order='GGGGGGGGLL'
    pred=predict_cost.Inference_Cost(_graph=target_graph,_order=_order,_freq='max',_debug=True)
    print(pred)
    real_eval.Real_Evaluation(g=target_graph,_ord=_order,_fs=[ '{{max}}' ],suffix='',gpu_host='B', npu_host='B')
    for orde in ['B','L','G']:
        _order=orde*NLayers[target_graph]
        print('\n\n\n***************************\n***************************\n')
        pred=predict_cost.Inference_Cost(_graph=target_graph,_order=_order,_freq='max',_debug=False)
        print(pred)
        real_eval.Real_Evaluation(g=target_graph,_ord=_order,_fs=[ '{{max}}' ],suffix='',gpu_host='B', npu_host='B')
    
    
    
    
'''for orde in ['B','L','G']:
    if orde=='G':
        freq=[[4,7]]*NLayers[target_graph]
    if orde=='L':
        freq=[[5]]*NLayers[target_graph]
    if orde=='B':
        freq=[[7]]*NLayers[target_graph]
    _order=orde*NLayers[target_graph]
    pred=predict_cost.Inference_Cost(_graph=target_graph,_order=_order,_freq=freq,_debug=False)
    print(pred)
    real_eval.Real_Evaluation(g=target_graph,_ord=_order,_fs=[ freq ],suffix='',gpu_host='B', npu_host='B')'''

        
# -


