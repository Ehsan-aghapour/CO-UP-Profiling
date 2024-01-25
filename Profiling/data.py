# +
from config import *


Layers_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time", "Power"])
Layers_df_indexed=pd.DataFrame()
Transfers_df=pd.DataFrame(columns=["Graph", "Layer", "Dest", "Src", "Time"])
Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq' 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Min_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Max_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])
Freq_Transition_Dealy_df=None

# +
initial_loading=None
def Load_Data():
    global Layers_df, Transfers_df, Transfer_Freq_df,\
        Transfer_Data_Size_Min_Freq_df,Transfer_Data_Size_Max_Freq_df,Layers_df_indexed,Freq_Transition_Dealy_df
    #### Load data of layer times with different freqs
    if Layers_csv.exists():
        Layers_df=pd.read_csv(Layers_csv)
    
    # set index to enable access with list of indexes (in value function)
    Layers_df_indexed = Layers_df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])
    
    #### Load transfer times of real layers
    if Transfers_csv.exists():
        Transfers_df=pd.read_csv(Transfers_csv)

       
    ### Load time and power of tranfer with syntethic layers for different layers
    if Transfer_Freq_csv.exists():
        Transfer_Freq_df=pd.read_csv(Transfer_Freq_csv)
    if Transfer_Freq_df.shape[0]:
        first_transfer_time = Transfer_Freq_df.groupby('order')['transfer_time'].first()
        first_transfer_power = Transfer_Freq_df.groupby('order')['transfer_power'].first()
        Transfer_Freq_df['time_ratio'] = Transfer_Freq_df['transfer_time'] / Transfer_Freq_df['order'].map(first_transfer_time)
        Transfer_Freq_df['power_ratio'] = Transfer_Freq_df['transfer_power'] / Transfer_Freq_df['order'].map(first_transfer_power)   
    
    ### Load tranfering VS data size with min freq
    if Transfer_Data_Size_Min_Freq_csv.exists():
        Transfer_Data_Size_Min_Freq_df=pd.read_csv(Transfer_Data_Size_Min_Freq_csv)
    
    ### Load tranfering VS data size with max freq
    if Transfer_Data_Size_Max_Freq_csv.exists():
        Transfer_Data_Size_Max_Freq_df=pd.read_csv(Transfer_Data_Size_Max_Freq_csv)
        
    ## Loading frequency transmition delay times 
    if Freq_Transition_Dealy_csv.exists():
        Freq_Transition_Dealy_df = pd.read_csv(Freq_Transition_Dealy_csv)
        Freq_Transition_Dealy_df.replace({'Little': 'L', 'Big': 'B', 'GPU': 'G'}, inplace=True)

    Layers_df_indexed = Layers_df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])

    
        
#if Test:
if not initial_loading:
    print("Loading Data (initialization)")
    Load_Data()
    initial_loading=True


