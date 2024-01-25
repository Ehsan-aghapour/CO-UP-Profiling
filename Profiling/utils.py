

# +
import subprocess
import time 




### ab command for checking the board connection and make root access
def ab():
    rr='ab'
    print(f'Command is: {rr}')
    p = subprocess.Popen(rr.split())
    p.communicate()
    while(p.returncode):
        print('ab not successful next try after 10s ...')
        time.sleep(10)
        p = subprocess.Popen(rr.split())
        p.communicate()  


# +
### Convert freqs list to string
def format_freqs(fs=[ [ [7],[6],[4],[3,6],[4],[5],[6],[7] ], [] ]):
        formated_fs=[]
        for f in fs:
            if f[0]=="min":
                formated_fs.append(f)
                continue
            if type(f)==str:
                f=[[int(j) for j in re.findall(r"\b\d+\b", l)] for l in f.split('),')]
            ff = '-'.join(['[' + str(sublist[0]) + ',' + str(sublist[1]) + ']' if len(sublist) > 1 else str(sublist[0]) for sublist in f])
            #print(ff)
            formated_fs.append(ff)
        return formated_fs


def format_to_list(fs):
    formated_fs=[]
    for f in fs:
        t=[[int(j) for j in re.findall(r"\b\d+\b", l)] for l in f.split('),')]
        formated_fs.append(t)
    return formated_fs







### Extract the value of a parameter in a specific row in pandas dataframe
def Value(graph,comp,freq,layer,metric,attr):
    #print(graph,comp,freq,layer,metric,attr)
    #print(Layers_df_indexed)
    #global Layers_df_indexed
    '''if Layers_df_indexed.shape[0]==0:
        Layers_df_indexed = Layers_df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])'''
        
    index_base = (graph, comp, freq[0], layer)
    if len(freq)==1 or comp!='G':
        if comp=='N':
            if attr=='Power':
                if metric=='in':
                    return Layers_df_indexed.loc[(*index_base, 'in', freq[0]), attr][0]
                else:
                    if(layer in [57,58,65,66]):
                        return 5707
                    return Layers_df_indexed.loc[(*index_base, 'task', freq[0]), attr][0]
            if attr=='Time':
                if metric=='task' or metric=="run":
                    return Layers_df_indexed.loc[(*index_base, 'NPU_run_profile', freq[0]), attr][0]
                if metric=='load':
                    return Layers_df_indexed.loc[(*index_base, 'NPU_in', freq[0]), attr][0]
                if metric=='unload':
                    t=(Layers_df_indexed.loc[(*index_base, 'NPU_run', freq[0]), attr][0] -  Layers_df_indexed.loc[(*index_base, 'NPU_run_profile', freq[0]), attr][0])
                    t+=Layers_df_indexed.loc[(*index_base, 'NPU_out', freq[0]), attr][0]
                    return t
                
                if metric=='in':
                    return Layers_df_indexed.loc[(graph, comp, freq[0], layer, 'in', freq[0]), attr][0]
                
                if metric=='out':
                    #print(f'getting value: {graph}, {comp},{layer}')
                    return Layers_df_indexed.loc[(*index_base, 'out', freq[0]), attr][0]
                
                if metric=='transfer':
                    return Layers_df_indexed.loc[(*index_base, 'transfer', freq[0]), attr][0]
                
        else:
            return Layers_df_indexed.loc[(*index_base, metric, -1), attr][0]
    if len(freq)==2:
        return Layers_df_indexed.loc[(*index_base, metric, freq[1]), attr][0]
    else:
        return -1

def Value2(graph, comp, freq, layer, metric, attr):
    print(graph, comp, freq, layer, metric, attr)
    try:
        # Ensure freq is a list for consistent processing
        if not isinstance(freq, list):
            freq = [freq]

        # Common index base
        index_base = (graph, comp, freq[0], layer)

        # Different cases based on 'comp', 'metric', and 'attr'
        if comp == 'N':
            if attr == 'Power':
                if metric in ['in', 'task']:
                    return Layers_df_indexed.at[(*index_base, metric, freq[0]), attr]
                elif layer in [57, 58, 65, 66]:
                    return 5707
            elif attr == 'Time':
                if metric in ['task', 'run', 'load', 'in']:
                    return Layers_df_indexed.at[(*index_base, 'NPU_' + metric, freq[0]), attr]
                elif metric == 'unload':
                    t = (Layers_df_indexed.at[(*index_base, 'NPU_run', freq[0]), attr] -
                         Layers_df_indexed.at[(*index_base, 'NPU_run_profile', freq[0]), attr])
                    t += Layers_df_indexed.at[(*index_base, 'NPU_out', freq[0]), attr]
                    return t
                elif metric in ['out', 'transfer']:
                    return Layers_df_indexed.at[(*index_base, metric, freq[0]), attr]
        else:
            return Layers_df_indexed.at[(*index_base, metric, freq[1] if len(freq) > 1 else -1), attr]
    except KeyError:
        # Handle cases where the index is not found
        print(f"Key not found for index: {index_base}")
        return None
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return None
    




# +
### counts the number of consequitive N in a string
def count_consecutive_N(cmps, i):
    # Check if the character at index i is 'N'
    if cmps[i] != 'N':
        return 0

    # Initialize count
    count = 1  # Count the 'N' at index i

    # Count consecutive 'N's to the left of i
    left_index = i - 1
    while left_index >= 0 and cmps[left_index] == 'N':
        count += 1
        left_index -= 1

    # Count consecutive 'N's to the right of i
    right_index = i + 1
    while right_index < len(cmps) and cmps[right_index] == 'N':
        count += 1
        right_index += 1

    return count




def generate_random_strings(_n, num_strings):
    chars = ['L', 'B', 'G']
    random_strings = []
    for _ in range(num_strings):
        random_string = ''.join(random.choice(chars) for _ in range(_n))
        random_strings.append(random_string)
    return random_strings
#random_strings = generate_random_strings(8, 100)



### DAC: for figures that shows the effect of performance of N for different layers
def generate_strings_with_changes_and_step(base_string, change_char='N', num_changes=2, step=5,):
    """
    Generate strings by changing a configurable number of characters in the base string
    to a specified character, with a constraint that the indices of changed characters
    must be a multiple of the step.

    :param base_string: The original string.
    :param change_char: The character to change to.
    :param num_changes: Number of characters to change.
    :param step: The indices of changes must be a multiple of this step.
    :return: List of generated strings.
    """
    from itertools import combinations

    generated_strings = []
    base_length = len(base_string)

    # Only consider indices that are multiples of the step
    eligible_indices = [i for i in range(base_length) if i % step == 0]

    for change_indices in combinations(eligible_indices, num_changes):
        new_string = list(base_string)
        for index in change_indices:
            new_string[index] = change_char
        generated_strings.append("".join(new_string))

    return generated_strings



