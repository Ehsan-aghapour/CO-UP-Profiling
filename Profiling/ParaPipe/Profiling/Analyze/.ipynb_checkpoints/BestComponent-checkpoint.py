import pandas as pd

# Load the data
file_path = 'Layers.csv'  # replace with your actual file path
data = pd.read_csv(file_path)

data

# +
# Create a mapping for the 'in' metric Power values
in_power_mapping = data[data['Metric'] == 'in'].set_index(['Graph', 'Freq', 'Freq_Host', 'Component'])['Power']

# Function to fill the 'out' metric Power values
def fill_out_power(row):
    if pd.isna(row['Power']) and row['Metric'] == 'out':
        return in_power_mapping.get((row['Graph'], row['Freq'], row['Freq_Host'], row['Component']), row['Power'])
    return row['Power']

# Apply the function to fill missing 'out' metric Power values
data['Power'] = data.apply(fill_out_power, axis=1)
# -

# Calculate Energy = Time * Power for each row
data['Energy'] = data['Time'] * data['Power'] / 1000
data

# Pivot the table to get separate columns for each component's Time, Power, and Energy
pivot_table = data.pivot_table(index=['Graph', 'Freq', 'Freq_Host', 'Layer', 'Metric'],
                               columns='Component',
                               values=['Time', 'Power', 'Energy'],
                               aggfunc='first').reset_index()

pivot_table

# Flatten the MultiIndex columns
pivot_table.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_table.columns.values]

pivot_table



pivot_table['Min_Time'] = pivot_table[['Time_B', 'Time_G', 'Time_L']].min(axis=1)
pivot_table['Min_Time_Component'] = pivot_table[['Time_B', 'Time_G', 'Time_L']].idxmin(axis=1).apply(lambda x: x.split('_')[1] if isinstance(x, str) else 'Unknown')
pivot_table['Min_Energy'] = pivot_table[['Energy_B', 'Energy_G', 'Energy_L']].min(axis=1)
pivot_table['Min_Energy_Component'] = pivot_table[['Energy_B', 'Energy_G', 'Energy_L']].idxmin(axis=1).apply(lambda x: x.split('_')[1] if isinstance(x, str) else 'Unknown')


pivot_table

# +
# Define custom sort order for 'Metric'
metric_order = ['in', 'task', 'out']
pivot_table['Metric'] = pd.Categorical(pivot_table['Metric'], categories=metric_order, ordered=True)

# Sort the DataFrame by 'Graph', 'Layer', and 'Metric'
pivot_table = pivot_table.sort_values(by=['Graph', 'Layer', 'Metric']).reset_index(drop=True)

# Inspect the resulting DataFrame
print(pivot_table)

# -

# Save the final DataFrame to a new CSV file
output_file_path = 'Layers_analyzed.csv'
pivot_table.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")


# +
def calculate_time_energy(graph, components, freq="max", freq_host="max",log=True):
    total_time = 0
    total_energy = 0
    
    # Filter pivot_table based on graph, freq, freq_host
    filtered_data = pivot_table[(pivot_table['Graph'] == graph) & 
                                (pivot_table['Freq'] == freq) & 
                                (pivot_table['Freq_Host'] == freq_host)]
    
    # Iterate over each component in the string
    for layer, comp in enumerate(components):
        Time_column = "Time_" + comp
        Energy_column = "Energy_" + comp
        total_time += filtered_data.loc[filtered_data['Layer'] == layer, Time_column].sum()
        total_energy += filtered_data.loc[filtered_data['Layer'] == layer, Energy_column].sum()
      
    if log:
        print(f'{graph}:{components}')
        print(f"Total Time Sum: {total_time}")
        print(f"Total Energy Sum: {total_energy}")
        #print(total_time,total_energy)
        print("\n************************************\n")
    return total_time,total_energy

# Example usage:
graph_name = 'InceptionResnetV2'
components_string = 'GGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'

time_sum, energy_sum = calculate_time_energy(graph_name, components_string)


# -

def print_columns(t_sum,e_sum):
    print("\n------------\n")
    for t in t_sum:
        print(t)
    print("\n------------\n")
    for e in e_sum:
        print(e)
    print("\n------------\n")
    for i,t in enumerate(t_sum):
        print(t,e_sum[i])
    print("\n------------\n")


# +
#Yujie Parapipe collaboration

graph_name = 'Google'
mappings = ['GGGGBBBBBBG', 'GGGGBBBBBBB', 'B'*11, 'G'*11] 
t_sum,e_sum=[],[]
for mapping in mappings:
    t, e = calculate_time_energy(graph_name, mapping)
    t_sum.append(t)
    e_sum.append(e)
print_columns(t_sum,e_sum)

    
graph_name = 'InceptionResnetV2'
mappings = ['GGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', 'B'*50, 'G'*50, 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBG', 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBB'] 
t_sum,e_sum=[],[]
for mapping in mappings:
    t, e = calculate_time_energy(graph_name, mapping)
    t_sum.append(t)
    e_sum.append(e)
print_columns(t_sum,e_sum)

graph_name='InceptionV3'
mappings = ['GGGGGGGGBBBBBBBBG', 'GGGGGGGGBBBBBBBBB', 'B'*17, 'G'*17]
t_sum,e_sum=[],[]
for mapping in mappings:
    t, e = calculate_time_energy(graph_name, mapping)
    t_sum.append(t)
    e_sum.append(e)
print_columns(t_sum,e_sum)

graph_name='InceptionV4'
mappings = ['GGGBGBGGGGBBBBBBBBBBBBB', 'GGGGGBGGGGBBBBBBBBBBBBB', 'GGGBGGGGGGBBBBBBBBBBBBB', 'GGGGGGGGGGBBBBBBBBBBBBB', 'B'*23, 'G'*23, 'GGGGGGGGGGGGGGGGGGGGGGB']
t_sum,e_sum=[],[]
for mapping in mappings:
    t, e = calculate_time_energy(graph_name, mapping)
    t_sum.append(t)
    e_sum.append(e)
print_columns(t_sum,e_sum)

# -


