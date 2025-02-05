import pandas as pd

# Load the data
file_path = 'Data/L.csv'  # replace with your actual file path
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
data['Energy'] = data['Time'] * data['Power']
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

# Save the final DataFrame to a new CSV file
output_file_path = 'Data/L2.csv'
pivot_table.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")
