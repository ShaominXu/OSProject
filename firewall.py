from scipy.io import arff
import pandas as pd

import config as cfg

def firewall_normal(data_path):
    # Load ARFF file
    data = arff.loadarff(data_path)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data[0])

    # Convert 'result' column values from bytes to string and then to numeric
    df['result'] = pd.to_numeric(df['result'].str.decode('utf-8'), errors='coerce')

    # Filter rows where 'result' is '0'
    result_0_data = df[df['result'] == 0]

    # Find max and min values of 'time' and 'crc_rate' in the filtered data
    max_time_result_0 = result_0_data['time'].max()
    min_time_result_0 = result_0_data['time'].min()

    max_crc_rate_result_0 = result_0_data['crc_rate'].max()
    min_crc_rate_result_0 = result_0_data['crc_rate'].min()

    # Print the results
    print(f"Max time when result is 0: {max_time_result_0}")
    print(f"Min time when result is 0: {min_time_result_0}")
    print(f"Max crc_rate when result is 0: {max_crc_rate_result_0}")
    print(f"Min crc_rate when result is 0: {min_crc_rate_result_0}")

    min_time_normal = 1.0
    max_time_normal = 1.3
    max_crc_rate_normal = 1
    min_crc_rate_normal = 0

    # Define the ranges for normal conditions
    normal_time_range = ( min_time_normal, max_time_normal)  # Replace with your desired range
    normal_crc_rate_range = (max_crc_rate_normal, min_crc_rate_normal)  # Replace with your desired range

    # Filter rows for normal conditions
    normal_commands = df[
        (df['time'] >= normal_time_range[0]) & (df['time'] <= normal_time_range[1])
        ]

    # Filter rows for attack conditions
    attack_commands = df.drop(normal_commands.index)

    # Print or use the normal_commands and attack_commands dataframes as needed
    print("Normal Commands:")
    print(normal_commands)

    print("\nAttack Commands:")
    print(attack_commands)

    return normal_commands


normal_commands = firewall_normal(cfg.TEST_DATASET)
