import pandas as pd
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the datasets
file_path_fm1 = os.path.join(base_dir, r"fm1_data/all_units_time_series_fm1.csv")
file_path_fm2 = os.path.join(base_dir, r"fm2_data/all_units_time_series_fm2.csv")

df_fm1 = pd.read_csv(file_path_fm1)
df_fm2 = pd.read_csv(file_path_fm2)


for t in [10, 15, 20, 50, 75]:
    # t = 10
    # Filter data for units 21 to 30 and 51 to 60 until time point 10
    filtered_df_fm1 = df_fm1[~((df_fm1['unit number'].isin(range(51, 61)) | df_fm1['unit number'].isin(range(111, 121))) & (df_fm1['time, in cycles'] > t))]
    filtered_df_fm2 = df_fm2[~((df_fm2['unit number'].isin(range(51, 61)) | df_fm2['unit number'].isin(range(111, 121))) & (df_fm2['time, in cycles'] > t))]

    # Vertically merge the filtered datasets
    merged_df = pd.concat([filtered_df_fm1, filtered_df_fm2], ignore_index=True)

    # Display or save the merged dataset
    output_file_path = f'historical_plus_test_data_{t}.csv'
    merged_df.to_csv(output_file_path, index=False)  # You can replace this with a save operation if needed



    ######################

    # Load the original dataset
    file_path = f'historical_plus_test_data_{t}.csv'  # Replace this with your actual file path
    merged_df = pd.read_csv(file_path)

    # Create a copy of the rows for unit numbers 21 to 30 with failure mode 2
    units_21_30 = merged_df[merged_df['unit number'].isin(range(51, 61))].copy()
    units_21_30['failure mode'] = 2

    # Create a copy of the rows for unit numbers 51 to 61 with failure mode 1
    units_51_61 = merged_df[merged_df['unit number'].isin(range(111, 121))].copy()
    units_51_61['failure mode'] = 1

    # Concatenate the original merged_df with the new rows for 21-30 and 51-61 with swapped failure modes
    updated_df = pd.concat([merged_df, units_21_30, units_51_61], ignore_index=True)

    # Save the updated dataframe to a new CSV file
    output_file_path_updated = f'historical_plus_test_data_updated_{t}.csv'
    updated_df.to_csv(output_file_path_updated, index=False)

    #########################

    import pandas as pd

    # Load the original dataset
    file_path = f'historical_plus_test_data_{t}.csv'  # Replace this with your actual file path
    merged_df = pd.read_csv(file_path)

    # Filter out the units in the range (21 to 30) and (51 to 60)
    filtered_df = merged_df[~merged_df['unit number'].isin(range(51, 61)) & ~merged_df['unit number'].isin(range(111, 121))]

    # Save the filtered dataset to a new CSV file
    output_file_path_filtered = f'../ELBO_Maximization/historical_data.csv'
    filtered_df.to_csv(output_file_path_filtered, index=False)

    print("Filtered data has been saved as historical_data.csv.")



    ############################

    import pandas as pd

    # Load the original dataset
    file_path = f'historical_plus_test_data_updated_{t}.csv'  # Replace this with your actual file path
    merged_df = pd.read_csv(file_path)

    # Filter out the units in the range (21 to 30) and (51 to 60)
    filtered_df = merged_df[merged_df['unit number'].isin(range(51, 61)) | merged_df['unit number'].isin(range(111, 121))]

    # Save the filtered dataset to a new CSV file
    output_file_path_filtered = f'test_data_{t}.csv'
    filtered_df.to_csv(output_file_path_filtered, index=False)

    print("Filtered data has been saved as historical_data.csv.")


    ############################

    import pandas as pd

    # Load the original dataset
    file_path = f'historical_plus_test_data_{t}.csv'
    merged_df = pd.read_csv(file_path)

    # Filter out the units in the range (21 to 30) and (51 to 60)
    filtered_df = merged_df[merged_df['unit number'].isin(range(51, 61)) | merged_df['unit number'].isin(range(111, 121))]

    # Save the filtered dataset to a new CSV file
    output_file_path_filtered = f'test_data_fm_{t}.csv'
    filtered_df.to_csv(output_file_path_filtered, index=False)

    print("Filtered data has been saved as historical_data.csv.")
