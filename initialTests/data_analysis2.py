import pandas as pd
from tabulate import tabulate

# Read the CSV files
globalvectorized_df = pd.read_csv('GlobalVectorizedfunctionresults.csv')
localvectorized_df = pd.read_csv('LocalVectorizedfunctionresults.csv')
iterative_df = pd.read_csv('LocalIterativefunctionresults.csv')


# SHOW OVERALL AVERAGES
# # Remove outlier from Function Result 2
# global_df = global_df[global_df['Function Number'] != 2]
# local_df = local_df[local_df['Function Number'] != 2]
# iterative_df = iterative_df[iterative_df['Function Number'] != 2]
#
# # Remove outlier from Function Result 3
# global_df = global_df[global_df['Function Number'] != 3]
# local_df = local_df[local_df['Function Number'] != 3]
# iterative_df = iterative_df[iterative_df['Function Number'] != 3]

# Create comparison tables
averageds_df = pd.DataFrame({
    'Approach': ['Global Vectorized', 'Local Vectorized', 'Local Iterative'],
    'Mean Error': [
        globalvectorized_df['Mean Error'].mean(),
        localvectorized_df['Mean Error'].mean(),
        iterative_df['Mean Error'].mean()
    ],
    'Standard Deviation': [
        globalvectorized_df['Standard Deviation'].mean(),
        localvectorized_df['Standard Deviation'].mean(),
        iterative_df['Standard Deviation'].mean()
    ],
    'Mean Time': [
        globalvectorized_df['Mean Time'].mean(),
        localvectorized_df['Mean Time'].mean(),
        iterative_df['Mean Time'].mean()
    ]
})

# Format the output tables
mean_error_table = tabulate(averageds_df[['Approach', 'Mean Error']], headers='keys', tablefmt='pipe', showindex=False)
std_dev_table = tabulate(averageds_df[['Approach', 'Standard Deviation']], headers='keys', tablefmt='pipe', showindex=False)
mean_time_table = tabulate(averageds_df[['Approach', 'Mean Time']], headers='keys', tablefmt='pipe', showindex=False)

# Print the formatted output tables
print('Mean Error Comparison:')
print(mean_error_table)
print()

print('Standard Deviation Comparison:')
print(std_dev_table)
print()

print('Mean Time Comparison:')
print(mean_time_table)
print()


# SHOW DETAILED FUNCTION RESULTS
# Merge the iterative_df with the mean error from global_df and local_df
merged_df = iterative_df[['Function Number', 'Mean Error']].copy()
merged_df = merged_df.merge(globalvectorized_df[['Function Number', 'Mean Error']], on='Function Number', suffixes=['', '_Global'])
merged_df = merged_df.merge(localvectorized_df[['Function Number', 'Mean Error']], on='Function Number', suffixes=['', '_Local'])

# Calculate the percentage difference between Global Vectorized and Local Iterative mean error
merged_df['%-diff Global'] = (merged_df['Mean Error_Global'] - merged_df['Mean Error']) / merged_df['Mean Error'] * 100

# Calculate the percentage difference between Local Vectorized and Local Iterative mean error
merged_df['%-diff Local'] = (merged_df['Mean Error_Local'] - merged_df['Mean Error']) / merged_df['Mean Error'] * 100

# Rearrange the columns in the desired order
merged_df = merged_df[['Function Number', 'Mean Error', 'Mean Error_Global', '%-diff Global', 'Mean Error_Local', '%-diff Local']]

# Calculate the column averages (except Function Number)
averages = merged_df.mean(axis=0)
averages['Function Number'] = 'Average'

# Convert the averages to a DataFrame and append it to merged_df
averages_df = pd.DataFrame([averages], columns=merged_df.columns)
merged_df.loc[len(merged_df)] = averages

# # Format the averages row to round the values to two decimal places
# merged_df.loc[merged_df['Function Number'] == 'Average', ['Mean Error', 'Mean Error_Global', 'Mean Error_Local', '%-diff Global', '%-diff Local']] = merged_df.loc[merged_df['Function Number'] == 'Average', ['Mean Error', 'Mean Error_Global', 'Mean Error_Local', '%-diff Global', '%-diff Local']].round(2)

# Print the table using tabulate with showindex='never'
table = tabulate(merged_df, headers='keys', tablefmt='psql', showindex=False)
print(table)



# Merge the iterative_df with the mean time from globalvectorized_df and localvectorized_df
merged_df_time = iterative_df[['Function Number', 'Mean Time']].copy()
merged_df_time = merged_df_time.merge(globalvectorized_df[['Function Number', 'Mean Time']], on='Function Number', suffixes=['', '_Global'])
merged_df_time = merged_df_time.merge(localvectorized_df[['Function Number', 'Mean Time']], on='Function Number', suffixes=['', '_Local'])

# Calculate the percentage difference between Global Vectorized and Local Iterative mean time
merged_df_time['%-diff Global'] = (merged_df_time['Mean Time_Global'] - merged_df_time['Mean Time']) / merged_df_time['Mean Time'] * 100

# Calculate the percentage difference between Local Vectorized and Local Iterative mean time
merged_df_time['%-diff Local'] = (merged_df_time['Mean Time_Local'] - merged_df_time['Mean Time']) / merged_df_time['Mean Time'] * 100

# Rearrange the columns in the desired order
merged_df_time = merged_df_time[['Function Number', 'Mean Time', 'Mean Time_Global', '%-diff Global', 'Mean Time_Local', '%-diff Local']]

# Calculate the column averages (except Function Number)
averages_time = merged_df_time.mean(axis=0)
averages_time['Function Number'] = 'Average'

# Convert the averages to a DataFrame and append it to merged_df_time
averages_df_time = pd.DataFrame([averages_time], columns=merged_df_time.columns)
merged_df_time = merged_df_time._append(averages_df_time, ignore_index=True)

# Print the table using tabulate with showindex='never'
table_time = tabulate(merged_df_time, headers='keys', tablefmt='psql', showindex=False)
print(table_time)
