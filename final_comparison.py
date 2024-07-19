###WITH MATCHED CONTROLS SEPARATED BY YEAR
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel

# Load the player statistics data
merged_subset_imputed = pd.read_csv('merged_subset_clusters_imputed.csv')

# Convert date columns to datetime
acl_injuries['Injury_Date'] = pd.to_datetime(acl_injuries['Injury_Date'])
acl_injuries['Return_Date'] = pd.to_datetime(acl_injuries['Return_Date'])

# Extract unique player names
acl_players = acl_injuries['Player']

# Define function to calculate similarity
def calculate_similarity(index_season_stats, potential_controls):
    # Ensure both datasets have only numeric columns
    index_season_stats = index_season_stats.select_dtypes(include=[np.number])
    potential_controls = potential_controls.select_dtypes(include=[np.number])
    
    # Calculate Euclidean distance between the index season stats and potential controls
    distances = cdist(index_season_stats, potential_controls, metric='euclidean')
    return distances

# Prepare a list to store matched controls
matched_controls = []

# Loop through each ACL-injured player
for player in acl_players:
    # Get the player's injury data
    injury_data = acl_injuries[acl_injuries['Player'] == player].iloc[0]
    injury_season = injury_data['Injury_Season']
    return_season = injury_data['Return_Season']
    
    # Get the player's index season stats
    index_season_stats = merged_subset_imputed[(merged_subset_imputed['player'] == player) & 
                                                (merged_subset_imputed['season'] == injury_season)]
    
    if index_season_stats.empty:
        continue
    
    # Get player's age and seasons of play from the index season stats
    player_age = index_season_stats['age'].values[0]
    player_season = index_season_stats['season'].values[0]
    
    # Get potential control players (within 1 year of age and 5 seasons of play)
    potential_controls = merged_subset_imputed[(merged_subset_imputed['age'].between(player_age - 1, player_age + 1)) &
                                                (merged_subset_imputed['season'].between(player_season - 5, player_season + 5)) &
                                                (merged_subset_imputed['player'] != player)]
    
    # Calculate similarity scores
    similarity_scores = calculate_similarity(index_season_stats.iloc[:, 2:], 
                                             potential_controls.iloc[:, 2:])
    
    # Get the two most similar players
    closest_indices = similarity_scores.argsort()[0][:2]
    closest_controls = potential_controls.iloc[closest_indices]
    
    matched_controls.append(closest_controls)

# Concatenate all matched controls into a single DataFrame
matched_controls_df = pd.concat(matched_controls)

# Calculate the number of seasons missed due to ACL injury
acl_injuries['seasons_missed'] = acl_injuries['Return_Season'] - acl_injuries['Injury_Season']

# Define function to get pre-injury and post-injury stats for each year separately
def get_pre_post_stats(player_name, injury_season, max_seasons_post_injury):
    # Get player's data
    player_data = merged_subset_imputed[merged_subset_imputed['player'] == player_name]
    
    # Get pre-injury stats (including the injury season)
    pre_injury_stats = player_data[player_data['season'] <= injury_season].select_dtypes(include=[np.number]).mean()
    
    # Get post-injury stats for each year separately
    post_injury_stats = {}
    for i in range(1, max_seasons_post_injury + 1):
        season = injury_season + i
        stats = player_data[player_data['season'] == season].select_dtypes(include=[np.number]).mean()
        if not stats.empty:
            post_injury_stats[season] = stats
            
    return pre_injury_stats, post_injury_stats

# Determine the maximum number of seasons post-injury to consider
max_seasons_post_injury = 5

# Prepare lists to store results separately for injured players and matched controls
pre_injury_injured_list = []
post_injury_injured_list = []
season_injured_list = []

pre_injury_control_list = []
post_injury_control_list = []
season_control_list = []

# Track players processed successfully
processed_players = []

# Loop through each ACL-injured player and their matched controls
for index, row in acl_injuries.iterrows():
    player_name = row['Player']
    injury_season = row['Injury_Season']
    return_season = row['Return_Season']
    
    # Get pre-injury and post-injury stats for the ACL-injured player
    pre_stats, post_stats = get_pre_post_stats(player_name, injury_season, max_seasons_post_injury)
    if pre_stats.isna().any():
        continue
    
    for season, post_season_stats in post_stats.items():
        if post_season_stats.isna().any():
            continue
        pre_injury_injured_list.append(pre_stats)
        post_injury_injured_list.append(post_season_stats)
        season_injured_list.append(season - injury_season)
        processed_players.append(player_name)
    
    # Get the matched controls
    matched_control_names = matched_controls_df[matched_controls_df['player'] == player_name]['player'].unique()
    
    for control_name in matched_control_names:
        pre_stats, post_stats = get_pre_post_stats(control_name, injury_season, max_seasons_post_injury)
        if pre_stats.isna().any():
            continue
        
        for season, post_season_stats in post_stats.items():
            if post_season_stats.isna().any():
                continue
            pre_injury_control_list.append(pre_stats)
            post_injury_control_list.append(post_season_stats)
            season_control_list.append(season - injury_season)

# Convert lists to DataFrames
pre_injury_injured_df = pd.DataFrame(pre_injury_injured_list)
post_injury_injured_df = pd.DataFrame(post_injury_injured_list)
season_injured_df = pd.Series(season_injured_list, name='season_post_injury')

pre_injury_control_df = pd.DataFrame(pre_injury_control_list)
post_injury_control_df = pd.DataFrame(post_injury_control_list)
season_control_df = pd.Series(season_control_list, name='season_post_injury')

# Ensure that the data frames have the same length by removing rows with NaN values
pre_injury_injured_df = pre_injury_injured_df.dropna()
post_injury_injured_df = post_injury_injured_df.dropna()
season_injured_df = season_injured_df[season_injured_df.index.isin(pre_injury_injured_df.index)]

pre_injury_control_df = pre_injury_control_df.dropna()
post_injury_control_df = post_injury_control_df.dropna()
season_control_df = season_control_df[season_control_df.index.isin(pre_injury_control_df.index)]

# Perform paired t-tests for each statistical category for each year post-injury for injured players
results_injured = {}

for column in pre_injury_injured_df.columns:
    if column not in ['player', 'season']:  # exclude non-stat columns
        for season in range(1, max_seasons_post_injury + 1):
            pre_injury_data_injured = pre_injury_injured_df[season_injured_df == season][column].dropna()
            post_injury_data_injured = post_injury_injured_df[season_injured_df == season][column].dropna()
            
            # Ensure the arrays have equal length
            min_length_injured = min(len(pre_injury_data_injured), len(post_injury_data_injured))
            pre_injury_data_injured = pre_injury_data_injured[:min_length_injured]
            post_injury_data_injured = post_injury_data_injured[:min_length_injured]
            
            # Perform t-test for injured players
            if len(pre_injury_data_injured) > 0 and len(post_injury_data_injured) > 0:  # Ensure there's data to compare
                t_stat_injured, p_val_injured = ttest_rel(pre_injury_data_injured, post_injury_data_injured)
                results_injured[(column, season)] = {'t_stat': t_stat_injured, 'p_val': p_val_injured}

# Perform paired t-tests for each statistical category for each year post-injury for matched controls
results_control = {}

for column in pre_injury_control_df.columns:
    if column not in ['player', 'season']:  # exclude non-stat columns
        for season in range(1, max_seasons_post_injury + 1):
            pre_injury_data_control = pre_injury_control_df[season_control_df == season][column].dropna()
            post_injury_data_control = post_injury_control_df[season_control_df == season][column].dropna()
            
            # Ensure the arrays have equal length
            min_length_control = min(len(pre_injury_data_control), len(post_injury_data_control))
            pre_injury_data_control = pre_injury_data_control[:min_length_control]
            post_injury_data_control = post_injury_data_control[:min_length_control]
            
            # Perform t-test for matched controls
            if len(pre_injury_data_control) > 0 and len(post_injury_data_control) > 0:  # Ensure there's data to compare
                t_stat_control, p_val_control = ttest_rel(pre_injury_data_control, post_injury_data_control)
                results_control[(column, season)] = {'t_stat': t_stat_control, 'p_val': p_val_control}

# Convert results to DataFrames for injured players and matched controls
results_injured_df = pd.DataFrame(results_injured).T
results_control_df = pd.DataFrame(results_control).T

# Rename columns for clarity
results_injured_df.columns = ['T-Statistic (Injured)', 'P-Value (Injured)']
results_control_df.columns = ['T-Statistic (Matched Controls)', 'P-Value (Matched Controls)']

# Display the results
print("Results for Injured Players:")
print(results_injured_df)
print("\nResults for Matched Controls:")
print(results_control_df)

# Save results to CSV files
results_injured_df.to_csv('acl_injury_impact_analysis_injured.csv')
results_control_df.to_csv('acl_injury_impact_analysis_matched_controls.csv')

# Identify players not included in the analysis
excluded_players = set(acl_players) - set(processed_players)
print("\nExcluded Players:", excluded_players)

