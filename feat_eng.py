#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:25:57 2018

@author: miller
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:43:18 2018

@author: miller
"""
import pandas as pd
import numpy as np

pred_set_indicator = 9999999 # Will serve as indicator for rows to be in pred set 

def feat_eng(all_games, expectation_stats, cumulative_stats):
    
    '''Calculating exponential moving averages for some stats as well as summing other stats.
       Using special indexing scheme to compute ith row for each player at the same time.'''    
    
    ### Building index containing rows in which players played first game
    ### Calculating how many games each player played in their career
    ### Based on first row for each player and how many games played, can compute same row for each player at the same time.
    
    # Calculating boolean array for rows where players change
    player_shifted_up_one = all_games.household_key.shift(-1)
    new_player_bool = np.where(all_games.household_key!= player_shifted_up_one, 1, 0)
    
    # Calculating row idx for first row for each player, missing zero for first player
    new_player_row_nums_missing_zero = np.flatnonzero(new_player_bool > 0)
    new_player_row_nums = np.zeros(shape=(new_player_row_nums_missing_zero.shape[0]+1)) # Initializing new array
    
    # Contains row idx for first row for each player
    ## Last value = len(df) + 1 --> used to calculate career length of last player (below)
    new_player_row_nums[1:] = new_player_row_nums_missing_zero+1
                       
    # Calculating the max number of games played by one player
    player_career_lengths = [(row_num - new_player_row_nums[index-1])  for index, row_num in enumerate(new_player_row_nums)][1:]
    max_games_played = int(max(player_career_lengths))
        
    all_games["DAY"] = all_games["DAY"].astype(float) # Casting to float
    
    all_games.sort_values(['household_key','DAY'], inplace=True)
    all_games.reset_index(inplace=True, drop=True)   

    ###############################################################
    ### Initializing arrays, lists to hold intermediate computation    
                 
    expectation_stats.append('days_since_last_trip')
                         
    # Initializing static retain weights in exponential decaying average --> lower wt = more dependent on recent values
    med_retain_wt = 0.7
    med_update_wt = 1 - med_retain_wt
                        
    # Creating lists of  exponential weighted average column names based on how much of average is retained at each time point
    med_retain_stat_list = ['exp_{0}_{1}_retain'.format(stat, med_retain_wt) for stat in expectation_stats]
    slow_anneal_retain_stat_list = ['exp_{0}_slow_anneal_retain'.format(stat) for stat in expectation_stats]
    fast_anneal_retain_stat_list = ['exp_{0}_fast_anneal_retain'.format(stat) for stat in expectation_stats]  
       
    exp_stat_list = med_retain_stat_list + slow_anneal_retain_stat_list + fast_anneal_retain_stat_list     
    cumulative_stat_list = ['cumulative_{0}'.format(stat) for stat in cumulative_stats]   
    
    ### Initializing columns ###   
    for stat in exp_stat_list + cumulative_stat_list:
        all_games[stat] = 0
              
    # Indicator for first trip ever
    all_games['first_trip'] = 0
    all_games.loc[new_player_row_nums[:-1], 'first_trip'] = 1 
                                  
    all_games['days_since_last_trip'] = 100 # Will only remain for first row for each player
    
    all_games['cumulative_trips'] = 1 # Computing separate of other cumulative stats to avoid creating useless "trip" column
            
    cumulative_array = np.zeros(shape=(len(all_games), len(cumulative_stats)))
    med_retain_array = np.zeros(shape=(len(all_games), len(expectation_stats)))
    slow_anneal_retain_array = np.zeros(shape=(len(all_games), len(expectation_stats)))
    fast_anneal_retain_array = np.zeros(shape=(len(all_games), len(expectation_stats)))
    
    print("Max Games: " + str(max_games_played))
                            
    ##############################################################
    ### Calculating stats ###
    
    for game_num in range(1, int(max_games_played)):
        
        if game_num % 100 == 0:
            
            print(game_num)
            
        # Indices of players who have played >= game_num games
        played_num_games_bool = np.zeros(shape=(len(player_career_lengths)))
        played_num_games_bool = np.where(np.array(player_career_lengths) > game_num, True, False)
    
        # List of row indices of first game for players who have played more games than game_num
        first_game_players_played_num_games = new_player_row_nums[:-1][played_num_games_bool]
        
        rows_to_increment = (first_game_players_played_num_games+game_num).astype(int)
            
        # Updating anneal retain weights --> do this to allow for rapid updating in early games, eventually tailing off when more confident about player
        fast_anneal_retain_wt = min(0.1 + ( (game_num-1)**(1/3) * 0.35 ), 0.925)
        fast_anneal_update_wt = 1 - fast_anneal_retain_wt
        
        slow_anneal_retain_wt = min(0.1 + ( (game_num-1)**(1/6) * 0.4 ), 0.8) 
        slow_anneal_update_wt = 1 - slow_anneal_retain_wt
                    
        # If a player just played their first game...
        if game_num == 1:
            
            all_games.loc[rows_to_increment, 'days_since_last_trip'] = np.array(all_games.loc[rows_to_increment, 'DAY']) - np.array(all_games.loc[rows_to_increment-1, 'DAY'])
            all_games.loc[ rows_to_increment , 'cumulative_trips'] = np.array(all_games.loc[ rows_to_increment -1 , 'cumulative_trips']) + 1
                   
            cumulative_array[rows_to_increment,:]  = np.array(all_games.loc[rows_to_increment - 1, cumulative_stats])
                             
            # Setting retain_stats = expectation_stats to initialize value for exp moving avg
            fast_anneal_retain_array[rows_to_increment,:] = np.array(all_games.loc[rows_to_increment - 1, expectation_stats])
            slow_anneal_retain_array[rows_to_increment,:]  = np.array(all_games.loc[rows_to_increment - 1, expectation_stats])
            med_retain_array[rows_to_increment,:]  = np.array(all_games.loc[rows_to_increment - 1, expectation_stats])
    
        else:
            
            all_games.loc[ rows_to_increment , 'days_since_last_trip'] = np.array(all_games.loc[ rows_to_increment , 'DAY']) - np.array(all_games.loc[ rows_to_increment - 1 , 'DAY'])
            all_games.loc[ rows_to_increment , 'cumulative_trips'] = np.array(all_games.loc[ rows_to_increment -1 , 'cumulative_trips']) + 1
                     
            cumulative_array[rows_to_increment,:]  = np.array(all_games.loc[rows_to_increment - 1, cumulative_stats]) + cumulative_array[ rows_to_increment-1 , :]
                         
            # EMAs
            fast_anneal_retain_array[rows_to_increment , :] = fast_anneal_retain_wt * fast_anneal_retain_array[ rows_to_increment-1 , :] + fast_anneal_update_wt * np.array(all_games.loc[rows_to_increment-1, expectation_stats])
            slow_anneal_retain_array[rows_to_increment , :] = slow_anneal_retain_wt * slow_anneal_retain_array[ rows_to_increment-1 , :] + slow_anneal_update_wt * np.array(all_games.loc[rows_to_increment-1, expectation_stats])
            med_retain_array[rows_to_increment , :] = med_retain_wt * med_retain_array[ rows_to_increment-1 , :] + med_update_wt * np.array(all_games.loc[rows_to_increment-1, expectation_stats])
            
    all_games[med_retain_stat_list] = med_retain_array
    all_games[fast_anneal_retain_stat_list] = fast_anneal_retain_array
    all_games[slow_anneal_retain_stat_list] = slow_anneal_retain_array
    all_games[cumulative_stat_list] = cumulative_array
         
    del med_retain_array
    del fast_anneal_retain_array
    del slow_anneal_retain_array
    
    return all_games
    

def extract_pred_set(df):
    
    '''Extract rows to be predicted on'''
    
    df = df[df.BASKET_ID.astype(float) == pred_set_indicator]
    
    df.reset_index(inplace=True, drop=True)
    
    dummy_cols = ["HH_COMP_DESC", 'HOMEOWNER_DESC', 'HOUSEHOLD_SIZE_DESC', 'INCOME_DESC','KID_CATEGORY_DESC', 'MARITAL_STATUS_CODE', 'AGE_DESC']
    
    present_dummy_cols = [col for col in dummy_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns = present_dummy_cols)
    
    df.drop(['BASKET_ID', 'PROD_PURCHASE_COUNT', 'COUPON_DISC', 'CUSTOMER_PAID', 'QUANTITY',
       'SALES_VALUE', 'STORE_ID', 'COUNT'], axis=1, inplace=True, errors='ignore')
    
    # Sort columns
    df.sort_index(axis = 1, inplace = True)
    
    return df

def prep_train_set(df):
    
    '''Final processing for rows to be trained on'''
        
    df.reset_index(inplace=True, drop=True)
    
    dummy_cols = ["HH_COMP_DESC", 'HOMEOWNER_DESC', 'HOUSEHOLD_SIZE_DESC', 'INCOME_DESC','KID_CATEGORY_DESC', 'MARITAL_STATUS_CODE', 'AGE_DESC']
    
    present_dummy_cols = [col for col in dummy_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns = present_dummy_cols)
    
    df.drop(['BASKET_ID', 'PROD_PURCHASE_COUNT', 'COUPON_DISC', 'CUSTOMER_PAID', 'QUANTITY',
       'SALES_VALUE', 'STORE_ID', 'COUNT'], axis=1, inplace=True, errors='ignore')
    
    # Sort columns
    df.sort_index(axis = 1, inplace = True)
    
    return df
 

if __name__ == "__main__":

    exp_stats = ['label', 'PROD_PURCHASE_COUNT', "QUANTITY"]

    df_eng_feats = feat_eng(df_grouped_basket, exp_stats, exp_stats)
    
    pred_set_final = extract_pred_set(df_eng_feats)
    


