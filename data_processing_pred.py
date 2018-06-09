import sys
import numpy as np
import pandas as pd
import csv
import math
import operator
import os

from feat_eng import *
from modeling import *

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
sys.path.append(file_dir)

sys.path.append("/home/miller/Documents/GT/Biz Anal/Projy/Code/data_analytics_inbusiness/")



os.chdir("/home/miller/Documents/GT/Biz Anal/Projy/Data/CSV/")

pred_set_indicator = 9999999 # Will serve as indicator for rows to be in pred set 

def get_grouped_basket(product_list, trans_merge, df_demographic):

    df_grouped_basket = trans_merge.groupby(['household_key', 'BASKET_ID', 'DAY'])
    
    # Getting purchase counts
    df_grouped_basket_copy = df_grouped_basket.size().reset_index()
    df_grouped_basket_copy.rename(columns={0:'PROD_PURCHASE_COUNT'}, inplace=True)
    
    # Calculating label
    df_grouped_basket = df_grouped_basket.apply(
            lambda x : 1 if len(set(x.PRODUCT_ID.tolist()) & set(product_list)) > 0 else 0
        ).reset_index().rename(columns={0:"label"})

    # Getting sum of purchases
    df_grouped_basket_2 = trans_merge.groupby(['household_key', 'BASKET_ID']).sum().reset_index()
    df_grouped_basket_2.drop(['RETAIL_DISC', 'TRANS_TIME', 'COUPON_MATCH_DISC', 'START_DAY', 'END_DAY'], axis=1, inplace=True)

    df_grouped_basket_merge = df_grouped_basket_2.merge(df_grouped_basket, on=["household_key", "BASKET_ID"]).reset_index(drop=True)

    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_grouped_basket_copy, on=["household_key", "BASKET_ID"]).reset_index(drop=True)

    df_grouped_basket_merge = df_grouped_basket_merge.drop(['DAY_x'], axis=1, errors = 'ignore')

    # Renaming columns
    df_grouped_basket_merge.rename(columns={"DAY_y":"DAY", "STORE_ID_x":"STORE_ID"}, inplace=True)
     
    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_demographic, on="household_key", how = "left").reset_index(drop=True)
    
    return df_grouped_basket_merge

def get_products_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['PRODUCT_ID'].unique()

def get_campaigns_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['CAMPAIGN'].unique()

def get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc):
    #get subset from campiagn table to get the households for the campaign
    subset = df_campaign_table[df_campaign_table['CAMPAIGN'].isin(campaigns)]
    hh_start_dates = subset.merge(df_campaign_desc, on='CAMPAIGN', how='left')
    hh_start_dates = hh_start_dates.sort_values(['household_key', 'START_DAY'])
    return hh_start_dates.drop_duplicates(['household_key'], keep="first")

def create_dummy_df(hh_start_dates):
    
    '''Creating dummy rows for households who received coupon.
       dummy_df emulates what rows for these households would have looked /
       like if they were in df_transactions. dummy_df will be appended to transaction_df.'''

    ### Prepping prediction set --> dummy_df = rows to be predicted on --> need to synthesize data
    dummy_df = hh_start_dates.copy()
        
    dummy_df['DAY'] = dummy_df['START_DAY'] # DAY is first day of campaign for each household
    
    dummy_df.drop(['START_DAY','END_DAY',"CAMPAIGN"], axis=1, inplace=True)
    
    dummy_df["BASKET_ID"] = pred_set_indicator
    dummy_df["PRODUCT_ID"] = pred_set_indicator    
    dummy_df["QUANTITY"] = 0    
    dummy_df["SALES_VALUE"] = 0  
    dummy_df["STORE_ID"] = pred_set_indicator
    dummy_df["RETAIL_DISC"] = 0    
    dummy_df["TRANS_TIME"] = 0    
    dummy_df["WEEK_NO"] =  dummy_df.apply(lambda row: math.ceil(float(row['DAY'])/ 7), axis=1)
    dummy_df["COUPON_DISC"] = 0    
    dummy_df["COUPON_MATCH_DISC"] = 0  
            
    return dummy_df

def get_transactions_for_hh(df_transactions, hh_start_dates):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
#    trans_merge["START_DAY"].fillna(10000, inplace = True) # New
    return trans_merge[trans_merge['DAY'].astype(float) <= trans_merge['START_DAY']]


if __name__ == "__main__":
    coupon_Id = "51800000050"

    df_coupon = pd.read_csv('coupon.csv', dtype={'COUPON_UPC': str, 'CAMPAIGN': str, 'PRODUCT_ID': str})
    campaigns = get_campaigns_for_coupon(coupon_Id, df_coupon)

    df_campaign_table = pd.read_csv('campaign_table.csv', dtype={'household_key': str, 'CAMPAIGN': str})
    df_campaign_desc = pd.read_csv('campaign_desc.csv', dtype={'CAMPAIGN': str})

    hh_start_dates = get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc)
    hh_start_dates.drop(['DESCRIPTION_x', 'DESCRIPTION_y'], axis=1, inplace=True)
    
    print("Creating dummy df")
    
    dummy_df = create_dummy_df(hh_start_dates)

    df_transactions = pd.read_csv('transaction_data.csv', dtype={'BASKET_ID': str, 'PRODUCT_ID': str, 'household_key': str, 'DAY': str})
    
    df_transactions = df_transactions.append(dummy_df) # Appending dummy_df aka future prediction set
    
    df_transactions = get_transactions_for_hh(df_transactions, hh_start_dates)
        
    df_transactions['CUSTOMER_PAID'] = df_transactions['SALES_VALUE'] + df_transactions['COUPON_DISC']
    
    product_list = get_products_for_coupon(coupon_Id, df_coupon)
    
    df_demographic = pd.read_csv('hh_demographic.csv', dtype={'household_key': str})
    
    print("Grouping")
    
    df_grouped_basket = get_grouped_basket(product_list, df_transactions, df_demographic) 
    
    df_grouped_basket['demo_missing'] = np.where(df_grouped_basket['INCOME_DESC'].isnull(), 1, 0) # Creating one-hot for where demographic data is missing
    
    df_grouped_basket["DAY"] = df_grouped_basket["DAY"].astype(float)
    
    df_grouped_basket.sort_values(["household_key", "DAY"], inplace=True)
    
    print("Feature engineering prediction set")
    
    # Feature engineering
    exp_stats = ['label', 'PROD_PURCHASE_COUNT', "QUANTITY"]

    df_eng_feats_pred = feat_eng(df_grouped_basket, exp_stats, exp_stats)
    
    df_eng_feats_pred = extract_pred_set(df_eng_feats_pred)
    
    print("Writing pred set with eng feats to csv")
    
    df_eng_feats_pred.to_csv("pred_set_feat_eng_{}.csv".format(coupon_Id), index=False)
    
    X,y = split_feats_label(df_eng_feats_pred)
    
    trained_lr = train_mod(X,y,3)


    
    
    
    
    
    
    

