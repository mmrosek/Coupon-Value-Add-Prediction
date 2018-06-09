import sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import csv
import math
import operator
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import time
import plotly as py
import plotly.graph_objs as go

abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
sys.path.append(file_dir)

from feat_eng import *
from modeling import *
from mlp_bayes_opt_legit import *
from create_pred_set import *

os.chdir("") # Insert path to dunhumby data sets

def group_basket_stats(product_list, df_transactions, df_demographic):

    print("Grouping Baskets...")
    df_grouped_basket = get_grouped_basket(df_transactions)

    print("getting product counts for each basket")
    df_grouped_basket_count = get_grouped_basket_count(df_grouped_basket)

    print("getting summed quantities for each basket id...")
    df_grouped_basket_sum = get_grouped_basket_sum(df_grouped_basket)

    print("Applying label...")
    df_grouped_basket = apply_label_grouped_basket(df_grouped_basket)

    print("merging count, sum and labels...")
    df_grouped_basket_merge = merging_sum_count_labels(df_grouped_basket, df_grouped_basket_count, df_grouped_basket_sum)

    print("merging with demmographic data....")
    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_demographic, on="household_key", how="left").reset_index(drop=True)

    print("First ten rows of the dataset...")
    print(df_grouped_basket_merge.head(10)) # Sanity check

    return df_grouped_basket_merge

def get_grouped_basket(df_transactions):
    return df_transactions.groupby(['household_key', 'BASKET_ID', 'DAY'])

def get_grouped_basket_count(df_grouped_basket):
    df_grouped_basket_count = df_grouped_basket.size().reset_index()
    df_grouped_basket_count = df_grouped_basket_count.rename(columns={0: 'PROD_PURCHASE_COUNT'})
    return df_grouped_basket_count

def apply_label_grouped_basket(df_grouped_basket):
    df_grouped_basket = df_grouped_basket.apply(
            lambda x : 1 if len(set(x.PRODUCT_ID.tolist()) & set(product_list)) > 0 else 0
        ).reset_index().rename(columns={0:"label"})
    return df_grouped_basket

def get_grouped_basket_sum(df_grouped_basket):
    df_grouped_basket_sum = df_grouped_basket.sum().reset_index()
    df_grouped_basket_sum.drop(['RETAIL_DISC', 'TRANS_TIME', 'COUPON_MATCH_DISC', 'START_DAY', 'END_DAY'], axis=1, inplace=True)
    return df_grouped_basket_sum

def merging_sum_count_labels(df_grouped_basket, df_grouped_basket_count, df_grouped_basket_sum):
    df_grouped_basket_merge = df_grouped_basket_sum.merge(df_grouped_basket, on=["household_key", "BASKET_ID"]).reset_index(drop=True)
    del df_grouped_basket
    del df_grouped_basket_sum

    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_grouped_basket_count, on=["household_key", "BASKET_ID"]).reset_index(drop=True)
    del df_grouped_basket_count

    df_grouped_basket_merge = df_grouped_basket_merge.drop(['DAY_x', 'DAY_y'], axis=1)

    return df_grouped_basket_merge

def get_products_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['PRODUCT_ID'].unique()

def get_campaigns_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['CAMPAIGN'].unique()

def get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc):
    #get subset from campaign table to get the households for the campaign
    subset = df_campaign_table[df_campaign_table['CAMPAIGN'].isin(campaigns)]
    hh_start_dates = subset.merge(df_campaign_desc, on='CAMPAIGN', how='left')
    hh_start_dates = hh_start_dates.sort_values(['household_key', 'START_DAY'])
    return hh_start_dates.drop_duplicates(['household_key'], keep="first")

def get_transactions_for_hh(df_transactions, hh_start_dates):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
    trans_merge['START_DAY'].fillna(10000, inplace=True)
    return trans_merge[trans_merge['DAY'].astype(float) < trans_merge['START_DAY']]

def get_transactions_for_hh_within(df_transactions, hh_start_dates, product_list):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
    trans_merge['START_DAY'].fillna(10000, inplace=True)
    trans_merge['END_DAY'].fillna(0, inplace=True)
    trans_filtered = trans_merge[(trans_merge['DAY'].astype(float) >= trans_merge['START_DAY']) & (
                trans_merge['DAY'].astype(float) <= trans_merge['END_DAY'])]
    trans_filtered['label'] = 0
    trans_filtered['label'] = trans_filtered.apply(lambda row: 1 if row['PRODUCT_ID'] in product_list else 0,
                                                   axis=1)
    trans_filtered = trans_filtered[trans_filtered['label'] == 1]
    
    return trans_filtered[['household_key', 'PRODUCT_ID', 'CAMPAIGN']], list(trans_filtered['household_key'].unique())

if __name__ == "__main__":
    
#    coupon_id_list = ["10000089073", "57940011075", "10000089061", "51800000050"]
    coupon_Id = "51800000050" 
    print("Coupon ID: " + coupon_Id)

    print("Reading coupon data...")
    df_coupon = pd.read_csv('coupon.csv', dtype={'COUPON_UPC': str, 'CAMPAIGN': str, 'PRODUCT_ID': str})
    campaigns = get_campaigns_for_coupon(coupon_Id, df_coupon)
    print("Campaigns associated with the coupon: " + str(len(campaigns)))

    product_list = get_products_for_coupon(coupon_Id, df_coupon)
    del df_coupon
    print("Products associated with the coupon: "+ str(len(product_list)))

    print("Reading in campaign_table and campaign_desc...")
    df_campaign_table = pd.read_csv('campaign_table.csv', dtype={'household_key': str, 'CAMPAIGN': str})
    df_campaign_desc = pd.read_csv('campaign_desc.csv', dtype={'CAMPAIGN': str})

    hh_start_dates = get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc)
    del df_campaign_table
    hh_start_dates.drop(['DESCRIPTION_x', 'DESCRIPTION_y'], axis=1, inplace=True)
    print("Households associated with the campaign: "+str(len(hh_start_dates)))

    print("Reading in transactions... it's huge")
    df_transactions = pd.read_csv('transaction_data.csv', dtype={'BASKET_ID': str, 'PRODUCT_ID': str, 'household_key': str, 'DAY': str})
    print("lenght of all transactions: "+str(len(df_transactions)))
    
    transactions_within_campaign, households_campaign_list = get_transactions_for_hh_within(df_transactions, hh_start_dates, product_list)
    
    print("filtering transactions for households ")
    df_transactions = get_transactions_for_hh(df_transactions, hh_start_dates)
    df_transactions['CUSTOMER_PAID'] = df_transactions['SALES_VALUE'] + df_transactions['COUPON_DISC']
    print("filtered transactions length: "+str(len(df_transactions)))

    df_demographic = pd.read_csv('hh_demographic.csv', dtype={'household_key': str})
    df_grouped_basket = group_basket_stats(product_list, df_transactions, df_demographic)

    df_grouped_basket['demo_missing'] = np.where(df_grouped_basket['INCOME_DESC'].isnull(), 1, 0) # Creating one-hot for where demographic data is missing
    
    print("\nFeature engineering on train set")
    print("Length of train set: " + str(len(df_grouped_basket)))
    start = time.time()
    exp_stats = ['label', 'PROD_PURCHASE_COUNT', 'QUANTITY']
    df_eng_feats_train = feat_eng(df_grouped_basket, exp_stats, exp_stats)
    end = time.time()
    print("Time to engineer features on train set: " + str(end-start))

    df_eng_feats_train = prep_train_set(df_eng_feats_train)
    print("length of feat eng: "+str(len(df_eng_feats_train)))

    X, y, _ = split_feats_label(df_eng_feats_train)
        
    scaler = StandardScaler()
    features_std = scaler.fit_transform(X) # Normaliizing features
    del X

    #train the model
    print("Training the model...")
    trained_mlp = train_mlp(features_std, y, 4, 1, 20, 20)
    
    print("\nGenerating prediction set")
    df_eng_feats_pred = gen_pred_set(coupon_Id)
    
    if len(set(df_eng_feats_train.columns) - set(df_eng_feats_pred.columns)) != 0: # Should be 0
          print("\nMismatched columns in pred and train set. Adding columns to pred set.\n")
          
          # Adding columns in train set to pred set
          cols_in_train_not_pred = list(set(df_eng_feats_train.columns) - set(df_eng_feats_pred.columns))
          for col in cols_in_train_not_pred:
              df_eng_feats_pred[col]=0
        
          # Sort columns
          df_eng_feats_pred.sort_index(axis = 1, inplace = True)
                               
          if len(set(df_eng_feats_train.columns) - set(df_eng_feats_pred.columns)) != 0: # Should be 0
                print("\nColumns still mismatched --> column(s) in pred that were not in training set.\n")
          
    X, y, pred_household_key = split_feats_label(df_eng_feats_pred)
        
    pred_features_std = scaler.transform(X)
    
    pred_set_pred_prob = trained_mlp.predict_proba(pred_features_std)[:,1]
    pred_set_preds = trained_mlp.predict(pred_features_std)
    
    pred_df = pd.DataFrame({'household_key': pred_household_key, 'pred': pred_set_preds, 'pred_soft':pred_set_pred_prob})
    
    pred_df['label'] = pred_df.apply(lambda row: 1 if row['household_key'] in households_campaign_list else 0, axis=1)

    pred_df["prob_added"] = pred_df["label"] - pred_df["pred_soft"]
    
    try:
        pred_df.drop(['CAMPAIGN', 'DESCRIPTION'], inplace=True, axis=1)
    except ValueError:
        pass
    
    ### Removing households who received TypeA campaigns
    pred_df_w_camp = pred_df.merge(hh_start_dates, on = 'household_key', how = 'left') # Adding campaign number
    pred_df_w_camp_type = pred_df_w_camp.merge(df_campaign_desc, on = 'CAMPAIGN', how = 'left') # Adding campaign type
    pred_df_w_camp_type = pred_df_w_camp_type[pred_df_w_camp_type.DESCRIPTION != 'TypeA']
    pred_df_w_camp_type.drop([col for col in pred_df_w_camp_type.columns if "DAY" in col], errors='ignore', axis=1, inplace=True)
        
    mean_prob_added = pred_df_w_camp_type['prob_added'].mean()

    coupon_demographics = pred_df_w_camp_type.merge(df_demographic, on='household_key', how='inner')

    demographic_columns = coupon_demographics.columns[8:]

    for i in demographic_columns:

        counter = coupon_demographics.groupby([i]).size().reset_index().rename(columns={0: 'count'})

        coupon_demographic = coupon_demographics.groupby([i]).mean().reset_index()

        coupon_demographic[i] = coupon_demographic[i].astype('str')

        data = [go.Bar(
            x=coupon_demographic[i],
            y=coupon_demographic['prob_added'],
            text=counter['count'],
            textposition='auto',
            textfont=dict(
                family='sans serif',
                size=18,
                color='#000000',

            ),
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]

        layout = go.Layout(
            title='Mean Purchase Probability Added Across ' + i + ' Groups',
            titlefont=dict(
                size=26
            ),
            xaxis=dict(
                title=i,
                categoryorder="array",
                categoryarray=coupon_demographic[i],
                titlefont=dict(
                    family='sans serif',
                    size=22,
                    color='#000000',
                ),
                tickfont=dict(
                    family='sans serif',
                    size=18,
                    color='black'
                ),
            ),
            yaxis=dict(
                title='Mean Purchase Probability Added',
                titlefont=dict(
                    family='sans serif',
                    size=22,
                    color='#000000'
                ),
                tickfont=dict(
                    family='sans serif',
                    size=18,
                    color='black'
                ),
            ),
        )
        fig = go.Figure(data=data, layout=layout)
        py.offline.plot(fig, filename=i + '.html')




