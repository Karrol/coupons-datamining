#!/usr/bin/env python
# coding: utf-8
# In[1]:
# target：导入程序包；
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
pd.options.display.max_columns=1000
pd.options.display.max_rows=1000
import numpy as np
import multiprocessing as mp
import time
import sqlite3
import gc

#In[0]
"""预定义目录集"""
DIRE_O='data/'
DIRE_N='data-zl/'
"""预定义函数集群"""

#In[0]-1

# In[2]:
# target：切割customer_trans数据；method：采用pandas的采样随机函数
original_customer_transaction_data=pd.read_csv(DIRE_O+'customer_transaction_data.csv').drop_duplicates()
"original_customer_transaction_data", original_customer_transaction_data.shape 
sample=original_customer_transaction_data.sample(frac=0.1)
print(sample.head())
sample=sample.reset_index()
print(sample.head())
#将样本写出
#df2sqlite(sample,tablename='0-sample-cunstomer-transaction-data')
sample.to_csv(DIRE_N+'0-sample-cunstomer-transaction-data.csv',header=True,index=False)
del original_customer_transaction_data

#In[3]
#target:导入所有的表；
train = pd.read_csv(DIRE_O + 'train.csv')
print("train", train.shape)

test = pd.read_csv(DIRE_O + 'test_QyjYwdj.csv')
print("test", test.shape)

campaign_data = pd.read_csv(DIRE_O + 'campaign_data.csv')
print("campaign_data", campaign_data.shape)

customer_demographics = pd.read_csv(DIRE_O + 'customer_demographics.csv')
print("customer_demographics", customer_demographics.shape)

customer_transaction_data = sample
del sample
gc.collect()
print("customer_transaction_data", customer_transaction_data.shape )
customer_transaction_data = customer_transaction_data.drop_duplicates()
print("customer_transaction_data", customer_transaction_data.shape )

item_data = pd.read_csv(DIRE_O + 'item_data.csv')
print("item_data",item_data.shape)

coupon_item_mapping = pd.read_csv(DIRE_O + 'coupon_item_mapping.csv')
print("coupon_item_mapping", coupon_item_mapping.shape)

#In[4]
#target:依次清洗所有表的字段，不做空缺值处理
campaign_data['start_date'] = pd.to_datetime(campaign_data.start_date, format='%d/%m/%y')
campaign_data['end_date'] = pd.to_datetime(campaign_data.end_date,  format='%d/%m/%y')
campaign_data['duration'] = (campaign_data.end_date - campaign_data.start_date).dt.days
customer_transaction_data['date'] = pd.to_datetime(customer_transaction_data.date, format='%Y-%m-%d') 
customer_transaction_data['coupon_discount'] = customer_transaction_data.coupon_discount.abs()
customer_transaction_data['other_discount'] = customer_transaction_data.other_discount.abs()
#依据coupon_discount可以推测出用户使用消费券的情况（但消费券不一定是商家活动的coupon表中有的coupon）
customer_transaction_data['coupon_applied'] = (customer_transaction_data['coupon_discount'] > 0).astype(int)
print(customer_transaction_data.head())

#In[5]
"""todo:target:对各个表进行描述统计"""


#In[6]
"""后面的merge会共同需要的处理：标记过的test表和train表与compaign进行合并"""
df_test = test.merge(campaign_data, how='left')
print(df_test.shape)
print(df_test.head())
df_train = train.merge(campaign_data, how='left')
print(df_train.shape)
print(df_train.head())
customer_transaction_data.date.min(), customer_transaction_data.date.max()
df_train.start_date.min(), df_train.end_date.max()
df_test.start_date.min(), df_test.end_date.max()
df_all = pd.concat([df_train, df_test], sort=False, axis=0)
print(df_all.shape)
print(df_all.head())
df_ = df_all[['campaign_id', 'coupon_id', 'customer_id', 'redemption_status', 'campaign_type', 'start_date', 'end_date']].drop_duplicates()
print(df_.shape)
df_.to_csv(DIRE_N+'df_.csv',index=False)


#In[7]
"""target:分组做表合并生成新表->待生成(cust_hist_trans)->待处理表（df_all,item_data,cust_transactions_data,）
研究目的是“商品数据（item）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""

cust_transactions = customer_transaction_data.merge(item_data, on = 'item_id', how='left')
cust_transactions.to_csv(DIRE_N+'cust_transactions.csv', index=False)
df_x = df_all[['customer_id', 'start_date']].drop_duplicates().merge(cust_transactions, on=['customer_id'])
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['customer_id', 'date'])

cust_hist_trans = df_x.groupby(['customer_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})

cust_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_hist_trans.columns.values]
cust_hist_trans = cust_hist_trans.reset_index()
"cust_hist_trans",cust_hist_trans.shape
cust_hist_trans.head()
cust_hist_trans.to_csv(DIRE_N+'1-cust_hist_trans.csv', index=False)

#In[8]
"""target:分组做表合并生成新表->待生成(cust_hist_trans_daily)->待处理表（df_all,cust_transactions）
研究目的是“商品数据（item）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
cust_daily = cust_transactions.groupby(['date', 'customer_id']).agg({
    'item_id': 'count',
    'quantity': 'sum',
    'selling_price': 'sum',
    'other_discount': 'sum',
    'coupon_discount': 'sum',
    'coupon_applied': 'sum'
})

# cust_daily.columns = ['_'.join(col).strip('_') for col in cust_daily.columns.values]
cust_daily = cust_daily.reset_index()
cust_daily = cust_daily.sort_values(by=['customer_id', 'date'])
##ewm是 加权移动平均数，能够让最近时间的数据更重要，每一个时间点都可以有一个ewm
cust_daily['selling_price_ewm'] = cust_daily.groupby('customer_id')['selling_price'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_daily['coupon_discount_ewm'] = cust_daily.groupby('customer_id')['coupon_discount'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_daily['coupon_applied_ewm'] = cust_daily.groupby('customer_id')['coupon_applied'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()

# cust_daily.head()

df_x = df_all[['customer_id', 'start_date']].drop_duplicates().merge(cust_daily, on=['customer_id'])
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['customer_id', 'start_date', 'date'])


cust_hist_trans_daily = df_x.groupby(['customer_id', 'start_date']).agg({
    'date': ['count'],
    'selling_price': ['mean', 'std', 'last'],
    'other_discount': ['mean', 'std', 'last'],
    'coupon_discount': ['mean', 'std', 'last'],
    'coupon_applied': ['mean', 'std', 'last'],
    'selling_price_ewm': ['mean', 'std'],
    'coupon_discount_ewm': ['mean', 'std'],
    'coupon_applied_ewm': ['mean', 'std'],
})

cust_hist_trans_daily.columns = ['_'.join(col).strip('_') for col in cust_hist_trans_daily.columns.values]
cust_hist_trans_daily = cust_hist_trans_daily.reset_index()
print(cust_hist_trans_daily.shape)
print(cust_hist_trans_daily.head())
cust_hist_trans_daily.to_csv(DIRE_N+'2-cust_hist_trans_daily.csv', index=False)

#In[9]
"""target:分组做表合并生成新表->待生成(cust_coup_hist_trans)->待处理表（coupon_item_mapping,cust_transactions,df_,coup_trans）
研究目的是“优惠券数据（coupon-item）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
coup_trans = coupon_item_mapping.merge(cust_transactions, on='item_id', how='left')
print("coup_trans",coup_trans.shape)
coup_trans.head()
df_x = df_.merge(coup_trans, on=['customer_id', 'coupon_id'], how='left')
print("df_x",df_x.shape)
df_x = df_x[df_x.start_date > df_x.date]
print("df_x",df_x.shape)
cust_coup_hist_trans = df_x.groupby(['customer_id', 'coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})

cust_coup_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_coup_hist_trans.columns.values]
cust_coup_hist_trans = cust_coup_hist_trans.reset_index()
print(cust_coup_hist_trans.shape)
print(cust_coup_hist_trans.head())
cust_coup_hist_trans.to_csv(DIRE_N+'3-cust_coup_hist_trans.csv', index=False)


# In[10]
"""target:分组做表合并生成新表->待生成(coup_hist_trans_daily)->待处理表（coup_trans,df_all）
研究目的是“优惠券数据（coupon-item）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
coup_daily = coup_trans.groupby(['coupon_id', 'date']).agg({
    'item_id': 'count',
    'quantity': 'sum',
    'selling_price': 'sum',
    'other_discount': 'sum',
    'coupon_discount': 'sum',
    'coupon_applied': 'sum'
})

# cust_daily.columns = ['_'.join(col).strip('_') for col in cust_daily.columns.values]
coup_daily = coup_daily.reset_index()
coup_daily = coup_daily.sort_values(by=['coupon_id', 'date'])

coup_daily['selling_price_ewm'] = coup_daily.groupby('coupon_id')['selling_price'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
coup_daily['coupon_discount_ewm'] = coup_daily.groupby('coupon_id')['coupon_discount'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
coup_daily['coupon_applied_ewm'] = coup_daily.groupby('coupon_id')['coupon_applied'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()


df_x = df_all[['coupon_id', 'start_date']].drop_duplicates().merge(coup_daily, on=['coupon_id'])
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['coupon_id', 'start_date', 'date'])


coup_hist_trans_daily = df_x.groupby(['coupon_id', 'start_date']).agg({
    'date': ['count'],
    'selling_price': ['mean', 'std', 'last'],
    'other_discount': ['mean', 'std', 'last'],
    'coupon_discount': ['mean', 'std', 'last'],
    'coupon_applied': ['mean', 'std', 'last'],
    'selling_price_ewm': ['mean', 'std'],
    'coupon_discount_ewm': ['mean', 'std'],
    'coupon_applied_ewm': ['mean', 'std'],
})

coup_hist_trans_daily.columns = ['_'.join(col).strip('_') for col in coup_hist_trans_daily.columns.values]
coup_hist_trans_daily = coup_hist_trans_daily.reset_index()
print(coup_hist_trans_daily.shape)
print(coup_hist_trans_daily.head())
coup_hist_trans_daily.to_csv(DIRE_N+'4-coup_hist_trans_daily.csv', index=False)


#In[11]
"""target:分组做表合并生成新表->待生成(coup_brand_hist_trans)->待处理表（coupon_item_mapping,item_data,cust_transactions,df_）
研究目的是“商品品牌与优惠券数据（coupon-item-brand）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
"""暂时有memery error"""

coupon_items = coupon_item_mapping.merge(item_data, on='item_id')
"coupon_items",coupon_items.shape
print(coupon_items.head())
coupon_brands = coupon_items[['coupon_id', 'brand']].drop_duplicates()
print("coupon_brands",coupon_brands.shape)
print(coupon_brands.head())
#memory error here
coupon_brand_trans = coupon_brands.merge(cust_transactions, on='brand', how='left')
print("coupon_brand_trans",coupon_brand_trans.shape)
print(coupon_brand_trans.head())
df_x = df_.merge(coupon_brand_trans, on=['customer_id', 'coupon_id'], how='left')
print(df_x.shape)
df_x = df_x[df_x.start_date > df_x.date]
"df_x",df_x.shape
coup_brand_hist_trans= df_x.groupby(['coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})

coup_brand_hist_trans.columns = ['_'.join(col).strip('_') for col in coup_brand_hist_trans.columns.values]
coup_brand_hist_trans = coup_brand_hist_trans.reset_index()
print(coup_brand_hist_trans.shape)
print(coup_brand_hist_trans.head())
coup_brand_hist_trans.to_csv(DIRE_N+'5-coup_brand_hist_trans.csv', index=False)

#In[12]coupon brand
"""target:分组做表合并生成新表->待生成(cust_brand_hist_trans)->待处理表（df_x来源于In[11]）
研究目的是“商品品牌与优惠券数据（coupon-item-brand）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
"""依赖于IN[11]，所以没有办法运作"""
cust_brand_hist_trans= df_x.groupby(['customer_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})

cust_brand_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_brand_hist_trans.columns.values]
cust_brand_hist_trans = cust_brand_hist_trans.reset_index()
print(cust_brand_hist_trans.shape)
print(cust_brand_hist_trans.head())
cust_brand_hist_trans.to_csv(DIRE_N+'6-cust_brand_hist_trans.csv', index=False)


#In[13]
"""target:分组做表合并生成新表->待生成(cust_coup_brand_hist_trans)->待处理表（df_x来源于In[11]）
研究目的是“商品品牌与优惠券数据（coupon-item-brand）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
"""依赖于IN[11]，所以没有办法运作"""
cust_coup_brand_hist_trans= df_x.groupby(['customer_id', 'coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})
cust_coup_brand_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_coup_brand_hist_trans.columns.values]
cust_coup_brand_hist_trans = cust_coup_brand_hist_trans.reset_index()
print(cust_coup_brand_hist_trans.shape)
print(cust_coup_brand_hist_trans.head())
cust_coup_brand_hist_trans.to_csv(DIRE_N+'7-cust_coup_brand_hist_trans.csv', index=False)

#In[14]
