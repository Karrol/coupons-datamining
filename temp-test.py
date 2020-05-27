import pandas as pd
pd.options.display.max_columns=1000
pd.options.display.max_rows=1000
import numpy as np
import multiprocessing as mp
import time
import sqlite3
import gc

#In[temp]
"""导入下列程序需要的表"""
DIRE_O='data/'
DIRE_N='data-zl/'
coupon_item_mapping = pd.read_csv(DIRE_O + 'coupon_item_mapping.csv')
"coupon_item_mapping", coupon_item_mapping.shape
item_data = pd.read_csv(DIRE_O + 'item_data.csv')
"item_data",item_data.shape
cust_transactions=pd.read_csv(DIRE_N+'cust_transactions.csv')
"cust_transactions",cust_transactions.shape
df_=pd.read_csv(DIRE_N+'df_.csv')
"df_",df_.shape
#In[11]
"""target:分组做表合并生成新表->待生成(coup_brand_hist_trans)->待处理表（coupon_item_mapping,item_data,cust_transactions,df_）
研究目的是“商品品牌与优惠券数据（coupon-item-brand）-交易数据(transaction)”
attention：所有的表以trans表为主，做左连接，这样可以保持行数一致"""
"""暂时有memery error"""

coupon_items = coupon_item_mapping.merge(item_data, on='item_id')
"coupon_items",coupon_items.shape
coupon_items.head()
coupon_brands = coupon_items[['coupon_id', 'brand']].drop_duplicates()
"coupon_brands",coupon_brands.shape
coupon_brands.head()
coupon_brands.to_csv()
#memory error here
coupon_brand_trans = coupon_brands.merge(cust_transactions, on='brand', how='left')
"coupon_brand_trans",coupon_brand_trans.shape
coupon_brand_trans.head()
df_x = df_.merge(coupon_brand_trans, on=['customer_id', 'coupon_id'], how='left')
df_x.shape
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
coup_brand_hist_trans.shape
coup_brand_hist_trans.head()
coup_brand_hist_trans.to_csv(DIRE_N+'5-coup_brand_hist_trans.csv', index=False)

#In[12]coupon brand
"""target:分组做表合并生成新表->待生成(coup_brand_hist_trans)->待处理表（coupon_item_mapping,item_data,cust_transactions,df_）
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
cust_brand_hist_trans.shape
cust_brand_hist_trans.head()
cust_brand_hist_trans.to_csv(DIRE_N+'6-cust_brand_hist_trans.csv', index=False)


#In[13]
cust_coup_brand_hist_trans= df_x.groupby(['customer_id', 'coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})
cust_coup_brand_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_coup_brand_hist_trans.columns.values]
cust_coup_brand_hist_trans = cust_coup_brand_hist_trans.reset_index()
cust_coup_brand_hist_trans.shape
cust_coup_brand_hist_trans.head()
cust_coup_brand_hist_trans.to_csv(DIRE_N+'7-cust_coup_brand_hist_trans.csv', index=False)

#In[14]