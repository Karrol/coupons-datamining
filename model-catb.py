#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
pd.options.display.max_columns=1000 
pd.options.display.max_rows=1000
import numpy as np


# In[2]:


import sys
sys.path.append('ml_modules/')


# In[3]:


DATA_DIR = ''

train = pd.read_csv(DATA_DIR + 'train.csv')
"train", train.shape

test = pd.read_csv(DATA_DIR + 'test_QyjYwdj.csv')
"test", test.shape

campaign_data = pd.read_csv(DATA_DIR + 'campaign_data.csv')
campaign_data['start_date'] = pd.to_datetime(campaign_data.start_date, format='%d/%m/%y')
campaign_data['end_date'] = pd.to_datetime(campaign_data.end_date,  format='%d/%m/%y')
campaign_data['duration'] = (campaign_data.end_date - campaign_data.start_date).dt.days 


# In[ ]:



# #### Merging Created Features

# In[4]:


cust_hist_trans = pd.read_csv('cust_hist_trans.csv')
cust_hist_trans['start_date'] = pd.to_datetime(cust_hist_trans.start_date)

cust_hist_trans_daily = pd.read_csv('cust_hist_trans_daily.csv')
cust_hist_trans_daily['start_date'] = pd.to_datetime(cust_hist_trans_daily.start_date)

coup_hist_trans = pd.read_csv('coup_hist_trans.csv')
coup_hist_trans['start_date'] = pd.to_datetime(coup_hist_trans.start_date)

coup_hist_trans_daily = pd.read_csv('coup_hist_trans_daily.csv')
coup_hist_trans_daily['start_date'] = pd.to_datetime(coup_hist_trans_daily.start_date)

cust_coup_hist_trans = pd.read_csv('cust_coup_hist_trans.csv')
cust_coup_hist_trans['start_date'] = pd.to_datetime(cust_coup_hist_trans.start_date)

cust_coup_hist_trans_daily = pd.read_csv('cust_coup_hist_trans_daily.csv')
cust_coup_hist_trans_daily['start_date'] = pd.to_datetime(cust_coup_hist_trans_daily.start_date)

coup_brand_hist_trans = pd.read_csv('coup_brand_hist_trans.csv')
coup_brand_hist_trans['start_date'] = pd.to_datetime(coup_brand_hist_trans.start_date)

cust_brand_hist_trans = pd.read_csv('cust_brand_hist_trans.csv')
cust_brand_hist_trans['start_date'] = pd.to_datetime(cust_brand_hist_trans.start_date)

cust_coup_brand_hist_trans = pd.read_csv('cust_coup_brand_hist_trans.csv')
cust_coup_brand_hist_trans['start_date'] = pd.to_datetime(cust_coup_brand_hist_trans.start_date)


# In[5]:


def merge_dfs(df):
    df = df.merge(campaign_data, on='campaign_id', how='left', suffixes=['', '_camp']
                  
    ).merge(coup_hist_trans, on=['coupon_id', 'start_date'], how='left', suffixes=['', '_coht']
    ).merge(coup_hist_trans_daily, on=['coupon_id', 'start_date'], how='left', suffixes=['', '_cohtd']
            
    ).merge(cust_hist_trans, on=['customer_id', 'start_date'], how='left', suffixes=['', '_cuht']
    ).merge(cust_hist_trans_daily, on=['customer_id', 'start_date'], how='left', suffixes=['', '_cuhtd']
            
    ).merge(cust_coup_hist_trans, on=['customer_id', 'coupon_id','start_date'], how='left', suffixes=['', '_cucoht']
    ).merge(cust_coup_hist_trans_daily, on=['customer_id', 'coupon_id','start_date'], how='left', suffixes=['', '_cucohtd']
            
    ).merge(coup_brand_hist_trans, on=['coupon_id', 'start_date'], how='left', suffixes=['', '_cobrht']
    ).merge(cust_brand_hist_trans, on=['customer_id', 'start_date'], how='left', suffixes=['', '_cubrht']
    ).merge(cust_coup_brand_hist_trans, on=['customer_id', 'coupon_id','start_date'], how='left', suffixes=['', '_cucobrht']
            
    )
    df.shape
    df.head()
    return df
train.shape, test.shape
train = merge_dfs(train)
test = merge_dfs(test)

train.shape, test.shape


# In[ ]:





# In[6]:


train.columns.tolist()


# In[7]:


id_col = 'id'
target_col = 'redemption_status'

columns_to_drop = []
columns_to_drop = [c for c in columns_to_drop if c in train.columns]

cat_cols = [
    'campaign_id',
    'coupon_id',
    'customer_id',
    'campaign_type',
    'start_date',
    'end_date',   
]
cat_cols = [c for c in cat_cols if c in train.columns]
cat_cols



# In[8]:


# imputing categorical columns
train[cat_cols] = train[cat_cols].fillna('Missing')
test[cat_cols] = test[cat_cols].fillna('Missing')

# imputing numerical columns
train = train.fillna(-1)
test = test.fillna(-1)


# In[9]:


group_col = train['campaign_id'].astype(str) 
group_col_coupon = train['coupon_id'].astype(str) 


# In[10]:


train.shape
train.head()


# In[ ]:





# #### Importing modelling dependencies 

# In[11]:


from custom_estimator import Estimator
from encoding import FreqeuncyEncoding, LabelEncoding
from custom_fold_generator import FoldScheme
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# In[12]:


# fE = FreqeuncyEncoding(categorical_columns=cat_cols, return_df=True, normalize=False)
lE = LabelEncoding(categorical_columns=cat_cols, return_df=True)


# In[13]:


train = lE.fit_transform(train)
test = lE.transform(test)


# In[14]:


# train = fE.fit_transform(train)
# test = fE.transform(test)


# In[15]:


train.head()


# In[16]:


test_ids = test[id_col]
train_ids = train[id_col]


# In[17]:


y = train[target_col]
train.drop(columns=[x for x in [id_col] + columns_to_drop + [target_col] if x in train.columns], inplace=True)
test.drop(columns=[x for x in [id_col] + columns_to_drop + [target_col] if x in test.columns], inplace=True)


# In[18]:


train.shape
train.head()


# In[19]:


cat_cols_indices = [train.columns.tolist().index(x) for x in cat_cols]
cat_cols_indices


# In[20]:


test.head()



# #### Catboost GroupKFold on Campaign ID

# In[25]:


catb = Estimator(CatBoostClassifier(learning_rate=0.01, eval_metric='AUC', od_wait=400, iterations=10000), 
    early_stopping_rounds=400, n_splits=5, random_state=100, variance_penalty=1, verbose=100,
    eval_metric='AUC', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col,
    categorical_features_indices = cat_cols_indices
)

catb_oof = catb.fit_transform(train.values, y.values)
catb.avg_cv_score


# In[26]:


catb_preds = catb.transform(test[train.columns].values)


# In[27]:


catb.save_model(file_name='catb-124-9142-gkf-camp.pkl')


# In[28]:


pd.DataFrame({"id": train_ids, "redemption_status": catb_oof}).to_csv('catb-124-9142-gkf-camp-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": catb_preds}).to_csv('catb-124-9142-gkf-camp-test.csv', index=False)


# #### Catboost GroupKFold on CouponID

# In[29]:


catb = Estimator(CatBoostClassifier(learning_rate=0.01, eval_metric='AUC', od_wait=400, iterations=10000), 
    early_stopping_rounds=400, n_splits=5, random_state=100, variance_penalty=1, verbose=100,
    eval_metric='AUC', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col_coupon,
    categorical_features_indices = cat_cols_indices
)

catb_oof = catb.fit_transform(train.values, y.values)
catb.avg_cv_score


# In[30]:


catb_preds = catb.transform(test[train.columns].values)


# In[31]:


catb.save_model(file_name='catb-124-9528-gkf-coup.pkl')


# In[32]:


pd.DataFrame({"id": train_ids, "redemption_status": catb_oof}).to_csv('catb-124-9528-gkf-coup-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": catb_preds}).to_csv('catb-124-9528-gkf-coup-test.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




