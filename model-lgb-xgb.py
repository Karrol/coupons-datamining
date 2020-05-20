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





# 
# #### Importing modelling dependencies 

# In[11]:


from custom_estimator import Estimator
from encoding import FreqeuncyEncoding, LabelEncoding
from custom_fold_generator import FoldScheme
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# In[12]:


fE = FreqeuncyEncoding(categorical_columns=cat_cols, return_df=True)
# lE = LabelEncoding(categorical_columns=label_cat_cols, return_df=True)


# In[13]:


# train = lE.fit_transform(train)
# test = lE.transform(test)


# In[14]:


train = fE.fit_transform(train)
test = fE.transform(test)


# In[15]:


train.head()


# In[16]:


train.dtypes


# In[17]:


test_ids = test[id_col]
train_ids = train[id_col]


# In[18]:


y = train[target_col]
train.drop(columns=[x for x in [id_col] + columns_to_drop + [target_col] if x in train.columns], inplace=True)
test.drop(columns=[x for x in [id_col] + columns_to_drop + [target_col] if x in test.columns], inplace=True)


# In[19]:


train.shape
train.head()


# In[ ]:





# In[ ]:





# #### LightGBM GroupKFold on CampaignID

# In[20]:


params={
    'n_estimators': 20000, 
    'learning_rate': 0.1,
    'boosting_type': 'gbdt', 
    'colsample_bytree': 0.80,        
    'min_child_weight': 40.0,
    'num_leaves': 138, 
    'objective': 'binary', 
    'subsample': 0.50, 
    'subsample_freq': 5,
    'metric': 'custom'
}

lgb = Estimator(
    LGBMClassifier(**params), early_stopping_rounds=500, n_splits=5, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='AUC', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col
)

lgb_oof = lgb.fit_transform(train.values, y.values)
lgb.avg_cv_score


# In[21]:


lgb_preds = lgb.transform(test[train.columns].values)


# In[22]:


lgb.save_model(file_name='lgb-124-9130-gkf-camp.pkl')
pd.DataFrame({"id": train_ids, "redemption_status": lgb_oof}).to_csv('lgb-124-9130-gkf-camp-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": lgb_preds}).to_csv('lgb-124-9130-gkf-camp-test.csv', index=False)


# In[ ]:





# #### LightGBM GroupKFold on CouponId

# In[23]:


params={
    'n_estimators': 20000, 
    'learning_rate': 0.01,
    'boosting_type': 'gbdt', 
    'colsample_bytree': 0.80,        
    'min_child_weight': 40.0,
    'num_leaves': 138, 
    'objective': 'binary', 
    'subsample': 0.50, 
    'subsample_freq': 5,
    'metric': 'custom'
}

lgb = Estimator(
    LGBMClassifier(**params), early_stopping_rounds=500, n_splits=5, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='AUC', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col_coupon
)

lgb_oof = lgb.fit_transform(train.values, y.values)
lgb.avg_cv_score


# In[24]:


lgb_preds = lgb.transform(test[train.columns].values)


# In[25]:


lgb.save_model(file_name='lgb-124-9418-gkf-coup.pkl')
pd.DataFrame({"id": train_ids, "redemption_status": lgb_oof}).to_csv('lgb-124-9418-gkf-coup-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": lgb_preds}).to_csv('lgb-124-9418-gkf-coup-test.csv', index=False)


# In[ ]:





# In[ ]:





# #### XGBoost GroupKFold on CampaignID

# In[26]:


params={
    'colsample_bytree': 0.8,
    'gamma': 0.9,
    'learning_rate': 0.1,
    'max_depth': 4,
    'min_child_weight': 10.0,
    'n_estimators': 10000,
    'objective': 'binary:logistic',
    'subsample': 0.8
}

xgb = Estimator(XGBClassifier(**params), early_stopping_rounds=500, n_splits=5, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='auc', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col
)

xgb_oof = xgb.fit_transform(train.values, y.values)
xgb.avg_cv_score


# In[27]:


xgb_preds = xgb.transform(test[train.columns].values)


# In[28]:


xgb.save_model(file_name='xgb-124-9119-gkf-camp.pkl')
pd.DataFrame({"id": train_ids, "redemption_status": xgb_oof}).to_csv('xgb-124-9119-gkf-camp-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": xgb_preds}).to_csv('xgb-124-9119-gkf-camp-test.csv', index=False)


# In[ ]:





# In[ ]:





# #### LightGBM GroupKFold on CouponID

# In[29]:


params={
    'colsample_bytree': 0.8,
    'gamma': 0.9,
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 10.0,
    'n_estimators': 10000,
    'objective': 'binary:logistic',
    'subsample': 0.8
}

xgb = Estimator(XGBClassifier(**params), early_stopping_rounds=500, n_splits=5, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='auc', scoring_metric=roc_auc_score, 
    validation_scheme=FoldScheme.GroupKFold, cv_group_col=group_col_coupon
)

xgb_oof = xgb.fit_transform(train.values, y.values)
xgb.avg_cv_score


# In[30]:


xgb_preds = xgb.transform(test[train.columns].values)


# In[31]:


xgb.save_model(file_name='xgb-124-9514-gkf-coup.pkl')
pd.DataFrame({"id": train_ids, "redemption_status": xgb_oof}).to_csv('xgb-124-9514-gkf-coup-oof.csv', index=False)
pd.DataFrame({"id": test_ids, "redemption_status": xgb_preds}).to_csv('xgb-124-9514-gkf-coup-test.csv', index=False)


# In[ ]:





# In[ ]:




