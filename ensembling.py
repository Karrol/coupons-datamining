#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[15]:


df_preds = None
for i, f in enumerate([
    'xgb-124-9119-gkf-camp-test.csv',
    'lgb-124-9130-gkf-camp-test.csv',
    'catb-124-9142-gkf-camp-test.csv',
    
    'lgb-124-9418-gkf-coup-test.csv',
    'xgb-124-9514-gkf-coup-test.csv',
    'catb-124-9528-gkf-coup-test.csv',
    ]):
    if df_preds is None:
        df_preds = pd.read_csv(f)
        df_preds.columns = ['id', 'pred_{}'.format(i)]
    else:
        df1 = pd.read_csv(f)
        df1.columns = ['id', 'pred_{}'.format(i)]
        df_preds = df_preds.merge(df1, on ='id', how='left')


# In[16]:


test_ids = df_preds.id
df_preds = df_preds.drop(columns=['id'])


# In[17]:


pd.DataFrame({"id": test_ids, "redemption_status": df_preds.mean(axis=1)}).to_csv('final_submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




