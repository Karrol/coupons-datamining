#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#处理jupyter notebook的代码
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#程序正式开始
import pandas as pd
pd.options.display.max_columns=1000
pd.options.display.max_rows=1000
import numpy as np
import multiprocessing as mp
import time
import sqlite3

# In[7]:
def get_mode(x):
    try:
        return x.value_counts().index[0]
    except:
        return np.nan
    
def get_mode_count(x):
    try:
        return x.value_counts().values[0]
    except:
        return np.nan
    
"""
def parse_dict_of_dict(_dict, _str = ''):
    ret_dict = {}
    for k, v in _dict.items():
        if isinstance(v, dict):
            ret_dict.update(parse_dict_of_dict(v, _str= '_'.join([_str, k]).strip('_')))
        elif isinstance(v, list):
            for index, item in enumerate(v):
                if isinstance(item, dict):
                    ret_dict.update(parse_dict_of_dict(item,  _str= '_'.join([_str, k, str(index)]).strip('_')))
                else:
                    ret_dict['_'.join([_str, k, str(index)]).strip('_')] = item
        else:
            try:
                ret_dict['_'.join([_str, k]).strip('_')] = str(v)
            except Exception as e:
                ret_dict['_'.join([_str, k]).strip('_')] = unicode.encode(v, errors='ignore')
    return ret_dict
"""
#新增函数，解决merge时的memoryerror
def merge_with_chunk(target_df,chunk_df,output_file_name,on_cloums,how_tomerge='left'):
    """
        函数用于分块合并两个表
        target_df：小表
        chunk_df：大表，待拆分
        output_file_name：最终合并成的表的名字
        on_cloums：依据哪些列进行合并，这要求on_cloums是一个列表
        how_tomerge:merge的方式，取值为left/right/inner...
    """
    ##将大表转存出去，然后分块读取,每行的index忽视，这样合并的时候不会带上index这个没用的列
    chunk_df.to_csv('chunks_for_'+str(output_file_name),index=False)
    reader=pd.read_csv('chunks_for_'+str(output_file_name),chunksize=1000)
    ##将合并后的结果表也转存出去，并且分块追加写入
    for chunk in reader:
        ###分块合并表
        df = pd.merge(target_df, chunk, on=on_cloums, how=str(how_tomerge))
        ###mode='a'代表追加数据，a->append
        df.to_csv(str(output_file_name), mode="a", header=True,index=False)
    df=pd.read_csv(str(output_file_name),chunksize=100000)
    return df

def csv2sqlite(csvPath,csvTablename):
    db=sqlite3.connect("coupondata.db")
    try:
        for c in pd.read_csv(csvPath,chunksize=100000):
            try:
                #Append al rows to a new database table
                c.to_sql(csvTablename,db,if_exists='append')
            except Exception as e:
                print("警告：转换过程中部分chunks导入数据库错误！")
                pass
            continue
        db.close()
        return True
    except Exception as e:
        print("转换过程出现异常,请重新检查")
        return False

def dfchunks2sqlite(df,tablename):
    db=sqlite3.connect("coupondata.db")
    try:
        for c in df:
            try:
                #Append al rows to a new database table
                c.to_sql(tablename,db,if_exists='append')
            except Exception as e:
                print("警告：转换过程中部分chunks导入数据库错误！")
                pass
            continue
        db.close()
        return True
    except Exception as e:
        print("转换过程出现异常,请重新检查")
        return False

def df2sqlite(df,tablename):
    db=sqlite3.connect("coupondata.db")
    try:
        try:
                #Append al rows to a new database table
            df.to_sql(tablename,db,if_exist='fail')
        except Exception as e:
            print("警告：表已存在，不进行覆盖")
            pass
        db.close()
        return True
    except Exception as e:
        print("转换过程出现异常,请重新检查")
        db.close()
        return False


# In[2]:
#数据文件目录，注意带上斜杠表示自己是个目录
DATA_DIR = 'data/'
#读取train数据
train = pd.read_csv(DATA_DIR + 'train.csv')
df2sqlite(train,'train')
"train",train.shape 
#读取train数据
test = pd.read_csv(DATA_DIR + 'test_QyjYwdj.csv')
"test", test.shape
df2sqlite(test,'test')

campaign_data = pd.read_csv(DATA_DIR + 'campaign_data.csv')
"campaign_data", campaign_data.shape
campaign_data['start_date'] = pd.to_datetime(campaign_data.start_date, format='%d/%m/%y')
campaign_data['end_date'] = pd.to_datetime(campaign_data.end_date,  format='%d/%m/%y')
campaign_data['duration'] = (campaign_data.end_date - campaign_data.start_date).dt.days 
df2sqlite(campaign_data,'campaign_data')

customer_demographics = pd.read_csv(DATA_DIR + 'customer_demographics.csv')
"customer_demographics", customer_demographics.shape 
df2sqlite(customer_demographics,'customer_demographics')

customer_transaction_data = pd.read_csv(DATA_DIR + 'customer_transaction_data.csv')
"customer_transaction_data", customer_transaction_data.shape 
customer_transaction_data = customer_transaction_data.drop_duplicates()
"customer_transaction_data", customer_transaction_data.shape
customer_transaction_data['date'] = pd.to_datetime(customer_transaction_data.date, format='%Y-%m-%d')
df2sqlite(customer_transaction_data,'customer_transaction_data') 

item_data = pd.read_csv(DATA_DIR + 'item_data.csv')
"item_data",item_data.shape
df2sqlite(item_data,'item_data') 

coupon_item_mapping = pd.read_csv(DATA_DIR + 'coupon_item_mapping.csv')
"coupon_item_mapping", coupon_item_mapping.shape
df2sqlite(coupon_item_mapping,'coupon_item_mapping')

# In[4]:
##合并数据，新test是带有优惠券活动数据的
##test和train表没有实际意义，拥有的字段是campaign_id，customer_id，coupon_id，是链接各个表的枢纽
df_test = test.merge(campaign_data, how='left')
df_test.shape
df_test.head()

# In[5]:
##合并数据，新train是带有优惠券活动数据的
df_train = train.merge(campaign_data, how='left')
df_train.shape
df_train.head()

# In[6]:
##查看数据日期最值，准备获取交易数据在优惠券活动日期内的用户数据
##用户消费交易表中的日期最大最小
customer_transaction_data.date.min(), customer_transaction_data.date.max()
##test和train中优惠券活动的最大最小日期
df_train.start_date.min(), df_train.end_date.max()
df_test.start_date.min(), df_test.end_date.max()


# In[7]:
##pd.contact()函数，不去重的全连接，axis指定延什么轴方向进行
##这里是x轴，相当于在train数据行后面添加了test数据行，没有新增的列
##对于df_all和df_train的shape就可以理解
df_all = pd.concat([df_train, df_test], sort=False, axis=0)#pd.concat（）函数，第一个参数要传递一个list之类的迭代器
df_all.shape
df_all.head()
df2sqlite(df_all,'train_test_duplicate')
# In[8]:
##合并交易数据和商品信息
cust_transactions = customer_transaction_data.merge(item_data, on = 'item_id', how='left')
cust_transactions.shape
cust_transactions.head()
##清洗合并后的数据字段的数据类型
##折扣抵消的费用全部转为正数
cust_transactions['coupon_discount'] = cust_transactions.coupon_discount.abs()
cust_transactions['other_discount'] = cust_transactions.other_discount.abs()
##判断用户是否使用了消费券，创造新列，将布尔值转化为int
cust_transactions['coupon_applied'] = (cust_transactions['coupon_discount'] > 0).astype(int)
##处理过后的交易表内含有：交易数据-商品信息-是否使用优惠券lable
cust_transactions.head()
df2sqlite(cust_transactions,tablename='cust_transactions_leftmerge_item')#cust_transactions数据量很大，清洗过数据后就存档

# In[9]:
##对compaign-train-test合并后的数据进行去重
df_ = df_all[['campaign_id', 'coupon_id', 'customer_id', 'redemption_status', 'campaign_type', 'start_date', 'end_date']].drop_duplicates()
df_.shape

# In[36]:
#to do:debug here
##内存错误，可能原因：NAN过多，或者单纯的内存不够###尝试解决方案1，分块merge->memory error解决成功
####查看哪个表的更大
#####复制出compaign-train-test中的顾客id和活动开始日期，保险起见进行去重
df_x=pd.DataFrame(df_all,columns=['customer_id', 'start_date']).drop_duplicates()
#这样作的话df_x是tuple不是df,df_x = df_all[['customer_id', 'start_date']].drop_duplicates()
df_x.shape#(6940, 2)
cust_transactions.shape#(1321650, 11)
#分块合并两个数据
##和交易数据进行连接，把merge函数的默认参数补上，how='inner'
##连接后只保留了两张表都有的顾客（customer_id）的数据
"""暂时注释掉，跑一次这个函数时间代价比较大"""
df_x=merge_with_chunk(df_x,cust_transactions,output_file_name='df_all_trans_item.csv',on_cloums=['customer_id'],how_tomerge='inner')
# In[37]:
##含义：要筛选出某顾客收到优惠券的生效日期大于该顾客购物交易日期的数据
##当df的cloumn被赋值之后，可以采用df_x.start_date来引用这一列
#to do:debug here
##尝试pd能不能一次性带动900+M的文件->带不动，一定要分节读取df_x现在是个chunks
#df_x=pd.read_csv('df_all_trans_item.csv',chunksize=100000)
db=sqlite3.connect('coupondata.db')
if db.execute("SELECT count(*) FROM sqlite_master WHERE type=\"table\" AND name=\"df_all_trans_item_filter\""):
    for chunk in df_x:
        ###相当于SELECT * FROM table WHERE start_date>chunk.date
        chunk = chunk[chunk.start_date > chunk.date]#这个可以剔除列不符合要求的行么？可以
        ###挨个chunk进行剔除，并分块写出数据以合并chunks
        """chunk.to_csv('df_all_trans_item_filter.csv', mode="a", header=True)"""
        ####将df_all_trans_item_filter.csv写入sqlite中进行聚合函数处理
        chunk.to_sql('df_all_trans_item_filter',db,if_exists='append')
else:
    print('df_all_trans_item_filter已存在，请删除后再执行操作')
del df_x#df_x使命告一段落，清理掉它，释放内存！

# In[37]:
#在sqlite中注册聚合函数，为了能够实现求标准差和加权移动平均数
class std:
    def __init__(self):
        self.count=0
# In[10]:
#利用已有的变量，创造新的变量，并将新变量存入新表cust_hist_trans：
    #从某一优惠活动开始日，某顾客的购物商品数量，这些商品的总价、均价和方差
    #购买这些交易商品过程中总享受的“其他折扣”的总价和均价，优惠券折扣的总结和均价，优惠券使用次数的总数和平均数
##对df_all_trans_item_filter按照customer_id和date进行排序，为groupby做准备
db.execute("SELECT * FROM \"df_all_trans_item_filter\" order by customer_id,date ASC")
db.execute("CREATE INDEX customer_id_index on \"df_all_trans_item_filter\"(\"customer_id\")")
db.commit()
cursor=db.cursor()
cursor.execute("""
    SELECT 
""")

# In[11]:


#cust_hist_trans.to_csv('cust_hist_trans.csv', index=False)


# In[12]:

##生成新表，代表每个顾客每天的消费
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
##依据日期合顾客id排序
cust_daily = cust_daily.sort_values(by=['customer_id', 'date'])
##对cust_daily做聚合函数，计算加权移动平均数，让近期的消费权重，数据就附加在cust_daily表上，因为是每个时刻（天）都做一个计算
cust_daily['selling_price_ewm'] = cust_daily.groupby('customer_id')['selling_price'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_daily['coupon_discount_ewm'] = cust_daily.groupby('customer_id')['coupon_discount'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_daily['coupon_applied_ewm'] = cust_daily.groupby('customer_id')['coupon_applied'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()

# cust_daily.head()
##合并训练测试数据集-顾客的每天消费数据，得到优惠券和顾客每天交易数据匹配的数据，并筛选出优惠券生效的数据
###即顾客当天的交易数据，以及他拥有的优惠券，优惠券是否使用了
df_x = df_all[['customer_id', 'start_date']].drop_duplicates().merge(cust_daily, on=['customer_id'])
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['customer_id', 'start_date', 'date'])

###使用聚合函数，实现对顾客从收到优惠券后的每天的交易数据的聚合数据
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
cust_hist_trans_daily.shape
cust_hist_trans_daily.head()


# In[13]:


cust_hist_trans_daily.to_csv('cust_hist_trans_daily.csv', index=False)

# In[14]:

#获取归属于优惠券活动的商品-与顾客的交易数据，这张表中所有的商品都是参与优惠券活动的
coup_trans = coupon_item_mapping.merge(cust_transactions, on='item_id', how='left')
coup_trans.shape
coup_trans.head()


# In[15]:

#优惠券-活动开始日期，和用户交易数据合并，可以获得在促销活动中，顾客获取到的优惠券的商品的交易数据。cust_his_trans是所有商品的交易数据，一些商品可能并不能享受优惠券折扣
df_x = df_all[['coupon_id', 'start_date']].drop_duplicates().merge(coup_trans, on='coupon_id', how='left')
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['coupon_id', 'date'])
df_x.shape
##一样的套路，先筛选出活动期限内的消费数据（才有可能能够使用优惠券），并按照优惠券和交易日期排序
##获得各个优惠券被顾客的使用情况
##依据优惠券id和活动开始日期排序，统计从促销活动开始时不同优惠券的交易数据
coup_hist_trans = df_x.groupby(['coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})


coup_hist_trans.columns = ['_'.join(col).strip('_') for col in coup_hist_trans.columns.values]
coup_hist_trans = coup_hist_trans.reset_index()
coup_hist_trans.shape
coup_hist_trans.head()


# In[16]:


coup_hist_trans.to_csv('coup_hist_trans.csv', index=False)


# In[17]:

##依据优惠券id和交易日期排序，统计从不同优惠券每天的交易数据
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
##再一次，为不同优惠券每天的交易数据加权移动平均
coup_daily['selling_price_ewm'] = coup_daily.groupby('coupon_id')['selling_price'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
coup_daily['coupon_discount_ewm'] = coup_daily.groupby('coupon_id')['coupon_discount'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
coup_daily['coupon_applied_ewm'] = coup_daily.groupby('coupon_id')['coupon_applied'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()

##去重后再次依据coupon_id与已标记的训练数据合并
df_x = df_all[['coupon_id', 'start_date']].drop_duplicates().merge(coup_daily, on=['coupon_id'])
##之前按照coupon_id进行分组，现在筛选出在活动日期内的数据，
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['coupon_id', 'start_date', 'date'])

##优惠券的日交易数据
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
coup_hist_trans_daily.shape
coup_hist_trans_daily.head()


# In[18]:


coup_hist_trans_daily.to_csv('coup_hist_trans_daily.csv', index=False)



# In[19]:


df_x = df_.merge(coup_trans, on=['customer_id', 'coupon_id'], how='left')
df_x.shape
df_x = df_x[df_x.start_date > df_x.date]
df_x.shape

cust_coup_hist_trans = df_x.groupby(['customer_id', 'coupon_id', 'start_date']).agg({
            'item_id': ['count'],
            'selling_price': ['sum', 'mean', 'std'],
            'other_discount': ['sum', 'mean'],
            'coupon_discount': ['sum', 'mean'], 
            'coupon_applied': ['sum', 'mean'], 
})

cust_coup_hist_trans.columns = ['_'.join(col).strip('_') for col in cust_coup_hist_trans.columns.values]
cust_coup_hist_trans = cust_coup_hist_trans.reset_index()
cust_coup_hist_trans.shape
cust_coup_hist_trans.head()


# In[20]:


cust_coup_hist_trans.to_csv('cust_coup_hist_trans.csv', index=False)


# In[21]:


cust_coup_daily = coup_trans.groupby([ 'customer_id', 'coupon_id', 'date']).agg({
    'item_id': 'count',
    'quantity': 'sum',
    'selling_price': 'sum',
    'other_discount': 'sum',
    'coupon_discount': 'sum',
    'coupon_applied': 'sum'
})
cust_coup_daily = cust_coup_daily.reset_index()
cust_coup_daily = cust_coup_daily.sort_values(by=['customer_id', 'coupon_id', 'date'])

cust_coup_daily['selling_price_ewm'] = cust_coup_daily.groupby(['customer_id', 'coupon_id'])['selling_price'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_coup_daily['coupon_discount_ewm'] = cust_coup_daily.groupby(['customer_id', 'coupon_id'])['coupon_discount'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()
cust_coup_daily['coupon_applied_ewm'] = cust_coup_daily.groupby(['customer_id', 'coupon_id'])['coupon_applied'].apply(
    lambda x: x.ewm(com=0.5).mean()).tolist()

cust_coup_daily.shape
cust_coup_daily.head()

df_x = df_[['customer_id', 'coupon_id', 'start_date']].drop_duplicates().merge(cust_coup_daily, on=['customer_id', 'coupon_id'])
df_x = df_x[df_x.start_date > df_x.date]
df_x = df_x.sort_values(by=['customer_id', 'coupon_id', 'start_date', 'date'])
df_x.shape

cust_coup_hist_trans_daily = df_x.groupby(['customer_id', 'coupon_id', 'start_date']).agg({
    'date': ['count'],
    'selling_price': ['mean', 'std', 'last'],
    'other_discount': ['mean', 'std', 'last'],
    'coupon_discount': ['mean', 'std', 'last'],
    'coupon_applied': ['mean', 'std', 'last'],
    'selling_price_ewm': ['mean', 'std'],
    'coupon_discount_ewm': ['mean', 'std'],
    'coupon_applied_ewm': ['mean', 'std'],
})

cust_coup_hist_trans_daily.columns = ['_'.join(col).strip('_') for col in cust_coup_hist_trans_daily.columns.values]
cust_coup_hist_trans_daily = cust_coup_hist_trans_daily.reset_index()
cust_coup_hist_trans_daily.shape
cust_coup_hist_trans_daily.head()


# In[22]:


cust_coup_hist_trans_daily.to_csv('cust_coup_hist_trans_daily.csv', index=False)



# #### coupon brand

# In[23]:


coupon_items = coupon_item_mapping.merge(item_data, on='item_id')
coupon_items.shape
coupon_items.head()


# In[24]:


coupon_brands = coupon_items[['coupon_id', 'brand']].drop_duplicates()
coupon_brands.shape
coupon_brands.head()


# In[25]:


coupon_brand_trans = coupon_brands.merge(cust_transactions, on='brand', how='left')
coupon_brand_trans.shape


# In[26]:


coupon_brand_trans.head()


# In[27]:


df_x = df_.merge(coupon_brand_trans, on=['customer_id', 'coupon_id'], how='left')
df_x.shape
df_x = df_x[df_x.start_date > df_x.date]
df_x.shape


# In[28]:


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


# In[29]:


coup_brand_hist_trans.to_csv('coup_brand_hist_trans.csv', index=False)


# In[30]:


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


# In[31]:


cust_brand_hist_trans.to_csv('cust_brand_hist_trans.csv', index=False)



# In[32]:


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


# In[33]:


cust_coup_brand_hist_trans.to_csv('cust_coup_brand_hist_trans.csv', index=False)