```
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# importing the data

# In[5]:


data = pd.read_excel(r"C:\Users\DELL\Documents\inventory of steel rods\cleaned.xlsx")
data


# identifying the null values

# In[13]:


data.isna()


# identifying the duolicate values

# In[6]:


a = data.duplicated()
a


# sum of the duplicate values

# In[7]:


sum(a)


# identifying the 1st quartile and third qurtile
# and upper limit and lower limit for the QUANTITY COLUMN

# In[6]:


Q1 = data['quantity'].quantile(0.25)
Q3 = data['quantity'].quantile(0.75)
IQR = Q3 - Q1
print("The inter quartile range is:",IQR)
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("lower limit is:",lower_limit)
print("upper limit is:",upper_limit)


# identifying the outliers using the lower and upper_limit

# In[8]:


outliers = data[(data['quantity'] < lower_limit) | (data['quantity'] > upper_limit)]
print("The outliers are:",outliers)


# In[16]:


a = np.sum(outliers)
a


# identifying the Q1 & Q3 and upper_limi and lower_limit for the rate column

# In[5]:


Q1 = data['rate'].quantile(0.25)
Q3 = data['rate'].quantile(0.75)
IQR = Q3 - Q1
print("The IQR is:", IQR)
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("The upper limit for rate  is:", upper_limit)
print("the lower limit rate  is:", lower_limit)


# finding the outliers

# In[19]:


outliers = data[(data['rate'] < lower_limit) | (data['rate'] > upper_limit)]
print("The outliers are:", outliers)


# using the winsorization removing the outliers capping mathod

# In[24]:


from feature_engine.outliers import Winsorizer


# In[45]:


a = Winsorizer(capping_method='iqr', tail='both',fold =1.5, variables =['quantity'])


# In[46]:


b = a.fit_transform(data[['quantity']])


# In[47]:


sns.boxplot(b.quantity)


# identifying the ouliers in rate column

# In[48]:


sns.boxplot(data.rate)


# four busniess decisions for quantity 

# 1st busniess decision: mean, median, mode.

# In[60]:


x= data.quantity.mean()
print("quantity mean:",x)
y = data.rate.mean()
print("rate mean:", y)


# In[61]:


x = data.quantity.median()
print("quantity median:",x)
y = data.rate.median()
print("rate median:",y)


# In[62]:


x= data.quantity.mode()
print("quantity mode:",x)
y = data.rate.mode()

print("rate mode:",y)


# 2nd business decision: std, var, range

# In[63]:


x = data.quantity.std()
print("quantity std:",x)
y = data.rate.std()
print("rate std:",y)


# In[64]:


x =data.quantity.var()
print("quantity var",x)
y = data.rate.var()
print("rate var:",y)


# In[56]:


range = max(data.quantity) - min(data.quantity)
range


# In[65]:


range = max(data.rate) - min(data.rate)
range


# 3rd business decision: skewness

# In[66]:


x = data.quantity.skew()
print("quantity skew",x)
y = data.rate.skew()
print("rate skew:",y)
    


# 4th business decision kurtosis

# In[68]:


x = data.quantity.kurt()
print("quantity kurtosis",x)
y = data.rate.kurt()
print("rate kurtosis:",y)


# In[69]:


data.describe()


# # univariate: which mean it is having only one variable
#     example: bar plot, box plot, histogram, desity plot and q-q plot

# histogram for quantity

# In[73]:


plt.hist(data.quantity)


# histogram for rate

# In[74]:


plt.hist(data.rate)


# boxplot for quantity

# In[79]:


plt.boxplot(b.quantity)


# boxplot for rate

# In[80]:


plt.boxplot(data.rate)


# barplot for quantity and rate

# In[84]:


sns.barplot(data.quantity)


# In[85]:


sns.barplot(data.rate)


# quantile quantile plot for quantity and rate

# In[93]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
a = sm.qqplot(data.quantity)
plt.show()


# In[94]:


b = sm.qqplot(data.rate)
plt.show()


# # bivariate: which mean it is having two variable
#     example: scatter plot
#         

# In[111]:


plt.scatter(x = data['quantity'], y = data['rate'],color = 'green')


# # correlation coefficient: ranges from -1 to +1

# In[112]:


data.corr()


# # multivariate plot: having more than 2 variables
#     exapmles: pair plot and interaction plot
#         

# In[117]:


df = pd.DataFrame(data)
sns.pairplot(df)


# In[14]:


from sklearn.preprocessing import RobustScaler


# In[123]:


robust_model = RobustScaler()


# In[15]:


a1 = data.describe()
a1


# robust scaling: if you have already removed the outliers from your data, you might not necessarily need to use robust scaling. The primary motivation behind robust scaling is to make your data less sensitive to outliers during the scaling process.
# 
# When outliers are present in your data, robust scaling can be beneficial because it uses the median and interquartile range (IQR) instead of the mean and standard deviation, making it less influenced by extreme values.

# In[125]:


df_robust = robust_model.fit_transform(a1)
dataset_robust = pd.DataFrame(df_robust)

res_robust = dataset_robust.describe()
res_robust


# # Auto EDA:
sweet viz:
# In[5]:


pip install sweetviz


# In[1]:


import sweetviz as sv


# In[6]:


a = sv.analyze(data)
a.show_html()


# Autoviz:

# In[11]:


pip install Autoviz


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
from autoviz.AutoViz_Class import AutoViz_Class


# In[13]:


av = AutoViz_Class()
a = av.AutoViz(r"C:\Users\DELL\Downloads\cleaned.xlsx")


# In[14]:


import dtale


# In[15]:


d = dtale.show(data)
d.open_browser()


# In[5]:


import scipy.stats as stats
import pylab


# In[7]:


prb = stats.probplot(data.quantity, dist = stats.norm, plot = pylab)


# In[ ]:




```
