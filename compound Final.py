
# coding: utf-8

# ## Compound
# ### To rank the top ten US cities (with more than 1 million residents) according to the following criteria:
# - population growth
# - job growth
# - affordability (per capita income vs average real estate prices)
# - decline in real estate prices since peak prices

# **Importing Libraries**

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing Dataset**

# In[5]:


df1 = pd.read_csv('top_ten.csv')
df1.head(4)


# **Lets Rename the Column names for easy usage **

# In[6]:


df1.columns = map(str.lower, df1.columns)
df1.rename({"2018 population":"2018_population","2016 population":"2016_population","2010 census":"2010_population","change":"avg_pop_growth","2018 density":"2018_density","name":"city"}, inplace = True, axis = 1)
df1.head()


# **Lets Deal with cities with more than 1 Million Residents as mentioned in the question**

# In[7]:


df1 = df1.loc[(df1['2018_population'] > 1000000)]
df1


# ** cleaning of data is not much required**

# In[8]:


del df1['2010_population']
del df1['rank']
df1


# **According to "Forbes", "WalletHub","Apartmentlist.com" I am adding relevant columns in helping us to rank the cities**

# In[9]:


job_growth = [2, 2.6, 1.5, 0.1, 3.3, 2.2, 2.9, 2.6, 3.9, 3.1]
median_income = [64605,61931,65999,61444,58173,50702,55576,70141,62673,107290]
median_homeprice = [385700,533900,258600,233500,244300,163500,215100,594300,265800,1122900]
unemployment = [4.3,4.6,4.5,5.5,4.4,5.7,3.9,4.2,4.0,3.4]
rental_rate = [1.0,1.9,-0.8,3.4,2.2,1.2,0.9,1.8,1.1,2.2]
real_estate_ranking = [258,150,254,165,135,232,249,83,36,61]
columns = [job_growth,median_income,median_homeprice,unemployment,rental_rate,real_estate_ranking]
column_names = ['job_growth','median_income','median_homeprice','unemployment','rental_rate','real_estate_ranking']
for i in range(0,6): 
    df1[column_names[i]] = columns[i]
df1


# **1. Ranking cities according to population growth**

# In[50]:


pg = df1.nlargest(10,'avg_pop_growth')
idx = 0
idy = 2
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
pg.insert(loc=idx, column='score', value=weightage)
pg.insert(loc=idx, column='rank', value=rank)
pg["points"] = pg.score * 7
pg = pg.reindex(['points'] + list(pg.columns[:14]), axis=1)
pg


# **2. Ranking Cities according to job growth**

# In[51]:


jg = df1.nlargest(10, 'job_growth')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
jg.insert(loc=idx, column='score', value=weightage)
jg.insert(loc=idx, column='rank', value=rank)
jg["points"] = jg.score * 5
jg = jg.reindex(['points'] + list(jg.columns[:14]), axis=1)
jg


# **3. Ranking cities according to Easy Affordability**

# In[12]:


df1['affordability'] = df1.median_homeprice - df1.median_income
af = df1.nsmallest(10,'affordability')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
af.insert(loc=idx, column='score', value=weightage)
af.insert(loc=idx, column='rank', value=rank)
af["points"] = af.score * 6
af = af.reindex(['points'] + list(af.columns[:15]), axis=1)
af


# ***By this its clear that people in Philadelphia can highly afford their own home by their Average Income***

# **4. Ranking cities according to medium size home Increased rental rates**

# In[13]:


rr = df1.nlargest(10,'rental_rate')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
rr.insert(loc=idx, column='score', value=weightage)
rr.insert(loc=idx, column='rank', value=rank)
rr["points"] = rr.score * 2
rr = rr.reindex(['points'] + list(rr.columns[:15]), axis=1)
rr


# **5. Ranking cities according to RealEstate Market Ranking in that particular place**

# In[14]:


rer = df1.nsmallest(10,'real_estate_ranking')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
rer.insert(loc=idx, column='score', value=weightage)
rer.insert(loc=idx, column='rank', value=rank)
rer["points"] = rer.score * 3
rer = rer.reindex(['points'] + list(rer.columns[:15]), axis=1)
rer


# **6. Ranking Cities according to Unemployment Ratio**

# In[15]:


ue = df1.nsmallest(10,'unemployment')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
ue.insert(loc=idx, column='score', value=weightage)
ue.insert(loc=idx, column='rank', value=rank)
ue["points"] = ue.score * 1
ue = ue.reindex(['points'] + list(ue.columns[:15]), axis=1)
ue


# **7. Ranking cities according to population Density**

# In[16]:


df1['2018_density'] = df1['2018_density'] * 0.386102


# In[17]:


p_den = df1.nsmallest(10,'2018_density')
idx = 0
rank = [1,2,3,4,5,6,7,8,9,10]
weightage = [10,9,8,7,6,5,4,3,2,1]
p_den.insert(loc=idx, column='score', value=weightage)
p_den.insert(loc=idx, column='rank', value=rank)
p_den["points"] = p_den.score * 4
p_den = p_den.reindex(['points'] + list(p_den.columns[:15]), axis=1)
p_den


# * Here we can see Phoenix top's the list being less denser, by this we can say that there is more available area for construction and other development purpose which will be useful for our investor to buy land at a cheap rate initially

# **Since a Investor wants to Know how to Overall Rank these cities with the above criteria Iam Giving score to each and every city aaccording to their rank, score lies from 10 to 1 for city with rank 1 to 10.**
# 
# **Weightage to every criteria is mentioned below taking investor's point of view which various from individual to individual and can be modified later**
# 
# 1. Population growth = 7
# 
# 2. Affordability = 6
# 
# 3. Job growth = 5
# 
# 4. Less Population Density = 4
# 
# 5. Real Estate Market Ranking = 3
# 
# 6. Rental Rates = 2
# 
# 7. Unemployment = 1

# ** Now lets get the overall score for each city**

# In[56]:


dict_1 = {'Phoenix':[],'San Antonio ':[],'Houston ':[],'Dallas ':[],'San Diego':[],'San Jose':[],'Los Angeles':[],'Philadelphia':[],'Chicago ':[],'New York ':[]} 

for i in range(0,10):
    dict_1[af['city'][i]].append(af['points'][i])
    dict_1[p_den['city'][i]].append(p_den['points'][i]) 
    dict_1[ue['city'][i]].append(ue['points'][i])
    dict_1[rr['city'][i]].append(rr['points'][i]) 
    dict_1[rer['city'][i]].append(rer['points'][i])
    dict_1[pg['city'][i]].append(pg['points'][i])
    dict_1[jg['city'][i]].append(jg['points'][i])
       
dict_1


# In[69]:


city = ['Phoenix','San Antonio ','Houston ','Dallas ','San Diego','San Jose','Los Angeles','Philadelphia','Chicago ','New York ']
dict_2 = {'Phoenix':[],'San Antonio ':[],'Houston ':[],'Dallas ':[],'San Diego':[],'San Jose':[],'Los Angeles':[],'Philadelphia':[],'Chicago ':[],'New York ':[]} 
for j in range(0,10):
    dict_2[city[j]] = sum(dict_1[city[j]])

dict_2


# In[77]:


import warnings
warnings.filterwarnings("ignore")
a = [1,2,3,4,5,6,7,8,9,10]
point =sorted(dict_2.values(), reverse = True)
cities = []
for i in range(10):
    cities.append(list(dict_2.keys())[list(dict_2.values()).index(point[i])])
    
df_3 = pd.DataFrame({'rank':a,'score':point,'City':cities})
df_3['City'][2] = 'Dallas'
df_3


# ## Data Visiualization

# **1. Population Growth Graph**

# In[39]:


plt.bar(df1.city, df1.avg_pop_growth, align='center', alpha=0.5, color = 'green')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('Population Growth')
plt.title('Popultion growth Rate')
plt.show()


# **2. Job growth graph**

# In[84]:


plt.bar(df1.city, df1.job_growth, align='center', alpha=0.5, color = 'Blue')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('Job Growth')
plt.title('Job Growth Rate')
plt.show()


# **3. Affordability Graph**

# In[46]:


plt.bar(df1.city, df1.affordability, align='center', alpha=0.5, color = 'red')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('affordable difficulty')
plt.title('Affordability')
plt.show()


# **4. Cities Vs Rental Rates**

# In[92]:


plt.bar(df1.city, df1.rental_rate, align='center', alpha=0.5, color = 'yellow')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('rental rate')
plt.title('City Vs rental Rate')
plt.show()


# **5. Real Estate Ranking**

# In[97]:


np.random.seed(19680801)
N = 50
x = df1.city
y = df1.real_estate_ranking
area = (30 * np.random.rand(N)) # 0 to 15 point radii
plt.xticks( rotation = 90)
plt.scatter(x, y, s=area, alpha=0.5)
plt.show()


# **6. Unemployment Ratio Graph**

# In[211]:


plt.bar(df1.city, df1.unemployment, align='center', alpha=0.5, color = 'pink')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('unemployment')
plt.title('City Vs Unemployment')
plt.show()


# **7. Population Density Graph**

# In[331]:


plt.bar(df1.city, df1["2018_density"], align='center', alpha=0.5, color = 'gold')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('population density')
plt.title('City Vs population density')
plt.show()


# <center>**Top 10 U.S. cities with the best growth, employment, and business opportunities(having more than 1 Million residents)**</center>

# In[83]:


plt.bar(dict_2.keys(),dict_2.values(), align='center', alpha=0.5, color = 'brown')
plt.xticks( rotation = 90)
plt.xlabel('city')
plt.ylabel('score')
plt.title('City Vs Score')
plt.show()
df_3


#  ***By this we can say that Pheonix has the highest score of 241 and ranked 1st Position***

# <center>**Top 10 U.S. cities with the best growth, employment, and business opportunities(having more than 1 Million residents)**</center>

# In[78]:


a = [1,2,3,4,5,6,7,8,9,10]
point =sorted(dict_2.values(), reverse = True)
cities = []
for i in range(10):
    cities.append(list(dict_2.keys())[list(dict_2.values()).index(point[i])])
    
df_3 = pd.DataFrame({'rank':a,'score':point,'City':cities})
df_3
    


# **Conclusion**
# 
# The above results are drawn by giving weightage to every factor and score for each city according to their ranks while considering every factor. These weightage are given by looking at the business perspective of an investor. This weightage may vary for every different perspective and different purposes. 
# 
# 
# 
# 
# 
# 
# **Referance** 
# 
# * Dataset : https://www.kaggle.com/muonneutrino/us-census-demographic-data
# 
# * Rental rates : https://docs.google.com/spreadsheets/d/1eqCtTK0T0pSmTihu_ETUMKdzLdLMH3yZ12OaD_3tSnQ/edit#gid=0
# 
# * https://www.apartmentlist.com/rentonomics/rents-growing-fastest/
# 
# * Real Estate Market Raking: https://wallethub.com/edu/best-real-estate-markets/14889/#
# 
# * US city population growth: http://worldpopulationreview.com/us-cities/
# 
# * Job growth, affordability:  Forbes about each city
# 
# **Acknowledgements**
# 
# The data from Kaggle were collected by the US Census Bureau. As a product of the US federal government, this is not subject to copyright within the US.
# 
