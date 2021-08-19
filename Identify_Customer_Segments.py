#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv',delimiter=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',delimiter=';')


# In[4]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print('general_pop # cols:',len(azdias.columns))
print('general_pop # cols:',np.size(azdias,0))

print('general_pop # cols:',len(feat_info.columns))
print('general_pop # cols:',np.size(feat_info,0))


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[5]:


azdias.head()


# In[6]:


# Identify missing or unknown data values and convert them to NaNs.
def string_to_list(strg):
    new_list=[]
    s_clean=strg[1:-1].split(',')
    # The below line removes the '[]' and splits on ',', creating a list of strings
   
    for i in s_clean:
        try:
            new_list.append( int(i) )
        except:
                new_list.append( i )
    
    return( new_list )

feat_info['NA_tags'] = feat_info['missing_or_unknown'].apply(string_to_list)
att_index = feat_info.set_index('attribute') # set the 

na_azdias = azdias[:]
for column in na_azdias.columns:
    na_azdias[column].replace(att_index.loc[column].loc['NA_tags'],np.nan,inplace=True)
na_azdias.isna().sum().sum()


# In[7]:


na_azdias.head()


# In[8]:


print(att_index.head())


# In[9]:


feat_info.head()


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[10]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

num_nas_azdias = na_azdias.isna().sum().reset_index()


# In[11]:


# Investigate patterns in the amount of missing data in each column.
big = num_nas_azdias[num_nas_azdias[0]>2000]
plt.figure()
plt.hist(num_nas_azdias[0])


# In[12]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
new_azdias = num_nas_azdias[num_nas_azdias<160000].reset_index()
new_azdias.sort_values(by=0, ascending=False,inplace=True)
new_azdias[0].hist()


# In[13]:



li_dropcol=[]
for col in na_azdias:
    if na_azdias[col].isna().sum()>160000:
        li_dropcol.append(col)
print(li_dropcol)


# In[14]:


new_azdias = na_azdias.drop(columns=li_dropcol)


# In[15]:


print(new_azdias)


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# seems to be three big groups: 1 group 0-200k, 200k - 700k, 800k+
# 
# within the 0-200k group, theres a smaller group that can also be sorted down even further. Depends on what we want to analyze, and how 'clean' we want our data. the more ' clean' it is, the less data we actually have....

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[16]:


# How much data is missing in each row of the dataset?

num_nas_azdias_rows = new_azdias.isna().sum(axis = 1)
# num_nas_azdias_rows.sort_values(by=0, ascending=False,inplace=True)


num_nas_azdias_rows.hist()
num_nas_azdias_rows.head()


# In[17]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
na_azdias_rows_small = na_azdias[num_nas_azdias_rows <= 9]
na_azdias_rows_big = na_azdias[num_nas_azdias_rows > 9]


# In[18]:


na_big_cols = na_azdias_rows_big.columns


# In[19]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

# test.hist()
na_azdias_rows_big[na_big_cols[1]].hist()
plt.figure()
na_azdias_rows_small[na_big_cols[1]].hist()
plt.figure()
na_azdias_rows_big[na_big_cols[2]].hist()
plt.figure()
na_azdias_rows_small[na_big_cols[2]].hist()


# In[20]:


plt.figure()
na_azdias_rows_big[na_big_cols[3]].hist()
plt.figure()
na_azdias_rows_small[na_big_cols[3]].hist()
plt.figure()
na_azdias_rows_big[na_big_cols[4]].hist()
plt.figure()
na_azdias_rows_small[na_big_cols[4]].hist()


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# average miss values / row  = 9, 2 groupings: 1) 0-30, 2) 30-50

# In[21]:


na_azdias_rows_final = new_azdias[num_nas_azdias_rows <=9]


# In[22]:


na_azdias_rows_final.head()


# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[23]:


feat_info.head()


# In[24]:


# How many features are there of each data type?
feat_info.type.hist()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[25]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
category_total={}
category_binary=[]
category_multi=[]
li_encode=[]

for col in feat_info[feat_info['type']=='categorical'].attribute:
    if col in na_azdias_rows_final.columns:
        category_total[col] = np.size(na_azdias_rows_final[col].unique())

        if category_total[col] ==2:
            category_binary.append(col)
            li_encode.append(col)
        else: category_multi.append(col) 
        
print(category_total)


# In[26]:


print(li_encode)


# In[27]:


# Re-encode categorical variable(s) to be kept in the analysis.
print('binary:', category_binary)
print('multi:',category_multi,'\n\n\n')


# In[28]:



na_azdias_rows_final[category_binary[3]].head()
# print(category_binary[4])


# In[29]:


replace_values = {'W': 0, 'O': 1} 
azdias_cleaned = na_azdias_rows_final.replace({'OST_WEST_KZ':replace_values})


# In[30]:


azdias_cleaned[category_binary[3]].unique()


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
#  dropping multi for simplisity, re-engineered ost_west_kz

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[31]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
li_mixed=[]
for col in feat_info[feat_info['type']=='mixed'].attribute:
    if col in azdias_cleaned:

        li_mixed.append(col)
        if col == 'PRAEGENDE_JUGENDJAHRE':
            li_encode.append(col)
            print(azdias_cleaned[col].head())

        


# In[32]:


print(li_encode)


# In[33]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables. M is 1 and A is 0
encoding = {
    
          0:[None,None],
          1: [40,1],
          2: [40,0],
          3: [50,1],
          4:[50,0],
          5:[60,1],
          6:[60,0],
          7:[60,0],
          8:[70,1],
          9:[70,0],
          10:[80,1],
          11:[80,0],
          12:[80,1],
          13:[80,0],
          14:[90,1],
          15:[90,0]
}

azdias_values = azdias_cleaned["PRAEGENDE_JUGENDJAHRE"].values
decades=[]
movement=[]
for row in azdias_values:
    if math.isnan(row):
        decades.append(np.NaN)
        movement.append(np.NaN)         
    else:
        decades.append(encoding[row][0])
        movement.append(encoding[row][1])

azdias_cleaned["decades"] = decades
azdias_cleaned["movement"]= movement


# In[34]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
for col in feat_info[feat_info['type']=='mixed'].attribute:
    if col in azdias_cleaned.columns:
        if col == 'CAMEO_INTL_2015':
            li_encode.append(col)
            print(na_azdias_rows_final[col].head())

li_encode = list(set(li_encode))


# In[35]:


azdias_cleaned["wealth"]= azdias_cleaned["CAMEO_INTL_2015"].str[0]
azdias_cleaned["life"]= azdias_cleaned["CAMEO_INTL_2015"].str[1]


# In[36]:


azdias_cleaned['wealth'].head()


# In[37]:


category_multi


# In[38]:


li_mixed


# In[39]:


azdias_cleaned = azdias_cleaned.drop(category_multi,axis = 1)
azdias_cleaned = azdias_cleaned.drop(li_mixed,axis = 1)


# In[40]:


print(azdias_cleaned.head())


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# i dropped everything except for the stuff that i re-encoded personally.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[42]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)
# azdias_cleaned = pd.get_dummies(azdias_cleaned , columns=azdias_cleaned.columns )


# In[41]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.
azdias_cleaned.head()


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[132]:



def clean(df):
    df_gen_pop = df.copy()
# ====================# Identify missing or unknown data values and convert them to NaNs.================    

    feat_info['NA_tags'] = feat_info['missing_or_unknown'].apply(string_to_list)
    att_index = feat_info.set_index('attribute') # set the 

    na_azdias = df_gen_pop[:]
    for column in na_azdias.columns:
        na_azdias[column].replace(att_index.loc[column].loc['NA_tags'],np.nan,inplace=True)
    
# ====================Remove na's from rows and cols================        
    li_dropcol=[]
    for col in na_azdias:
        if na_azdias[col].isna().sum()>160000:
            li_dropcol.append(col)
    new_azdias = na_azdias.drop(columns=li_dropcol)
    # hard coded this in here:
    li_dropcol = ['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX']
    new_azdias = na_azdias.drop(columns=li_dropcol)
    #end hard code
    num_nas_azdias_rows = new_azdias.isna().sum(axis = 1)

    na_azdias_rows_final = new_azdias[num_nas_azdias_rows <=9]
# ====================1111111111111================    
            
    azdias_cleaned = na_azdias_rows_final.replace({'OST_WEST_KZ':{'W': 0, 'O': 1} })
# ====================1111111111111================    
    category_total={}
    category_binary=[]
    category_multi=[]
    li_encode=[]

    for col in feat_info[feat_info['type']=='categorical'].attribute:
        if col in na_azdias_rows_final.columns:
            category_total[col] = np.size(na_azdias_rows_final[col].unique())

            if category_total[col] ==2:
                category_binary.append(col)
                li_encode.append(col)
            else: category_multi.append(col) 
        
            
        
    li_mixed=[]
    for col in feat_info[feat_info['type']=='mixed'].attribute:
        if col in azdias_cleaned:

            li_mixed.append(col)
            if col == 'PRAEGENDE_JUGENDJAHRE':
                li_encode.append(col)
#                 print(azdias_cleaned[col].head())
    for col in feat_info[feat_info['type']=='mixed'].attribute:
        if col in azdias_cleaned.columns:
            if col == 'CAMEO_INTL_2015':
                li_encode.append(col)
#                 print(na_azdias_rows_final[col].head())

    li_encode = list(set(li_encode))
            
            
# ====================1111111111111================    
    encoding = {
    
          0:[None,None],
          1: [40,1],
          2: [40,0],
          3: [50,1],
          4:[50,0],
          5:[60,1],
          6:[60,0],
          7:[60,0],
          8:[70,1],
          9:[70,0],
          10:[80,1],
          11:[80,0],
          12:[80,1],
          13:[80,0],
          14:[90,1],
          15:[90,0]
}

    azdias_values = azdias_cleaned["PRAEGENDE_JUGENDJAHRE"].values
    decades=[]
    movement=[]
    for row in azdias_values:
        if math.isnan(row):
            decades.append(np.NaN)
            movement.append(np.NaN)         
        else:
            decades.append(encoding[row][0])
            movement.append(encoding[row][1])
# ====================1111111111111================    

    azdias_cleaned["decades"] = decades
    azdias_cleaned["movement"]= movement

    azdias_cleaned["wealth"]= azdias_cleaned["CAMEO_INTL_2015"].str[0]
    azdias_cleaned["life"]= azdias_cleaned["CAMEO_INTL_2015"].str[1]
# ====================1111111111111================    
    azdias_cleaned = azdias_cleaned.drop(category_multi,axis = 1)
    azdias_cleaned = azdias_cleaned.drop(li_mixed,axis = 1)
    
    azdias_cleaned = pd.get_dummies(azdias_cleaned , columns=azdias_cleaned.columns)


    
    return df


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[43]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

azdias_cleaned.isna().sum().hist()


# In[44]:


# Apply feature scaling to the general population demographics data.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

azdias_Cleaned = azdias_cleaned.copy()
imp = Imputer()
azdias_scaled = imp.fit_transform(azdias_Cleaned)
ss = StandardScaler()

scaled_azdias = ss.fit_transform(azdias_scaled)


# ### Discussion 2.1: Apply Feature Scaling
# 
# feature scaling reduces impact of biases, and increases impact of data with smaller values. I didnt need an imputer or to use .dropnan() because I already had no missing values after pre-processing.
# 
# Ideally fit all data between 0-1.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[45]:


# Apply PCA to the data.

from sklearn.decomposition import PCA
li_explained_variance = []
li_pca = [PCA(n_components) for n_components in range(1,60,10)]
for pca in li_pca:
    
    pca_azdias = pca.fit(scaled_azdias)
    li_explained_variance.append(pca.explained_variance_ratio_)
    


# In[46]:


plt.bar(range(np.size(li_explained_variance[5])),li_explained_variance[5])


# the above graph shows the indidual contribution of each component. The first 4 are very large in comparison to the others, with a big taper off. 

# In[47]:


print(li_explained_variance[5].cumsum())


# In[48]:


# Investigate the variance accounted for by each principal component.

plt.bar(range(np.size(li_explained_variance[5].cumsum())),li_explained_variance[5].cumsum())


# The above graph shows how the increasing  n_component impacts the overal strength of the pca. At about 30, you are at over 80%. From 30 - 51 you only get 20% increase. I have the option to reduce the complexity and go with a pca of 30, but reduce the strength of the model a bit.

# In[67]:


# Re-apply PCA to the data while selecting for number of components to retain.

pca = PCA(20)
pca_azdias = pca.fit_transform(scaled_azdias)


# In[68]:


print(pca_azdias)


# In[51]:


np.size(azdias_cleaned,axis =1)


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# retaining 60 pca components would cause the max cumsum of the explained variability. I have 63 total parameters, so 'reducing' down to 60 pca parameters makes sense that this would be the maximum. Good news as well because that means that most of our data is relevant. However, theres a big decrease in the slope after 40. I am goign to pick 40 because its a nice middle ground, with still a lot of relevancy. 

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[52]:


def pca_results(full_dataset, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

#     # Create a bar plot visualization
#     fig, ax = plt.subplots(figsize = (14,8))

#     # Plot the feature weights as a function of the components
#     components.plot(ax = ax, kind = 'bar');
#     ax.set_ylabel("Feature Weights")
#     ax.set_xticklabels(dimensions, rotation=0)


#     # Display the explained variance ratios
#     for i, ev in enumerate(pca.explained_variance_ratio_):
#         ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)


# In[70]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
pca_results(azdias_Cleaned,pca)


# In[54]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

def pca_results_individual(full_dataset, pca, pca_no):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    
    dimension=components.loc[  'Dimension '+str(pca_no)  ,  :].sort_values(ascending=False)
    
    # Return a concatenated DataFrame
    pd.concat([ components, variance_ratios], axis = 1)
    
    print('===POSITIVE===\n')
    print(dimension[dimension>0].head())
    print('===NEGATIVE===\n')
    dimension=components.loc[  'Dimension '+str(pca_no)   ,  :].sort_values(ascending=True)
    print(dimension[dimension<0].head())


# In[55]:


pca_results_individual(azdias_Cleaned,pca,1)


# In[56]:


pca_results_individual(azdias_Cleaned,pca,2)


# In[59]:


pca_results_individual(azdias_Cleaned,pca,3)


# ### Discussion 2.3: Interpret Principal Components
# the main thing we want to look at is the abs( pca value). The sign tells u s if its a positive or negative correlation.  Since out dataset is segments of the population that form the core customer base for a mail-order sales company in Germany, the following PCs will give insight into how the dataset is best splitup/characterized. The three cells above show the top positive and negative values for each PC. I will go into detail about each value below:
# 
# 
# PC1: 
# =======================================================
# [Positive correlation with bigger familes, negative correlation with small familes. This one is more tied to size of your community/family.]
# 
# 
# ===POSITIVE===
# 
# PLZ8_ANTG3            0.2267        Number of 6-10 family houses in the PLZ8 region
# PLZ8_ANTG4            0.2198        Number of 10+ family houses in the PLZ8 region
# wealth                0.2076        
# HH_EINKOMMEN_SCORE    0.2043        
# ORTSGR_KLS9           0.1974        Size of community
# 
# ===NEGATIVE===
# 
# MOBI_REGIO          -0.2422        
# PLZ8_ANTG1          -0.2264        Number of 1-2 family houses in the PLZ8 region
# KBA05_ANTG1         -0.2257        Number of 1-2 family houses in the microcell
# FINANZ_MINIMALIST   -0.2189        Most descriptive financial type for individual: low financial interest
# KBA05_GBZ           -0.2172        Number of buildings in the microcell
# 
# PC2:
# =========================================================
# [PC2 is more associated with personality type ( simliar to PC3). This principal component is more focused on behaviors ( religious vs 'sensual'), or  vs age vs money saving type.]
# 
# ===POSITIVE===
# 
# 
# ALTERSKATEGORIE_GROB    0.2604        Estimated age based on given name analysis
# FINANZ_VORSORGER        0.2327        finaincial : be prepared
# SEMIO_ERL               0.2310        Personality typology, for each dimension: event-oriented
# SEMIO_LUST              0.1831        Personality typology, for each dimension: sensual-minded
# RETOURTYP_BK_S          0.1644        
# 
# ===NEGATIVE===
# 
# SEMIO_REL       -0.2554        Personality typology, for each dimension:religious
# decades         -0.2514        
# FINANZ_SPARER   -0.2360       Money saver
# SEMIO_TRADV     -0.2312       Personality typology, for each dimension: tradional-minded
# SEMIO_PFLICHT   -0.2290       Personality typology, for each dimension:dutiful
# 
# PC3:
# ==========================================================
# [PC3 has more to do with personality type, having a positive correlation with more positive energies, and negative association with negative / dominate energies. This PC also includes how you save money, having a positive correlation with 'low financial interest' and negative association.]
# ===POSITIVE===
# 
# SEMIO_VERT           0.3463        Personality typology, for each dimension:dreamful
# SEMIO_SOZ            0.2631        Personality typology, for each dimension:socially-minded
# SEMIO_FAM            0.2501        Personality typology, for each dimension:family-minded
# SEMIO_KULT           0.2341        Personality typology, for each dimension:cultural-minded
# FINANZ_MINIMALIST    0.1563        Most descriptive financial type for individual: low financial interest
# Name: Dimension 3, dtype: float64
# ===NEGATIVE===
# 
# ANREDE_KZ    -0.3688        
# SEMIO_KAEM   -0.3373        Personality typology, for each dimension:combative attitude
# SEMIO_DOM    -0.3141        Personality typology, for each dimension:dominant-minded
# SEMIO_KRIT   -0.2742        Personality typology, for each dimension:critical-minded
# SEMIO_RAT    -0.2171        Personality typology, for each dimension:rational
# Name: Dimension 3, dtype: float64
# 

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[97]:


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score

def fit_mods():
    scores = []
    centers = list(range(1,11))

    for center in centers:
        scores.append(get_kmeans_score(data, center))

    return centers, scores
def plot_data(data, labels):
    '''
    Plot data with colors associated with labels
    '''
    fig = plt.figure();
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10');


# In[108]:



scores = []
# centers = list(range(1,3))
centers = [1,5,7,8,9,12]

for center in centers:
    scores.append(get_kmeans_score(pca_azdias, center))
    
plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');


# In[110]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.

pred_8 = KMeans(8).fit_predict(pca_azdias)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# saw a big fall off in rate of change at 5, then again another change at around 8.  From 8-12 there wasn't a dramatic change in  sum of the squared distance between centroid and each member of the cluster. Therefore, I think 8 is an ideal number

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[128]:


# Load in the customer demographics data.
customers = pd.read_csv("Udacity_CUSTOMERS_Subset.csv", sep = ";")
print(customers.shape)
customers.head()


# In[133]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.
clean_customers_data = clean(customers)


# In[134]:


clean_customers_data.head()


# In[135]:


customer_transformations = PCA(20).fit_transform(StandardScaler().transform(imp.transform(clean_customers_data)))
# print("After Feature Scaling: {}".format(customer_transformations.shape))


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[ ]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


# In[ ]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




