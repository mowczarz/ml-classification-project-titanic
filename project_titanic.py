#!/usr/bin/env python
# coding: utf-8

# # 🛳️ Which passengers survived the Titanic shipwreck? 🛳️
# **Machine Learning Tabular Data Binary Classification**
# 
# ---
# 
# Ahoy!
# 
# How many times have you wondered if you would survive the Titanic crash? Alright, maybe not to much. I think you'd rather know the answer to: why Rose didn't share that goddamned door!? 🚪 💔 (I hope you've seen the movie).
# 
# I can't help you with this question (though perhaps displacement is a clue). But we're going to have fun with machine learning to create a model that predicts which passengers will survive the Titanic shipwreck!
# 
# To do this, we're going to pick up the gauntlet and we take part in the Kaggle [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition. We'll be working on train dataset containing nearly 900 passengers data records and after we create the best model we can afford, we'll send our predictions on test dataset to the Kaggle platform which will evaluate our model.
# 
# Sounds exciting, doesn't it? I hope you enjoy it as much as I did. 🙂
# 
# ---
# 
# The plan for this project is as follows:
# 1. Preparing our workspace
# 2. Getting familiar with data
# 3. Feature Engineering
# 4. Data Preprocessing
# 5. 🤖 Testing different models 🤖
# 6. Improving selected models
# 7. Trying to experiment (a little) and final testing results
# 8. Saving, loading and final training selected models
# 9. Making predictions and submissions to send
# 10. Getting evaluated by Kaggle and summary
# 
# ---
# 
# It looks simple, right? Let's not waste time and get to work!

# ## 1. Preparing our workspace
# 
# Firstly, we need necessary packages with core libraries. As you can see, I commented out this section because in my case I'm sure that I have them installed.

# In[1]:


# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install sklearn
# !pip install seaborn
# !pip install pandas_profiling
# !pip install xgboost
# !pip install lightgbm
# !pip install ipywidgets
# !pip install IPython


# Now we can import libraries which we are going to use in this project. I decided to import all of functions and modules here, so there's a little spoiler here of what we are going to use. 🙂

# In[2]:


# Basic data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data exploration
from pandas_profiling import ProfileReport

# Data preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

# Split data
from sklearn.model_selection import train_test_split

# Supervised Learning Estimators - Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Classification Metrics
from sklearn.model_selection import cross_val_score

# Tune model
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Hide warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

# Experiment
from sklearn.ensemble import VotingClassifier
from itertools import combinations

# Save and load model
import pickle

# Others
import os
from IPython.display import clear_output, Image
from copy import deepcopy


# At the end, we'll check and display versions of our packages - this may be helpful in the future if we want to compare other results at a different time.

# In[3]:


import sys
import matplotlib
import sklearn
import pandas_profiling
import xgboost
import lightgbm

print(f'Python: {sys.version}')
print(f'pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'pandas_profiling: {pandas_profiling.__version__}')
print(f'XGBoost: {xgboost.__version__}')
print(f'LightGBM: {lightgbm.__version__}')


# Alright, so now we can go to the next, more exciting step.

# ## 2. Getting familiar with data
# 
# Firstly, we can look on discription of our data which we could find on [Kaggle website](https://www.kaggle.com/competitions/titanic/data). There we can find the list of variables, their specifications and types.
# 
# However, while reading the data description is important, it is not enough. We have to do deeper data exploration to be sure, that our data is ready for later prediction.
# 
# We'll start with loading data using pandas. I downloaded them earlier from [here](https://www.kaggle.com/competitions/titanic/data) and put them in the `data` folder.

# In[4]:


data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")


# In[5]:


data_train.head()


# In[6]:


data_test.head()


# Ok, it worked. The data has been successfully loaded. 
# 
# As we can see right now, the structre of data is the same for training and test set. The only difference in the features is the variable to estimate `Survived` like it should be.
# 
# At a later stage it will be useful to have a combined database of training and testing sets, so we'll prepare it.

# In[7]:


data_train['train_test'] = 1
data_test['train_test'] = 0
data_test['Survived'] = np.NaN
all_data = pd.concat([data_train,data_test], ignore_index=True)
all_data


# All right, we can move on to the data exploration. There's no one particular way of doing this. But what we should do is to become more familiar with the dataset.

# ### Basic review
# We'll focus primarily on training data, cause we have it labeled. First, a brief description of the data.

# In[8]:


data_train.info()


# In[9]:


data_train.describe()


# Missing values

# In[10]:


data_train.isna().sum()


# In[11]:


data_test.isna().sum()


# Hmm, filling missing values of `Emabarked` shouldn't be problematic. In the next steps of the project we'll consider how to approach `Age` and `Cabin`.
# 
# Ok, now let's take a look at numeric and categorical variables separately.

# In[12]:


data_train_num = data_train[['Age','SibSp','Parch','Fare']]
data_train_cat = data_train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# ### Numeric data
# Histograms

# In[13]:


for column in data_train_num.columns:
    plt.hist(data_train_num[column])
    plt.title(column)
    plt.show()


# Correlation matrix

# In[14]:


print(data_train.drop(["PassengerId", "train_test"], axis=1).corr())
sns.heatmap(data_train.drop(["PassengerId", "train_test"], axis=1).corr());


# Mean values in pivot table

# In[15]:


print(pd.pivot_table(data_train, index="Survived", values=['Age','SibSp','Parch','Fare'], aggfunc=np.mean))


# ### Categorical data
# 
# Histograms

# In[16]:


for col in data_train_cat.columns:
    sns.countplot(hue='Survived', x=col, data=data_train)
    plt.show()


# Number of survived by category (no point doing by `Ticket` and `Cabin`).

# In[17]:


for col in ['Pclass', 'Sex', 'Embarked']:
    print(pd.pivot_table(data_train, values='PassengerId', index='Survived', columns=col, aggfunc='count'))
    print()


# ### Generate reports
# The other way to get basic analysis quickly is using the `pandas_profiling` library.
# 
# To be honest - it includes what we generated above plus lots of other details. We'll generate them for both of our datasets.

# In[18]:


profile_train = ProfileReport(data_train, title='Train Dataset', html = {'style':{'full_width': True}})
profile_train


# In[19]:


profile_test = ProfileReport(data_test, title='Test Dataset', html = {'style':{'full_width': True}})
profile_test


# Problem with displaying reports? Check the `data` folder, reports have been exported there for this case.

# In[20]:


# profile_train.to_file("data/Report-train_dataset.html")
# profile_test.to_file("data/Report-test_dataset.html")


# ### Summary for the next steps
# 
# * Our training dataset is a slightly unbalanced - 62:38 with a prodominance for non-survived. This isn't critical, but in the case of model evaluations, we should be careful.
# 
# * `Sex` appears to be the variable with the greatest direct impact on the target variable.
# 
# * Relatively many of our variables are correlated with each other. However, we'll not worry about it at this stage.
# 
# * We have 3 high cardinality (unique or almost unique values) features: `Name`, `Ticket`, `Cabin`. We'll try some ideas to get something interesting from them:
#     * We'll get a title from `Name` (like "Miss", "Ms.", "Mrs.").
#     * Some tickets from `Ticket` seems to have markings at beginning - we'll check their cardinality. We'll also check if and how often the ticket numbers are repeated.
#     * 687 out of 891 records have no value for `Cabin`, the others have an alphabetic, single-letter designation at beginning - we'll turn this information into a new variable. Also, one cell of record may contain information about more than one cabin. We'll consider whether we use this information.
#     
#     
# * We have to deal with missing values
# 
# * One ticket can be assigned to several people, which gives us information about companions. We will think about whether and how to use this information.
# 
# * It seems that we can combine the features `SibSp` (# of siblings/spouses aboard) and `Parch` (# of parents / children aboard) into one, meaning the size of the family. We'll check it.
# 
# * We'll check the sense of grouping some variables.

# Ok, we have some work ahead of us, so let's not extend it - let's start!

# ## 3. Feature Engineering
# At this stage of the project, we will play a little bit with the data to add / remove / change variables.
# 
# We'll work here on a copy of our data and in the next section "Data preprocessing" we'll introduce the developed changes.

# In[21]:


temp_dataset = all_data.copy()
temp_dataset


# In this section, I have outlined all the ideas I found worth checking out. Eventually, I abandoned some of them at the stage of section *5. Testing different models* of this project. To keep the notebook clear, I have omitted to present the results of such tests, and have only described the results.

# ### `Title`
# 
# We'll get title from passager's name.

# In[22]:


temp_dataset['Title'] = temp_dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
temp_dataset.groupby('Title').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="PassengerId", ascending=False).rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# I think it's a good idea to reduce the number of titles a bit. We'll group them to: "Mr", "Miss", "Mrs", "Master" and "Other" in next section.

# ### `TicketMark`
# 
# In the process of reviewing the data I noticed that some marks are the same, but with different case of letters or with the signs `.` and `/` in various places.
# I deciated to fuze them.

# In[23]:


temp_dataset['TicketMark'] = temp_dataset['Ticket'].apply(lambda x: x[:x.rfind(' ')].replace('.', '').replace('/', '').replace(' ', '').upper() if ' ' in x else "is_null")
temp_dataset.groupby('TicketMark').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="PassengerId", ascending=False).rename(columns={"PassengerId": "count_all", "train_test": "count_train", "Survived": "% survived (train set)"})


# It's not a good idea to create separate categories for each. We group values with a count less than 10 in the training set.

# ### `CabinMark`

# In[24]:


temp_dataset['CabinMark'] = temp_dataset['Cabin'].apply(lambda x: 'is_null' if pd.isna(x) else str(x)[0])
temp_dataset.groupby('CabinMark').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="PassengerId", ascending=False).rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# Although it's a small thing, we'll group the two least numerous categories.

# ### `MultiCabins` (abandoned)
# At first, I thought of a variable `NumberOfCabins`.

# In[25]:


temp_dataset['NumberOfCabins'] = temp_dataset['Cabin'].apply(lambda x: "is_null" if pd.isna(x) else len(x.split(' ')))
temp_dataset.groupby('NumberOfCabins').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="PassengerId", ascending=False).rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# In[26]:


sns.countplot(hue='Survived', x='NumberOfCabins', data=temp_dataset);


# But I found that there is no need to crumble. Next I tried to group them as "is_null" and "is_not_null" into a binary variable `MultiCabins` but that didn't help either.

# ### `FamilySize`

# In[27]:


temp_dataset.groupby('SibSp').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="SibSp").rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# In[28]:


temp_dataset.groupby('Parch').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="Parch").rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# I thought to join the above variables into one, which I quickly tested later.

# In[29]:


temp_dataset['FamilySize'] = temp_dataset['SibSp'] + temp_dataset['Parch'] + 1
temp_dataset.groupby('FamilySize').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="FamilySize").rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# In[30]:


sns.countplot(hue='Survived', x='FamilySize', data=temp_dataset);


# As I tested, this variable proved to be the best when it replaced `SibSp` and `Parch`.

# ### `PeopleOnTicket`
# 
# We can see that one ticket can be assigned to several people. In part, this information may coincide with the information on the number of family members. But not only that - also it can give us information about unmarried couples or groups of friends.
# 
# In addition, you can see that the ticket price given in the variable `Fare` applies to the entire ticket, including the multi-person tickets. 
# 
# It seems to me a good idea to calculate the price of the ticket "per person", which may be a better indication of the "quality" of the purchased cruise.
# 
# **But caution! In this case we extend the information in the training set using information from the test set! We have to be VERY careful with something like this.**
# 
# On consideration, however, I don't think it should hurt at all.

# In[31]:


temp_dataset['PeopleOnTicket'] = temp_dataset.groupby('Ticket')['Ticket'].transform('count')
temp_dataset.groupby('PeopleOnTicket').agg({'PassengerId': 'count', 'train_test': 'sum', 'Survived':'mean'}).sort_values(by="PassengerId", ascending=False).rename(columns={"PassengerId": "count", "train_test": "count_train", "Survived": "% survived"})


# In[32]:


sns.countplot(hue='Survived', x='PeopleOnTicket', data=temp_dataset);


# ### `IsAlone`

# This variable specifies people traveling alone.

# In[33]:


temp_dataset['IsAlone'] = (temp_dataset['FamilySize'] + temp_dataset['PeopleOnTicket']) == 2
sns.countplot(hue='Survived', x='IsAlone', data=temp_dataset);


# In[34]:


temp_dataset['IsAlone'] = temp_dataset['FamilySize'] == 1
sns.countplot(hue='Survived', x='IsAlone', data=temp_dataset);


# We'll find this variable useful.

# ### `FarePerPerson`
# We will create this new variable and look at it briefly.

# In[35]:


temp_dataset['Fare'].fillna(temp_dataset['Fare'].median(), inplace=True) # only one record
temp_dataset['FarePerPerson'] = temp_dataset['Fare'] / temp_dataset['PeopleOnTicket']


# In[36]:


temp_dataset[['Fare', 'FarePerPerson']].describe()


# In[37]:


print("Corelations:", temp_dataset['Fare'].corr(temp_dataset['Survived']), temp_dataset['FarePerPerson'].corr(temp_dataset['Survived']))


# I quickly checked the use of this variable and compared with the situation when we take it instead and with the `Fare` variable. It looks like it's best to use both of them.

# ### Grouping / bucketing ?
# 
# I tried a bit with different combinations of grouping (bucketing) of numerical variables `Age`, `Fare`, `FamilySize`, `FarePerPerson` and `PeopleOnTicket`. Finally - I abandoned this idea. It didn't improve the results of the initial tests.
# 
# Okay, we've been working a bit... Time to move on.

# ## 4. Data Preprocessing
# 
# Alright, now we have a hard nut to crack. We need to make a function that will transform our data to the desired form as we established earlier. Additionally, we'll encode categorical features (which is necessary) and normalize numerical features (which is usually helpful).
# 
# It's good idea to include all of operations to one function - then we will be able to use it again when we need it.
# 
# To sum up, we need to include the following operations:
# 
# 1. Create new features that we defined in the previous section.
# 2. Fill in missing values of `Age`, `Fare` and `Embarked`.
# 3. Encode binary categorical features using label-encoding
# 3. Encode non-binary categorical features as a one-hot numeric array.
# 4. Drop the variables that we deem unnecessary.
# 
# Let's code a little.

# In[38]:


def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    
    # Copy the dataset (it's necessary to preserve the original dataset)
    df = df.copy()   
   
    # Create new features    
    # Create `Title`
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())    
    df['Title'] = df['Title'].apply(lambda x: 'Other' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
    # Create `TicketMark`
    df['TicketMark'] = df['Ticket'].apply(lambda x: x[:x.rfind(' ')].replace('.', '').replace('/', '').replace(' ', '').upper() if ' ' in x else "is_null")
    df['TicketMark'] = df['TicketMark'].apply(lambda x: 'Other' if x not in ['is_null', 'PC', 'CA', 'A5', 'SOTONOQ', 'STONO2', 'SCPARIS', 'WC', 'A4', 'FCC'] else x) 
    # Create 'CabinMark'
    df['CabinMark'] = df['Cabin'].apply(lambda x: 'is_null' if pd.isna(x) else str(x)[0])
    df['CabinMark'] = df['CabinMark'].apply(lambda x: 'is_null' if x in ['G', 'T'] else x)
    # Create `FamilySize`
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Create `PeopleOnTicket
    df['PeopleOnTicket'] = df.groupby('Ticket')['Ticket'].transform('count')
    # Fill missing values - Fare
    df['Fare'] = df.groupby(['Pclass', 'FamilySize'])['Fare'].apply(lambda x: x.fillna(x.median()))
    # Create `FarePerPerson`
    df['FarePerPerson'] = df['Fare'] / df['PeopleOnTicket']
    # Create `IsAlone`
    df['IsAlone'] = (df['FamilySize'] + df['PeopleOnTicket']) == 2 
    
    # Fill missing values - Age
    df['Age'] = df.groupby(['Sex', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))    
    df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
    
    # Age < 1
    df['Age'] = np.where(df['Age'] < 1, 0, df['Age'])
   
    # Transform binary categorical variable features
    labelencoder = LabelEncoder()
    df['Sex'] = labelencoder.fit_transform(df['Sex'])
    df['IsAlone'] = labelencoder.fit_transform(df['IsAlone'])
    
    # Transform non-binary categorical features with OneHotEncoder
    features_to_transform = ['Pclass', 'Embarked', 'Title', 'TicketMark', 'CabinMark']
    transformer = make_column_transformer((OneHotEncoder(sparse=False), features_to_transform))
    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    df = df.join(transformed_df)
    df.drop(features_to_transform , axis=1, inplace=True)
     
    # Drop unnecessary features
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'] , axis=1, inplace=True)
    return df


# Wow, quite pretty function. Let's see if it works.

# In[39]:


preprocessed_data = preprocess_data(all_data)
preprocessed_data.head()


# Ok, everything seems fine. 
# 
# Now we'll prepare a function that we'll use to normalize our data. We will provide a possibility to do this in three ways:
# 
# * Standarize data 
# * Normalize data to [0, 1]
# * Normalize data to [-1, 1]
# 
# We will test every possibility and see which our models will do best.
# 
# Note that according to the normalization rules, we will ignore outliers when computing normalizing values.

# In[40]:


def normalize_data(df, norm=0, skip=[], non_extremum=False):
    """
    Normalizes data and returns a normalized set
    df: dataset
    norm: type of normalization (0: standarize, 1: normalize to [0, 1], 2: normalize to [-1, 1]
    skip: list of columns to be omitted
    non_extremum: normalizing values are calculated disregarding extreme values
    """  
    
    # Copy the dataset
    df = df.copy()          
    df_no_extremum = df.copy()
            
    # Columns to transoform
    columns = [column for column in df.columns if column not in skip]
    
    if non_extremum:
        if "Fare" in columns:
            df_no_extremum.drop(df_no_extremum[df_no_extremum["Fare"] > 270].index, inplace=True)
        if "FarePerPerson" in columns:    
            df_no_extremum.drop(df_no_extremum[df_no_extremum["FarePerPerson"] > 56].index, inplace=True)
        if "Age" in columns:    
            df_no_extremum.drop(df_no_extremum[df_no_extremum["Age"] > 65].index, inplace=True)     
    
    for column in columns:        
        # Standarization
        if norm == 0:        
            m = df_no_extremum[column].mean()
            s = df_no_extremum[column].std()             
        # Normalization [0, 1]
        if norm == 1:
            m = df_no_extremum[column].min()
            s = df_no_extremum[column].max() - df[column].min()    
        # Normalization [-1, 1]
        if norm == 2:
            m = (df_no_extremum[column].max() + df_no_extremum[column].min()) / 2
            s = (df_no_extremum[column].max() - df_no_extremum[column].min()) / 2            
        df[column] = (df[column] - m) / s
        
    return df


# Now we'll process our data with function we've created. We'll make 3 separate dataset depending on whether and how we normalize the data.

# In[41]:


# Non-normalized data
df = preprocessed_data

# Standarized data 
df_stand = normalize_data(preprocessed_data, norm=0, skip=["PassengerId", "Survived", "train_test"], non_extremum=True)

# Normalized data to [0, 1]
df_norm1 = normalize_data(preprocessed_data, norm=1, skip=["PassengerId", "Survived", "train_test"], non_extremum=True)

# Normalized data to [-1, 1]
df_norm2 = normalize_data(preprocessed_data, norm=2, skip=["PassengerId", "Survived", "train_test"], non_extremum=True)


# And now we'll split them for training and test data again. Note that it's enough to create just one target vector as it can be common to all databases.

# In[42]:


y = df[df["train_test"]==1]["Survived"]

# Non-normalized data
X = df[df["train_test"]==1].drop(["PassengerId", "train_test", "Survived"], axis=1)
X_test = df[df["train_test"]==0].drop(["train_test", "Survived"], axis=1).reset_index(drop=True)

# Standarized data
X_stand = df_stand[df_stand["train_test"]==1].drop(["PassengerId", "train_test", "Survived"], axis=1)
X_test_stand = df_stand[df_stand["train_test"]==0].drop(["train_test", "Survived"], axis=1).reset_index(drop=True)

# Normalized data to [0, 1]
X_norm1 = df_norm1[df_norm1["train_test"]==1].drop(["PassengerId", "train_test", "Survived"], axis=1)
X_test_norm1 = df_norm1[df_norm1["train_test"]==0].drop(["train_test", "Survived"], axis=1).reset_index(drop=True)

# Normalized data to [-1, 1]
X_norm2 = df_norm2[df_norm2["train_test"]==1].drop(["PassengerId", "train_test", "Survived"], axis=1)
X_test_norm2 = df_norm2[df_norm2["train_test"]==0].drop(["train_test", "Survived"], axis=1).reset_index(drop=True)


# So we can go to the best part of our little project - creating real machine lerning models. Let's go!

# ## 5. 🤖 Testing different models 🤖
# 
# Alright! It's been a long road but it was worth it. We begin the most exciting moment - the first fits and model test on our data.
# 
# But first things first. At the beginng we'll list (in dictionary) the models that we are using.
# 
# If you want to know where they are from, you can back to the beginning of the project to section: *1. Prepareing our workspace*.

# In[43]:


models = {"KNeighborsClassifier": KNeighborsClassifier(),
          "LogisticRegression": LogisticRegression(), 
          "GaussianNB": GaussianNB(),
          "SupportVectorClassifier": SVC(probability=True),
          "RandomForestClassifier": RandomForestClassifier(),
          "DecisionTreeClassifier": DecisionTreeClassifier(),
          "XGBoost": XGBClassifier(),
          "LightGBM": LGBMClassifier()}


# And we'll create similar dictionary, but without Logistic Regression - we'll need it in two cases.

# In[44]:


models_no_LR = models.copy()
del models_no_LR["LogisticRegression"]


# We come to the main point of this section of project. We'll evaluate and compare our models in order to choose the best.
# 
# Our evaluation and comparing will be simplified and limited to one metric only - to **accuracy**. The reason is that our final predictions will be evaluate with this metric (it is stated [here](https://www.kaggle.com/competitions/titanic/overview/evaluation)).
# 
# Now we get to the point. At the beginning we'll fit and test our models with their basic form with their default (hiper)parameters. Each model will be tested several times, on differently drawn data split (hence the list of seeds, affecting different data splits). I've decided to test models in two ways: for different seeds with the usual split of data and for different seeds (in smaller amounts) with using corss-validation. In theory, the results should be similiar.
# 
# Important! **An essential element of human work with machine learning models is a well-thought-out approach to data spliting.** Knowing what data we want to predict (test data), we should properly select training and validation data on which we could check our models. However, in our case, the test set provided by Kaggle does not seem to have been extracted in any particular way. We can assume that it was selected randomly, so we can also split our data to the training and validation sets at random.
# 
# Ok, let's back to work and let's write some function. We'll make sure that it's universal enough to use it later for modificated models. In case of big number of seeds, the using of this function may take a long time, so I'm going add some progress marker which will be printed.

# In[45]:


def fit_and_score(models, X, y, seeds_list = []):
    """
    Fits and evaluates given machine learning models by accuracy.
    models : dict of different machine learning models
    X : data with no labels
    y : labels assosciated with data
    seeds_list : list of seeds (optionaly) 
    """
    # Random seeds for reproducible results
    if not seeds_list:
        seeds_list = [np.random.randint(0,1000) for _ in range(20)]       
    # Make a dict to keep model scores
    model_scores = {}
    # Progress string
    progress = ""
    # Loop through models  
    for name, model in models.items():
        # Make lists to keep model scores depending on seed
        scores = []
        cv_scores = []
        # Variables to print progress
        count = 0
        num_seeds = len(seeds_list)
        progress += name + ": " 
        # Loop through seeds
        for seed in seeds_list:
            # Setup random seed
            np.random.seed(seed)
            # Split the data, fit the model and add it's score to the list
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))  
            # Get cross-validation score and add it to list
            if count % 5 == 0:
                cv_scores += list(cross_val_score(model, X, y, cv=5, n_jobs=-1))
            # Print progress
            count += 1
            percent = "{:.0%}".format(count / num_seeds)
            clear_output(wait=True)
            print(progress + str(percent))         
        progress += str(percent) + "\n"
        # Evaluate the model by its mean of its scores and add it to model_scores dict
        model_scores[name] = (np.mean(scores), np.std(scores), np.mean(cv_scores), np.std(cv_scores))      
    return model_scores


# Now we'll set up list of seeds for which random operations will be performed. By assigning the list with random seeds to a variable, we'll later be able to repeat the fitting of other models under the same conditions.
# 
# (Random seed for generete random seeds... that could looks funny. 🙂 But I see the sense in it.) 

# In[46]:


np.random.seed(22)
number_of_seeds = 20
seeds_list = [np.random.randint(0,1000) for _ in range(number_of_seeds)] 


# Ok, now we'll test all of our models on 4 generated databases. We'll immediately assign the results to the dataframes. 
# 
# As you can see, in the two cases we'll use models without logistic regression because it's not recommended to use it on not normalized data (as it turned out later, also with normalized values up to [-1, 1]). In the name of science, I tried it before anyway. 🙂 And model had problem with its.

# In[47]:


model_scores = pd.DataFrame(fit_and_score(models_no_LR, X, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score", 1: "std", 2: "cv_score", 3: "cv_std"})
model_scores_stand = pd.DataFrame(fit_and_score(models, X_stand, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score_stand", 1: "std_stand", 2: "cv_score_stand", 3: "cv_std_stand"})
model_scores_norm1 = pd.DataFrame(fit_and_score(models, X_norm1, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score_norm1", 1: "std_norm1", 2: "cv_score_norm1", 3: "cv_std_norm1"})
model_scores_norm2 = pd.DataFrame(fit_and_score(models_no_LR, X_norm2, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score_norm2", 1: "std_norm2", 2: "cv_score_norm2", 3: "cv_std_norm2"})


# We'll do some simple operations on our dataframes...

# In[48]:


all_model_scores = model_scores.join(model_scores_stand, how='outer').join(model_scores_norm1, how='outer').join(model_scores_norm2, how='outer')
all_model_scores["max_score"] = all_model_scores[["score", "score_stand", "score_norm1", "score_norm2"]].max(axis=1)
all_model_scores["cv_max_score"] = all_model_scores[["cv_score", "cv_score_stand", "cv_score_norm1", "cv_score_norm2"]].max(axis=1)
pd.options.display.float_format = '{:.2%}'.format


# ... and here they are:

# In[49]:


all_model_scores.sort_values(by="max_score", ascending=False, inplace=True)
all_model_scores[["score", "std", "score_stand", "std_stand", "score_norm1", "std_norm1", "score_norm2", "std_norm2", "max_score"]]


# In[50]:


all_model_scores.sort_values(by="cv_max_score", ascending=False, inplace=True)
all_model_scores[["cv_score", "cv_std", "cv_score_stand", "cv_std_stand", "cv_score_norm1", "cv_std_norm1", "cv_score_norm2", "cv_std_norm2", "cv_max_score"]]


# Yeah, finally! After a quick look at the results, it's clear that `GaussianNB` model is failing. Considering we have several prosperus models (and limited computing power 😞 ) I think what we'll leave `DecisionTreeClassifier` model too.
# 
# Although in some cases these are not big differences, I think from now on we will only focus on one type of data. In the next part we'll process the data normalized to the value [0, 1].

# ### First results
# 
# Ok, so let's make a more clear dataframe. I decided to create the scoreboard using the mean of the two model testing methods above.

# In[51]:


selected_model_score = pd.DataFrame(all_model_scores[["score_norm1", "cv_score_norm1"]].mean(axis=1), columns=['score'])
selected_model_score.sort_values(by='score', ascending=False, inplace=True)
selected_model_score.drop(selected_model_score.tail(2).index,inplace=True)
selected_model_score


# I think that these scores above look not bad. But, of course, we won't stop here. Now we're going to try to improve the results of the models.

# ## 6. Improving selected models

# Each of our models has some certain parameters that we can change, which will affect the effectiveness of the model. To distinguish them from immutable parameters, we call them **hyperparameters**.
# 
# There may be many such hypeparameters for one model. This leads to the fact, there may be hundreds or even thousends different combinations to test.
# 
# Depending on the available computing power, we can try different possibilities. In our case we'll do it in two different ways:
# 1. With using `GridSearchCV` - testing all combinations for a few selected hyperparameters
# 2. With using `RandomizedSearchCV` - testing randomly selected combinations for more hyperparameters
# 
# Each model has its own hyperparameters. How do we know which ones and how to tune them? The answer isn't simple. Certainly, a good first step is to check the model documentation. The next good step might be to find some tuning examples on the internet. However, it's worth paying attention to whether its use in such cases is similar to our problem. Next steps - just experiment. 🙂
# 
# Earlier, when we were testing our models, we created a function by which the models were tested on many different data splits. Now, when using `GridSearchCV` and `RandomizedSearchCV` we'll use the `RepeatedKFold` to make things easier for us.

# In[52]:


rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=22)


# The parameters set above mean that the model will be tested four times (`n_repeats=4`) with using cross-validation divided into 5 folds (`n_splits=5`).
# 
# Ok, finally, let's move on!

# ### GridSearchCV
# 
# 

# We'll create a hyperparameter grid - a dictionary with hyperparameters to test - for each of our tested model. 
# 
# To setup this below, I used the information from the model documentation, traced various similar problems on the internet, and just used my previous experience.

# In[53]:


C = [0.5, 1, 1.5, 2, 4, 6, 8, 10]

# Support Vector Classifier
svc_grid = [
    {'C': C, 
     'gamma': ['scale', 'auto', 0.1, 0.05, 0.01, 0.005, 0.001],
     'kernel': ['rbf']},
    {'C': C,
     'kernel': ['poly']}]

# Logistic Regression
max_iter = [50, 100, 250, 500, 750, 1000, 2000]
lr_grid = [
    {'penalty': ['l2'],
     'C': C,
     'solver': ['lbfgs'],
     'max_iter': max_iter},
    {'penalty': ['l1', 'l2'],
     'C': C,
     'solver': ['liblinear'],
     'max_iter': max_iter},
    {'penalty': ['elasticnet'],
     'C': C,
     'solver': ['saga'],
     'max_iter': [1000, 2000, 3000, 4000],
     'l1_ratio':[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]}
     ]

# K Neighbors Classifier
knc_grid = {'n_neighbors' : [3, 5, 7, 10, 15, 20, 25, 31, 40, 50],
            'leaf_size': [5, 10, 15, 20, 30, 45, 60, 75],            
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree','kd_tree'],
            'metric' : ['minkowski', 'chebyshev', 'euclidean', 'manhattan'],
            'p' : [1, 2]}

# LightGBM
lgbm_grid = {'num_leaves': [5, 10, 15, 31, 45, 60],
             'learning_rate':  [0.5, 0.25, 0.1, 0.075, 0.05, 0.01],
             'n_estimators': [30, 50, 100, 200],
             'max_depth': [-1, 4, 5, 6, 7, 8, 10]}

# XGBoost 
xgb_grid = {'min_child_weight': [0.5, 1, 1.5],
            'gamma': [0, 0.1, 0.01],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.5, 1.0],
            'max_depth': [None, 6],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 0.25, 0.1, 0.075, 0.05]}

# Random Forest Classifier
rfc_grid = {'criterion': ['gini','entropy'],
            'max_depth': [4, 6, 10, None],
            'max_features': [None, 5, 10],
            'min_samples_leaf': [.1, 2, 3],
            'min_samples_split': [2, 3, 6],
            'n_estimators': [100, 200, 300]}


# Next we'll create instances of `GridSearchCV` for each model.

# In[54]:


grid_search_models = {}

# Support Vector Classifier
grid_search_models["SupportVectorClassifier"] = GridSearchCV(SVC(probability=True),
                                                             param_grid=svc_grid,
                                                             cv=rkf,
                                                             n_jobs=-1,
                                                             verbose=1)

# Logistic Regression
grid_search_models["LogisticRegression"] = GridSearchCV(LogisticRegression(),
                                                        param_grid=lr_grid,
                                                        cv=rkf,
                                                        n_jobs=1,
                                                        verbose=1)

# K Neighbors Classifier
grid_search_models["KNeighborsClassifier"] = GridSearchCV(KNeighborsClassifier(),
                                                          param_grid=knc_grid,
                                                          cv=rkf,
                                                          n_jobs=-1,
                                                          verbose=1)

# LightGBM
grid_search_models["LightGBM"] = GridSearchCV(LGBMClassifier(),
                                              param_grid=lgbm_grid,
                                              cv=rkf,
                                              n_jobs=-1,
                                              verbose=1)

# XGBoost 
grid_search_models["XGBoost"] = GridSearchCV(XGBClassifier(),
                                             param_grid=xgb_grid,
                                             cv=rkf,
                                             n_jobs=-1,
                                             verbose=1)

# Random Forest Classifier
grid_search_models["RandomForestClassifier"] = GridSearchCV(RandomForestClassifier(),
                                                            param_grid=rfc_grid,
                                                            cv=rkf,
                                                            n_jobs=-1,
                                                            verbose=1)


# Now we can start fitting our combinations of models.
# 
# Models may have difficulties for some combinations of parameters. We don't mind that some of the possibilities will be missed. However, we'll set the warning to be suppressed.

# In[55]:


for name, model in grid_search_models.items():    
    print(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_norm1, y)


# We get the best combination of parameters for each model.

# In[56]:


for name, model in grid_search_models.items():
    print(name)
    print(model.best_params_)


# Ok, now we'll setup dictionaries with models with the best combinations of parameters found.

# In[57]:


grid_search_best_estimators = {}
for name, model in grid_search_models.items():
    grid_search_best_estimators[name] = model.best_estimator_

We'll make a similar check as before.
# In[58]:


grid_model_scores = pd.DataFrame(fit_and_score(grid_search_best_estimators, X_norm1, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score", 1: "std", 2: "cv_score", 3: "cv_std"})


# And here they are:

# In[59]:


grid_model_scores["mean_score"] = grid_model_scores[["score", "cv_score"]].mean(axis=1)
grid_model_scores.sort_values(by="mean_score", inplace=True, ascending=False)
grid_model_scores


# We'll compare our previous scores with current.

# In[60]:


selected_model_score = selected_model_score.join(grid_model_scores['mean_score']).rename(columns={'mean_score': 'grid_score'})
selected_model_score['grid_improve'] = selected_model_score['grid_score'] - selected_model_score['score']
selected_model_score.sort_values(by='grid_score', ascending=False, inplace=True)
selected_model_score


# We've improved our scores so it was worth to overheat the processor. 🙂
# 
# We tried out the combinations that seemed to be the best. Now we'll try to find better parameters from a larger pool at random.

# ### RandomizedSearchCV

# Similarly to how we did it with `GridSearchCV`, we'll create hyperparameter grids.

# In[61]:


C = np.logspace(-1, 1, 20)

# Support Vector Classifier
svc_random = [
    {'C': C, 
     'gamma': ['scale', 'auto']+list(np.logspace(-4, 0, 20)),
     'kernel': ['rbf']},
    {'C': C,
     'kernel': ['poly']}]


# Logistic Regression
max_iter = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 4000]

lr_random = [
    {'penalty': ['l2'],
     'C': C,
     'solver': ['lbfgs', 'newton-cg'],
     'max_iter': max_iter},
    {'penalty': ['l1', 'l2'],
     'C': C,
     'solver': ['liblinear'],
     'max_iter': max_iter},
    {'penalty': ['elasticnet'],
     'C': C,
     'solver': ['saga'],
     'max_iter': [1000, 1500, 2000, 2500, 3000, 3500, 4000],
     'l1_ratio': np.arange(1,10,1)/10}
]

# K Neighbors Classifier
knc_random = {'n_neighbors' : np.arange(1,40,2),
              'leaf_size': np.arange(5,51,2),            
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'metric' : ['minkowski', 'chebyshev', 'euclidean', 'manhattan'],
              'p' : [1, 2]}

# LightGBM
lgbm_random = {'num_leaves': np.arange(10,101,5),
               'learning_rate':  np.logspace(-4,0,20),
               'n_estimators': [30, 50, 100, 200, 350, 500, 750],
               'max_depth': [-1, 3, 6, 10, 15]}

# XGBoost 
xgb_random = {'min_child_weight': np.logspace(-3,2,10),
              'gamma': np.logspace(-3,1,10),
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
              'colsample_bytree': [0.2, 0.35, 0.5, 0.65, 0.8, 1],
              'max_depth': [None, 3, 6, 10, 15],
              'n_estimators': [20, 50, 100, 200, 350, 500, 750],
              'learning_rate': np.logspace(-4,0,20)}

# Random Forest Classifier
rfc_random = {'bootstrap': [True, False],
              'criterion': ['gini','entropy'],
              'max_depth': [None, 3, 6, 10, 15],
              'max_features': [None, 5, 10],
              'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15],
              'min_samples_split': [2, 4, 6, 8, 10, 15],
              'n_estimators': [50, 100, 200, 350, 500, 750]}


# We'll create instance of `RandomizedSearchCV` for each model.

# In[62]:


random_search_models = {}

# Support Vector Classifier
random_search_models["SupportVectorClassifier"] = RandomizedSearchCV(SVC(probability=True),
                                                                   param_distributions=svc_random,
                                                                   n_iter=300,
                                                                   cv=rkf,
                                                                   n_jobs=-1,
                                                                   verbose=1)
# Logistic Regression
random_search_models["LogisticRegression"] = RandomizedSearchCV(LogisticRegression(),
                                                                param_distributions=lr_random,
                                                                n_iter=300,
                                                                cv=rkf,
                                                                n_jobs=1,
                                                                verbose=1)


# K Neighbors Classifier
random_search_models["KNeighborsClassifier"] = RandomizedSearchCV(KNeighborsClassifier(),
                                                                  param_distributions=knc_random,
                                                                  n_iter=5000,
                                                                  cv=rkf,
                                                                  n_jobs=-1)


# LightGBM
random_search_models["LightGBM"] = RandomizedSearchCV(LGBMClassifier(),
                                                      param_distributions=lgbm_random,
                                                      n_iter=1500,
                                                      cv=rkf,
                                                      n_jobs=-1,
                                                      verbose=1)

# XGBoost 
random_search_models["XGBoost"] = RandomizedSearchCV(XGBClassifier(),
                                                     param_distributions=xgb_random,
                                                     n_iter=500,
                                                     cv=rkf,
                                                     n_jobs=-1,
                                                     verbose=1)

# Random Forest Classifier
random_search_models["RandomForestClassifier"] = RandomizedSearchCV(RandomForestClassifier(),
                                                                    param_distributions=rfc_random,
                                                                    n_iter=500,
                                                                    cv=rkf,
                                                                    n_jobs=-1,
                                                                    verbose=1)


# And we start fitting our combinations of models.

# In[63]:


for name, model in random_search_models.items():    
    print(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_norm1, y)


# We get the best found combination of parameters for each model.

# In[64]:


for name, model in random_search_models.items():
    print(name)
    print(model.best_params_)


# We'll setup dictionaries with models with the best combinations of parameters found.

# In[65]:


random_search_best_estimators = {}
for name, model in random_search_models.items():
    random_search_best_estimators[name] = model.best_estimator_


# Same as we did before:

# In[66]:


random_model_scores = pd.DataFrame(fit_and_score(random_search_best_estimators, X_norm1, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score", 1: "std", 2: "cv_score", 3: "cv_std"})


# And we have:

# In[67]:


random_model_scores["mean_score"] = random_model_scores[["score", "cv_score"]].mean(axis=1)
random_model_scores.sort_values(by="mean_score", inplace=True, ascending=False)
random_model_scores


# Finally we can compare with previous scores.

# In[68]:


selected_model_score = selected_model_score.join(random_model_scores['mean_score']).rename(columns={'mean_score': 'random_score'})
selected_model_score['random_improve'] = selected_model_score['random_score'] - selected_model_score['score']
selected_model_score['max_score'] = selected_model_score[['grid_score', 'random_score']].max(axis=1)
selected_model_score.sort_values(by='max_score', ascending=False, inplace=True)
selected_model_score


# Alright, we've undoubtedly made some improvements. Let’s move on to the summary of this part of project.

# ### Summary
# 
# The current scoreboard looks like this:

# In[69]:


best_model_scores = pd.DataFrame(selected_model_score['max_score']).rename(columns={'max_score': 'score'})
best_model_scores


# In my opinion, considering our problem with which we are facing and the amount of data, these results look pretty good.
# 
# But… we want to try something more 🙂
# 
# Let's create new dictionaries with our favorites…

# In[70]:


best_models = {"XGBoost": random_search_models["XGBoost"].best_estimator_,
               "LightGBM": grid_search_models["LightGBM"].best_estimator_,
               "RandomForestClassifier": random_search_models["RandomForestClassifier"].best_estimator_,
               "SupportVectorClassifier": random_search_models["SupportVectorClassifier"].best_estimator_,
               "LogisticRegression": grid_search_models["LogisticRegression"].best_estimator_,
               "KNeighborsClassifier": grid_search_models["KNeighborsClassifier"].best_estimator_}


# … and let’s try some experiments!

# ## 7. Trying to experiment (a little) and final testing results
# 
# Of course, the possibilities of experimenting with models are limited only by our imagination. But it is necessary to ask yourself the question - how much time is worth spending on this?
# 
# In my humble opinion, the data that we have is not sufficient to achieve an effectiveness close to 1.0. In a real situation, we could try to get more data - both more records and more variables. I think it could help us to improve the quality of our models the most.
# 
# We could also spend more time with Feature Engineering. We should examine the significance of the variables and test their various combinations (in our project, we left it to the models).
# 
# But we'll try something different here. 🙂 We’ll use one of possibilities and we’ll play with combing models using `VotingClassifier`.
# 
# At the beginning, we need to create a list that will store our best models in the form of tuples (name, model).

# In[71]:


best_models_tuples = []
for name, model in best_models.items():
    best_models_tuples.append((name, model))


# Now we'll create a function, which will test all of possible combinations of our models and which will write them into the list with their scores.

# In[72]:


combination_scores_hard = []
combination_scores_soft = []

len_combination = 2**len(best_models_tuples)-len(best_models_tuples)-1
count = 0

for L in range(2, len(best_models_tuples)+1):
    for combination in combinations(best_models_tuples, L):
        # Hard Voting
        voting = VotingClassifier(estimators = combination, 
                                  voting = 'hard')
        combination_scores_hard.append((combination, cross_val_score(voting, X_norm1, y, cv=rkf).mean()))
        # Soft Voting
        voting = VotingClassifier(estimators = combination, 
                                  voting = 'soft')
        combination_scores_soft.append((combination, cross_val_score(voting, X_norm1, y, cv=rkf).mean()))
        # Print progress
        count += 1
        percent = "{:.0%}".format(count / len_combination)
        clear_output(wait=True)
        print("Progess: " + str(percent))


# Now we'll just sort this list by score and then we will get the best combinations of models.

# In[73]:


combination_scores_hard.sort(reverse=True, key=lambda x: x[1])
combination_scores_soft.sort(reverse=True, key=lambda x: x[1])

print("*** Hard Voting ***")
for n in range(0, 3):
    print(f"{n+1}. combination of models:")
    for combination_score in combination_scores_hard[n][0]:
        print ("   -", combination_score[0])
    print("   Score: " + "{:.2%}".format(combination_scores_hard[n][1]))
    
print("\n*** Soft Voting ***")
for n in range(0, 3):
    print(f"{n+1}. combination of models:")
    for combination_score in combination_scores_soft[n][0]:
        print ("   -", combination_score[0])
    print("   Score: " + "{:.2%}".format(combination_scores_soft[n][1]))


# Ok, now we'll setup models from the obtained combinations and we'll add them to our dictionary `best_models`. We'll choose two from each voting method to get a round number of 10 models tested. 🙂

# In[74]:


# Hard Voting
estimators = list(combination_scores_hard[0][0])
model_voting = VotingClassifier(estimators = estimators,
                                voting = 'hard')
best_models['VotingHard_xgb_rfc_svc_lr_knc'] = model_voting

estimators = list(combination_scores_hard[1][0])
model_voting = VotingClassifier(estimators = estimators,
                                voting = 'hard')
best_models['VotingHard_lgbm_rfc_svc_lr_knc'] = model_voting


# Soft Voting
estimators = list(combination_scores_soft[0][0])
model_voting = VotingClassifier(estimators = estimators,
                                voting = 'soft')
best_models['VotingSoft_xgb_lgbm_rfc_svc_lr'] = model_voting

estimators = list(combination_scores_soft[1][0])
model_voting = VotingClassifier(estimators = estimators,
                                voting = 'soft')
best_models['VotingSoft_rf_svc'] = model_voting


# ### Final results
# Now it's time to final comparing of our models.
# 
# Are you ready? I hope so. 🙂

# In[75]:


best_models_scores = pd.DataFrame(fit_and_score(best_models, X_norm1, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score", 1: "std", 2: "cv_score", 3: "cv_std"})


# In[76]:


best_models_scores["mean_score"] = best_models_scores[["score", "cv_score"]].mean(axis=1)
best_models_scores.sort_values(by="mean_score", inplace=True, ascending=False)
best_models_scores


# **Ok, and this is it.** 
# 
# A long way behind us. In my opinion these scores look promising. Of course, there is always something we could do more, which we will write more about in the last part of this project. But I think this is a good time to (at least for the time being) end our fun with testing and improving models. 
# 
# Now we'll proceed to train our models on the whole train data. Then we'll save and load our models to create the possibility to use them later without repeating the entire process.

# ## 8. Saving, loading and final training selected models
# 
# Having selected models with adjusted hyperparameters, we'll save them to files. Thanks to this, we will be able to use then in the future without having to repeat all the earlier, time-consuming calculations.
# 
# First, however, we will train our models one last time, but this time on the entire training data.

# In[77]:


for name, model in best_models.items():
    model.fit(X_norm1, y)


# Now we can save (dump) them to files.

# In[78]:


for name, model in best_models.items():
    pickle.dump(model, open('models/' + name + '.pkl', "wb"))


# And we'll load them.

# In[79]:


loaded_models = {}
for filename in os.listdir("models"):
    name = filename.split('.')[0]
    model = pickle.load(open('models/' + filename, 'rb'))
    loaded_models[name] = model


# Let's check if everything is fine.

# In[80]:


loaded_models_score = pd.DataFrame(fit_and_score(loaded_models, X_norm1, y, seeds_list = seeds_list)).transpose().rename(columns={0: "score", 1: "std", 2: "cv_score", 3: "cv_std"})


# In[81]:


loaded_models_score["mean_score"] = loaded_models_score[["score", "cv_score"]].mean(axis=1)
loaded_models_score.sort_values(by="mean_score", inplace=True, ascending=False)
loaded_models_score


# Okay, everything looks fine, so let's redo the models fit to the full data.

# In[82]:


for name, model in loaded_models.items():
    model.fit(X_norm1, y)


# Now we are ready to start making predictions.

# ## 9. Making predictions and submissions to send

# **The time has come** - we'll prepere our predicts to send them to Kaggle. I'm really very curious about how we'll go. 🙂
# 
# First, let's take a look at our test set and see if everything is ok.

# In[83]:


pd.reset_option('display.float_format')
X_test_norm1.head()


# Now we'll make seperated predictions fo each of our selected models and at one go we'll export them to `.csv` files according to the pattern expected by Kaggle.

# In[85]:


for name, model in loaded_models.items():
    model.fit(X_norm1, y)
    predictions = model.predict(X_test_norm1.drop("PassengerId", axis=1)).astype(int)
    submission = pd.DataFrame(data={'PassengerId': X_test_norm1["PassengerId"], 'Survived': predictions})
    submission.to_csv('submissions/' + name + '.csv', index=False)


# Alright, we did it! Now I'm sending our predictions to Kaggle and I'll show results in the next section. Are you ready? 🙂

# ## 10. Getting evaluated by Kaggle and summary

# Without more ado, let's check how Kaggle has evaluated us.

# ### Kaggle's evaluation

# ![kaggle_scores](images/kaggle_scores.png)

# Ooookay! We'll write something about that in a moment, but first let's put them in the dataframe and compare them to the results we expected.

# In[86]:


kaggle_scores = {"SupportVectorClassifier": 0.79425,
                "LogisticRegression": 0.77511,
                "KNeighborsClassifier": 0.78947,
                "LightGBM": 0.76315,
                "XGBoost": 0.77033,
                "RandomForestClassifier": 0.77272,
                "VotingHard_xgb_rfc_svc_lr_knc": 0.78468,
                "VotingHard_lgbm_rfc_svc_lr_knc": 0.78229,
                "VotingSoft_xgb_lgbm_rfc_svc_lr": 0.77272,
                "VotingSoft_rf_svc": 0.77272}


# In[88]:


final_scores = loaded_models_score.join(pd.DataFrame.from_dict(kaggle_scores, orient='index', columns=['kaggle_score'])).rename(columns={'mean_score': 'our_test_score'})
final_scores.sort_values(by='kaggle_score', ascending=False, inplace=True)
pd.options.display.float_format = '{:.2%}'.format
final_scores[['our_test_score', 'kaggle_score']]


# ### Summary
# 
# We’ve made it to the end. Is our score good or bad? The answer isn’t simple.
# 
# Firstly - why are our final results worse than those of the our tests? 
# 
# We've probably overfitted models a bit. This is a situation where our models have fitted too much to the training data, which is why they do not do that well on the test.
# 
# Another reason is probably that the structure of the test data is slightly different than that of the training data. We haven't spent much time exploring this possibility.
# 
# I think the key might be the different distribution of the variable `sex`, which is by far the most significant of all the variables. You know, "Women and children first" but age is not that important in this case. 🙂
# 
# Check this out. I did one more submission aside, *'sex_only.csv'*, where I assumed all women survived and all men did not. Just look at the result.

# ![survived_by_sex](images/survived_by_sex.png)

# An impressive result, isn't it? 😉
# 
# I saw in the traning dataset that the vast majority of men didn't survive, which makes them much more predictable. With women the matter is more difficult. Women like women - are more difficult to predict. 🙂
# 
# We have a greater proportion of women in the test dataset than in the training set, which made our models a more difficult task.
# 
# What about the Kaggle leaderboard?

# ![kaggle_leaderboard](images/kaggle_leaderboard.png)

# **We placed 588 out of 13360 place (although the score is also the same for 508-620 places), so we are among the top 3.8% - 4.6%.** 
# 
# But - that cannot be a serious point of reference. For example you can easily find on the Kaggle webiste in Titanic subpage codes like "How to be on the top of leaderboard…” thanks to which you can easly generate a file with a ready target vector that will give you an accuracy equal to 1.0... 🙃
# 
# The problem we were facing was quite unusual. Although the data set was generated specifically for this task, taking into account the number and appearance of variables – it’s hard to expect an accuracy close to 1. Our 79,425% score of our best model sounds quite reasonable.
# 
# I think we can stay with this result. Although, of course, it's a good idea to ask...

# ### What can we do more?
# 
# Several things come to mind:
# * First of all, if we want to maximize the result, we should focus on a specific model and adjust the entire methodology to it. In this project, which is somewhat of a guide to the basics of ML, we've tested the different models in a general way.
# * We should pay more attention to recognizing the structure of the test data. I think that if we had tailored the training data structure to it, we would have obtained better results.
# * If our project concerned a real-time issue - we definitely should take care of extending the dataset as a priority. Working on models is important, but there is a good chance that we could improve our results with more data.
# 
# And that's all for now. Thanks for the journey together through this project. I hope you enjoyed it as much as I did. And feel free to visit my other projects that can be found on my GitHub profile!
# 
# See you next time!
# 
# Marek
