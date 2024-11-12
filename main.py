# -*- coding: utf-8 -*-
# Global variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('ds_salaries.csv')
df.dropna(axis=1, how='any')

#Checks for duplicated data.
df.duplicated()

#Checks # of missing (NaN) values
df.isna().sum()

#Deletes 'Unnamed: 0' Column from original df
dfnew = df.drop(columns=['Unnamed: 0'])
# Columns with String values
cat_col = [col for col in dfnew.columns if dfnew[col].dtype == 'object']
# Columns with Int values
num_col = [col for col in dfnew.columns if dfnew[col].dtype != 'object']

def actual_vs_predicted_salary():
    # Split the data into training and test sets
    encoder = LabelEncoder()
    dfnewCopy = dfnew.copy()
    dfnewCopy['experience_level_encoded'] = encoder.fit_transform(dfnewCopy['experience_level'])
    dfnewCopy['company_size_encoded'] = encoder.fit_transform(dfnewCopy['company_size'])
    X = dfnewCopy[['experience_level_encoded', 'remote_ratio', 'company_size_encoded']]
    y = dfnewCopy['salary_in_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Salary")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.show()

st.header("Proposal 2")
st.markdown("""
**Section:** BM7

**Group Number:** 10

**Group Members:**
  1. Villamil, Prince Jeffrey
  2. Ofrin
  3. Caseria
  4. Julia Agustin
  5. Tagorda

**Goal:**


---


For our group project, we want to predict the possible average data scientist salaries along with experience level for said salaries, and if possible we would like to see if it can give insight into the future job market.  

We believe this is the best data set to use for this because the job market for tech has been in an influx thus having a data set that focuses on years where this has occurred will give better results in our productions.

This data will be very interesting to see because as we aspire to join the job market for
data science/tech jobs, we can have a better grasp for what we will be confrutned with
once we graduate.

Lastly, models to use on the dataset, since we want to predict using historical data, we
want to use models focused on years of experience x year x year salary. Based on what we
have searched, time series models are what we want so for things like the exploratory data
analysis models, we also want to experiment with using a regression analysis model to see
which factors contribute the most when predicting salaries based on work_year,
experience_level, job_title, and etc

# Prince Jeffrey Villamil
""")

actual_vs_predicted_salary()
