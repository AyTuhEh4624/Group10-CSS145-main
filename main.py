# -*- coding: utf-8 -*-
# Global variable
import streamlit as st
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
    st.pyplot()

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

st.markdown("""
**Supervised Learning For Salary Prediction**

From this scatter graph, we are able to understand that the predicted salary is considerably lower than the actual salary. Using an encoder to convert "experience_level" and "company_size" into usable values, we are able to get predicted salaries. This plot visualized how linear regression model predictions align with the aculary salary given from the data set. In this case, the predicted values are significantly lower.
""")

actual_vs_predicted_salary()
def distribution_of_remote_work_ratio_and_average_salary_in_USD():
    encoder = LabelEncoder()
    dfnewCopy = dfnew.copy()
    remote_ratio_counts = dfnewCopy['remote_ratio'].value_counts()
    custom_labels = {
        0: "Less than 20%",
        50: "Partially Remote (50%)",
        100: "Fully Remote (More than 80%)"
    }
    labels = [custom_labels[val] for val in remote_ratio_counts.index]
    plt.pie(remote_ratio_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Remote Work Ratio')
    plt.show()

    avg_salary_by_size = dfnewCopy.groupby('company_size')['salary_in_usd'].mean()
    avg_salary_by_size.plot(kind='bar', color='skyblue')
    plt.title("Average Salary by Company Size")
    plt.xlabel("Company Size")
    plt.ylabel("Average Salary in USD")
    plt.show()

def important_factors_in_salary_prediction():
    encoder = LabelEncoder()
    dfnewCopy = dfnew.copy()
    dfnewCopy['experience_level_encoded'] = encoder.fit_transform(dfnewCopy['experience_level'])
    dfnewCopy['company_size_encoded'] = encoder.fit_transform(dfnewCopy['company_size'])
    remote_ratio_mapping = {0: "Less than 20%", 50: "Partially Remote (50%)", 100: "Fully Remote (More than 80%)"}
    dfnewCopy['remote_ratio_str'] = dfnewCopy['remote_ratio'].map(remote_ratio_mapping)
    X = dfnewCopy[['experience_level_encoded', 'remote_ratio', 'company_size_encoded']]
    y = dfnewCopy['salary_in_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    feature_importance = pd.Series(model.coef_, index=['Experience Level', 'Remote Ratio', 'Company Size'])
    feature_importance.plot(kind='bar', color='lightgreen')
    plt.title("Important Factors in Salary Prediction")
    plt.xlabel("Factors")
    plt.ylabel("Predicted Salary Changes")
    plt.show()

distribution_of_remote_work_ratio_and_average_salary_in_USD()
important_factors_in_salary_prediction()
st.markdown("""
**Supervised Learning: Important Factors in Salary Prediction**

This graph highlights the factors that most significantly contribute to predicting the salary.


1. **Experience Level:** This factor has a strong influence on salary, with more senior roles usually demending higher salaries. This was very much expected.

2. **Company Size:** It seems like larger companies will most likely offer higher salaries compared to smaller companies.

3. **Remote Ratio:** The remote work percentage may affect salary; however, from our data, it seems like the average salary given did not see much changes.

# Ofrin
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")

st.markdown("""
""")
