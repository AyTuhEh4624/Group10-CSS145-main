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

st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

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
    st.pyplot()

    avg_salary_by_size = dfnewCopy.groupby('company_size')['salary_in_usd'].mean()
    avg_salary_by_size.plot(kind='bar', color='skyblue')
    plt.title("Average Salary by Company Size")
    plt.xlabel("Company Size")
    plt.ylabel("Average Salary in USD")
    st.pyplot()

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
    st.pyplot()

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

def average_data_science_salaries_by_experience_level_actual_vs_predicted():
    experience_mapping = {
        0: 'Entry Level',
        1: 'Mid Level',
        2: 'Senior Level',
        3: 'Expert Level'
    }
    dfnew['experience_level_code'] = dfnew['experience_level'].astype('category').cat.codes
    X = dfnew[['experience_level_code']]
    y = dfnew['salary_in_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X)
    dfnew['predicted_salary'] = predictions
    avg_predicted_salary = dfnew.groupby('experience_level_code')['predicted_salary'].mean().reset_index()
    avg_actual_salary = dfnew.groupby('experience_level_code')['salary_in_usd'].mean().reset_index()
    avg_actual_salary['experience_level'] = avg_actual_salary['experience_level_code'].map(experience_mapping)
    avg_predicted_salary['experience_level'] = avg_predicted_salary['experience_level_code'].map(experience_mapping)
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(experience_mapping))
    plt.bar(index - bar_width/2, avg_actual_salary['salary_in_usd'], bar_width, label='Actual', color='b')
    plt.bar(index + bar_width/2, avg_predicted_salary['predicted_salary'], bar_width, label='Predicted', color='g')
    plt.title('Average Data Science Salaries by Experience Level (Actual vs Supervised Learning Predicted)')
    plt.xlabel('Experience Level')
    plt.ylabel('Average Salary in USD')
    plt.xticks(index, experience_mapping.values(), rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot()

average_data_science_salaries_by_experience_level_actual_vs_predicted()

st.markdown("""
From this bar graph, we can see that for entry-level positions, the predicted values are in good agreement with the actual ones. For mid-level, the actual salaries are much higher than the predicted ones. Thus, it underestimates salaries for this category. For senior-level positions, the difference is less significant, as it underestimates. Lastly, the model's predictions for expert-level positions are in good agreement with the actual ones.
""")

def average_data_science_salaries_by_employement_type():
    avg_salary = dfnew.groupby('employment_type')['salary_in_usd'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.bar(avg_salary['employment_type'], avg_salary['salary_in_usd'], color='purple')
    plt.title('Average Data Science Salaries by Employment Type')
    plt.xlabel('Employment Type')
    plt.ylabel('Average Salary in USD')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()
    
average_data_science_salaries_by_employement_type()

st.markdown("""
From what we can see on the bar chart, the highest average salary was for CT positions, with an average that far surpasses the rest, and is nearly 175,000. The average for FT employment is relatively middle-of-the-pack at around 100,000, and FL and PT averages are significantly smaller, and their averages run below 50,000. This would indicate that more pay in data science is often associated with contract-based employment, perhaps reflecting the premium put for short-term, high-skilled work, where freelancer and part-time will take lower pay rates.

# Caseria
""")
def data_science_salaries_by_job_title():
    encoder = LabelEncoder()
    dfnewCopy = dfnew.copy()
    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['salary_in_usd'], dfnew['job_title'], 'o', color='m')
    plt.title('Data Science Salaries by Job Title')
    plt.xlabel('Salary in USD')
    plt.ylabel('Job Title')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()
data_science_salaries_by_job_title()

st.markdown("""
Based on the Data Science Salaries by Job Title chart, the highest salaries are for job titles like "Lead Machine Learning Engineer" and "Head of Data Science," with some reaching over 300,000. Positions like "Data Scientist" and "Data Analyst" are more common and tend to have lower salaries, usually below 150,000.

Moreover, this suggests that the highest-paying jobs in data science are usually specialized or leadership roles, which require advanced skills or experience. In contrast, general roles like data analysts or entry-level data scientists have lower average pay, showing that salary in data science often grows with job seniority and specialization.

# Julia Agustin
""")

def data_science_salaries_by_remote_ratio():
    encoder = LabelEncoder()
    dfnewCopy = dfnew.copy()
    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['salary_in_usd'], dfnew['remote_ratio'], 'o', color='orange')
    plt.title('Data Science Salaries by Remote Ratio')
    plt.xlabel('Salary in USD')
    plt.ylabel('Remote Ratio')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()
data_science_salaries_by_remote_ratio()

st.markdown("""
From what we can see on the Data Science Salaries by Remote Ratio chart, the highest salaries in data science are found in fully remote roles (with a 100% remote ratio), with some reaching over 400,000. Roles that are partially remote (around 50-60% remote ratio) or fully in-office (0% remote) tend to have lower salaries, mostly below 200,000.

Furthermore, this suggests that fully remote positions in data science are often higher-paying compared to in-office or partially remote roles. This might indicate a higher demand for remote data science jobs or a premium placed on flexibility and the ability to work from anywhere.

# Tagorda
""")

def elbow_method_for_optimal_k():
    data_filtered = dfnew[['salary_in_usd', 'company_location']].copy()
    label_encoder = LabelEncoder()
    data_filtered['company_location_encoded'] = label_encoder.fit_transform(data_filtered['company_location'])
    data_filtered = data_filtered.drop(columns=['company_location'])

    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_filtered)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    st.pyplot()

def clusters_of_job_salaries_by_company_location():
    data_filtered = dfnew[['salary_in_usd', 'company_location']].copy()
    label_encoder = LabelEncoder()
    data_filtered['company_location_encoded'] = label_encoder.fit_transform(data_filtered['company_location'])
    data_filtered = data_filtered.drop(columns=['company_location'])

    kmeans = KMeans(n_clusters=4, random_state=42)
    data_filtered['cluster'] = kmeans.fit_predict(data_filtered)
    data_filtered['company_location'] = label_encoder.inverse_transform(data_filtered['company_location_encoded'])
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='company_location_encoded',
        y='salary_in_usd',
        hue='cluster',
        data=data_filtered,
        palette='viridis',
        style='cluster',
        s=100
    )
    plt.xlabel("Company Location (Encoded)")
    plt.ylabel("Salary in USD")
    plt.title("Clusters of Job Salaries by Company Location")
    plt.legend(title="Cluster")
    st.pyplot()

# Call the functions to visualize the data
elbow_method_for_optimal_k()
clusters_of_job_salaries_by_company_location()

st.markdown("""
**Unsuperivsed Learning for Data Science Salary According to Company Location**

To be able to identify an optimal balance for segmenting data, the elbow method was used, which was able to identify four clusters. These clusters show notable differences in data science salary distribution between locations.

*(For the encoded number for these exact locations: refer to the list below*)

The difference in salary are caused by several factors, but more significantly may be due to economic factors, industry presence, or cost of living.

For example, in high-cost regions such as North America and parts of europe, data science positions tend to offer higher salaries compared to regions with lower costs of living. This is likely due to the high demand for skilled data scientists in these regions, alongside the higher operational costs that companies in these areas face.

Additionally, the analysis shows that some countries with a strong presence of tech and finance industries (such as the US and UK) cluster together with higher salary ranges. These industries are major employers of data scientists, and they often offer competitive salaries to attract top talent.

""")

def location_encoding_tagorda():
    label_encoder = LabelEncoder()
    location_encoding = dict(zip(dfnew['company_location'].unique(), label_encoder.transform(dfnew['company_location'].unique())))
    for location, code in location_encoding.items():
        st.write(f"{location}: {code}")

location_encoding_tagorda()
