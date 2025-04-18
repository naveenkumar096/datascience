#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = pd.read_csv('D:\\Data-science\\Titanic_train.csv')
#data.head()
#%%
#data.info()
#%%
data.describe()
#%%
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1,inplace=True)
#%%
data.isna().sum()
#%%
data['Cabin'].unique()

#%%
data.drop_duplicates()
#%%
long_df = data['Cabin'].str.split(' ', expand=True)
split_cabin_long = pd.concat([data[['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']].loc[data.index.repeat(2)].reset_index(drop=True), long_df.stack().reset_index(drop=True)], axis=1)
split_cabin_long.columns = ['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','cabin_split']

#%%
split_cabin_long.describe()
#%%
split_cabin_long.drop('Ticket', axis=1, inplace=True)
#%% md
# ### Data visulation Histogram
#%%
fig, axs = plt.subplots(3, 2,figsize=(20, 20))
axs[0, 0].hist(split_cabin_long['Survived'], bins=50, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Histogram of Survived Data')

axs[0, 1].hist(split_cabin_long['Pclass'], bins=50, color='lightgreen', edgecolor='black')
axs[0, 1].set_title('Histogram of Pclass Data')

axs[1, 0].hist(split_cabin_long['Age'], bins=50, color='skyblue', edgecolor='black')
axs[1, 0].set_title('Histogram of Age Data')

axs[1, 1].hist(split_cabin_long['SibSp'], bins=50, color='lightgreen', edgecolor='black')
axs[1, 1].set_title('Histogram of SibSp Data')

axs[2, 0].hist(split_cabin_long['Parch'], bins=50, color='skyblue', edgecolor='black')
axs[2, 0].set_title('Histogram of Parch Data')


#%%
sns.set_style(style='darkgrid')
sns.pairplot(split_cabin_long)
#%%
data = split_cabin_long
#%% md
# ### Data Preprocessing:
#%%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data1 = data[['cabin_split']]
numeric_data = encoder.fit_transform(data1)
data2 = data[['Sex']]
numeric_data2 = encoder.fit_transform(data2)
data3 = data[['Embarked']]
numeric_data3 = encoder.fit_transform(data3)

#%%
data['cabin_numeric']=numeric_data
data['Gender'] = numeric_data2
data['Embarked_numeric'] = numeric_data3

#%%
train_data = data.copy()   
#%%
train_data.drop('Sex', axis=1, inplace=True)
train_data.drop('Embarked', axis=1, inplace=True)
train_data.drop('cabin_split', axis=1, inplace=True)
train_data.drop('Fare', axis=1, inplace=True)
#%%
train_data.drop('Cabin', axis=1, inplace=True)

#%%
train_data.isnull().sum()
train_data = train_data.dropna()

#%% md
# ####  Model Building:
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#%%
X = train_data.iloc[:,1:]
Y = train_data.iloc[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#%%
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)
#%%
# Fitting logistic regression to the training set
Classifier = LogisticRegression(random_state=0)
Classifier.fit(X_train, Y_train)
#%% md
# #### Make Predictions and Evaluate the Model
# 
#%%
y_pred = Classifier.predict(X_test)

#%%
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)
#%%
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
#%% md
# ##### ROC (Receiver Operating Curve)
#%%
from sklearn.metrics import roc_curve, auc
#%%
y_prob = Classifier.predict_proba(X_test)[:, 1]
#%%
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")
#%%
# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (no skill)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#%%
import streamlit as st

#%%
# Streamlit App
st.title("Titanic Survived Classification")

st.write("""
This app uses logistic regression to classify whether Titanic Passenger Survived ratio.
""")
#%%
print(data['Embarked_numeric'].unique())
#%%

df = pd.DataFrame({'Cabin_name':data.cabin_split.unique(),'Cabin_No':data.cabin_numeric.unique()})
df1=df[df['Cabin_name']=='C85']
#%%
pclass_input = st.sidebar.selectbox('Pclass', train_data.Pclass.unique())
age_input = st.sidebar.number_input('Please enter age', min_value=0,max_value=150)
sib_input =st.sidebar.selectbox('SibSp', train_data.SibSp.unique())
prach_input =st.sidebar.selectbox('prach', train_data.Parch.unique())
cabin_input =st.sidebar.selectbox('Cabin', data.cabin_split.unique())
Gender_input =st.sidebar.selectbox('Gender', data.Sex.unique())
Embarked_numeric_input =st.sidebar.selectbox('Embraked', data.Embarked.unique())
df1=df[df['Cabin_name']==cabin_input]
gender = 0
if Gender_input == 'Male': gender = 1

embarked = 3
if Embarked_numeric_input == 'C': 
  embarked = 0 
elif Embarked_numeric_input == 'Q': 
  embarked = 1
elif Embarked_numeric_input == 'S': 
  embarked = 2


#%%
user_input = np.array([[pclass_input, pclass_input, sib_input, prach_input, df.loc[0, 'Cabin_No'], gender, embarked]])
print(user_input)

#%%
user_input_scaled = sc.transform(user_input)
print(user_input_scaled)
#%%
prediction = Classifier.predict(user_input_scaled)
print(prediction)
#%%
if prediction[0] == 1:
    st.write("### The Customer is **Survived**!")
else:
    st.write("### The Customer is **Not Survived**!")