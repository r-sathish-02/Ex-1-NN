<H3>ENTER YOUR NAME: SATHISH R</H3>
<H3>ENTER YOUR REGISTER NO: 212222230138</H3>
<H3>EX. NO.1</H3>
<H3>DATE: </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

## PROGRAM:

Import Libraries
```python

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

Read the dataset

```python
df=pd.read_csv("Churn_Modelling.csv")
```
Checking Data
```python
df.head()
df.tail()
df.columns
```

Check the missing data
````python
df.isnull().sum()
```

Check for Duplicates
```python
df.duplicated()
```
Assigning Y
```python
y = df.iloc[:, -1].values
print(y)
````
Check for duplicates
```python
df.duplicated()
```
Check for outliers
```python
df.describe()
```
Dropping string values data from dataset
```python
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```
Normalize the dataset
```python
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
Split the dataset
```python
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
Training and testing model
```python
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```



## OUTPUT:

Data checking:

![307074014-1652081a-e434-418c-8bfa-ffd0097a5f58](https://github.com/user-attachments/assets/8ba75174-8100-4bdb-89be-7d3e00780de9)


Missing Data:

![307074349-c22867c3-a304-4890-849a-d3f88e2278c3](https://github.com/user-attachments/assets/088a6881-2f44-4cc7-971f-0692b332cdbd)

Duplicates identification:


![307074474-beb23c01-7e40-4a4f-a743-b803cd87154d](https://github.com/user-attachments/assets/31fc2ec1-36e6-42f5-b853-71bc36e5e7b3)

Vakues of 'Y':

![307074646-12a5643f-4058-4095-9a6e-10ecc1ff4357](https://github.com/user-attachments/assets/6290b84b-f75a-4285-93ec-3fc87639dc03)

Outliers:

![307074794-0a937472-82aa-47f8-94fd-6604fba691f5](https://github.com/user-attachments/assets/82446eac-38f8-4805-935c-e0303a184e73)

Checking datasets after dropping string values data from dataset:

![307074934-b0b2687b-a0c6-4c88-82c0-98299d2a64ca](https://github.com/user-attachments/assets/2eb93b8e-89bf-4ff1-bf6e-ff7f37aeed50)

Normalize the dataset:

![307075193-b982e971-f9dc-4d8d-8fcc-58163db111f2](https://github.com/user-attachments/assets/816b82c0-b589-4a55-b9ac-3a223a3539b8)

Split the dataset:

![307075366-c0b451e1-40f7-4551-8c2f-a00d6003b38e](https://github.com/user-attachments/assets/f1d746e9-d167-426d-be08-b8c9138b2ea7)

Training and testing model:

![307075605-01fade29-e4e7-4e0f-a7d7-c0d554cb1885](https://github.com/user-attachments/assets/f533a2d5-f4a2-4bf5-8504-a6f70b2966d4)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


