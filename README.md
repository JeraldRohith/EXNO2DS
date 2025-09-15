# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

# CODING:
```
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df

```
# OUTPUT:
<img width="1513" height="667" alt="image" src="https://github.com/user-attachments/assets/5b64da74-6ec6-42d5-aca9-7ca4d4a6fc22" />

```
df.info()

```
# OUTPUT:
<img width="410" height="469" alt="image" src="https://github.com/user-attachments/assets/571e3942-f6f9-4c59-a78a-b2634fe02198" />

```
df.describe()

```
# OUTPUT:
<img width="872" height="419" alt="image" src="https://github.com/user-attachments/assets/ff7eb7ad-51e9-4f4e-af6d-b127af09d7b1" />

```
df.dtypes

```
# OUTPUT:
<img width="242" height="609" alt="image" src="https://github.com/user-attachments/assets/57480cbb-f368-4b07-bab6-ced664d2f2a9" />

```
df.value_counts()

```
# OUTPUT:
<img width="1511" height="648" alt="image" src="https://github.com/user-attachments/assets/29dd19d2-4246-4045-a6e6-0d69f7c2178a" />

```
df["Age"].value_counts()

```
# OUTPUT:
<img width="252" height="651" alt="image" src="https://github.com/user-attachments/assets/06237d83-02e4-4600-b255-4a941a74eb09" />

```
df.shape

```
# OUTPUT:
<img width="109" height="86" alt="image" src="https://github.com/user-attachments/assets/7e1d4d4b-907d-4bfc-bf1a-d830844a5967" />

```
df.set_index("PassengerId",inplace=True)
df.describe()

```
# OUTPUT:
<img width="764" height="435" alt="image" src="https://github.com/user-attachments/assets/f7a5596b-c51c-45c9-8ab7-fb48823e2721" />

```
df.nunique()

```
# OUTPUT:
<img width="190" height="568" alt="image" src="https://github.com/user-attachments/assets/184f99b4-663f-4ea4-9d87-8a7d98756c9a" />

```
 df["Survived"].value_counts()

```
# OUTPUT:
<img width="354" height="250" alt="image" src="https://github.com/user-attachments/assets/d85b0d58-ec65-4888-b160-e2445c10bd44" />

```
sns.countplot(data=df,x="Survived")

```
# OUTPUT:
<img width="776" height="624" alt="image" src="https://github.com/user-attachments/assets/a5102834-30db-432c-8e50-29c1d1d6f006" />

```
df

```
# OUTPUT:
<img width="1429" height="605" alt="image" src="https://github.com/user-attachments/assets/4dfabef3-b064-4340-89aa-845770a0e8c2" />

```
 df.Pclass.unique()

```
# OUTPUT:
<img width="204" height="86" alt="image" src="https://github.com/user-attachments/assets/8956860f-deb2-4a46-8770-f91cda1e2247" />

```
 df.rename(columns={'Sex':'Gender'},inplace=True)
 df
```
# OUTPUT:
<img width="1398" height="621" alt="image" src="https://github.com/user-attachments/assets/3430e008-78f5-4656-97f7-a07f44e912d3" />

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)

```
# OUTPUT:
<img width="940" height="698" alt="image" src="https://github.com/user-attachments/assets/5fce5f76-f766-4cd4-bd1b-874e0b45e76e" />

```
 sns.catplot(x="Survived",hue="Gender",data=df,kind="count")

```
# OUTPUT:
<img width="802" height="697" alt="image" src="https://github.com/user-attachments/assets/51fe03df-b754-44c5-86f8-d9edc1473e14" />

```
 df.boxplot(column="Age",by="Survived")

```
# OUTPUT:
<img width="743" height="659" alt="image" src="https://github.com/user-attachments/assets/f8a9903e-ac0b-49e5-ad8a-345591d35cbf" />

```
 sns.scatterplot(x=df["Age"],y=df["Fare"])

```
# OUTPUT:
<img width="754" height="624" alt="image" src="https://github.com/user-attachments/assets/b6ff6c82-7db5-40df-83bb-d4781aa4d68e" />

```
 sns.jointplot(x="Age",y="Fare",data=df)

```
# OUTPUT:
<img width="768" height="823" alt="image" src="https://github.com/user-attachments/assets/c3f09cf6-c925-4466-b18c-7ce0a0fdce56" />

```
 fig, ax1 = plt.subplots(figsize=(8,5))
 plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)

```
# OUTPUT:
<img width="896" height="647" alt="image" src="https://github.com/user-attachments/assets/f3987aea-a65c-4e45-878c-e801733da470" />

```
 sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")

```
# OUTPUT:
<img width="1336" height="696" alt="image" src="https://github.com/user-attachments/assets/fc37f473-14fe-4d4b-80ed-5fe52406c6c1" />

```
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True)

```
# OUTPUT:
<img width="699" height="629" alt="image" src="https://github.com/user-attachments/assets/e206a401-1038-4ac1-b831-4d00c4989e66" />

# RESULT
        Thus the Exploratory Data Analysis was done successfully.
