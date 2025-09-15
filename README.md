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

# CODING
```
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df

```
# OUTPUT:
<img width="1427" height="782" alt="image" src="https://github.com/user-attachments/assets/88b0e7a5-9fa6-42af-8dc3-f2ede256b7dd" />

```
df.info()

```
# OUTPUT:
<img width="575" height="487" alt="image" src="https://github.com/user-attachments/assets/3e175ad1-ad95-4351-9291-ace5f7a0d799" />

```
df.describe()


```
# OUTPUT:
<img width="1077" height="429" alt="image" src="https://github.com/user-attachments/assets/a09858d1-77a4-4a02-810d-aa623a1ab1f2" />

```
df.dtypes

```
# OUTPUT:
<img width="427" height="626" alt="image" src="https://github.com/user-attachments/assets/0aafa2d3-8fc7-4526-9a93-45ec07221584" />

```
df.value_counts()

```
# OUTPUT:
<img width="1470" height="779" alt="image" src="https://github.com/user-attachments/assets/8f7cda88-ab24-4aa4-b62d-7d0eef21a549" />

```
df["Age"].value_counts()

```
# OUTPUT:
<img width="502" height="665" alt="image" src="https://github.com/user-attachments/assets/bdf6fad7-2a9b-425f-8161-e72e0df16b75" />

```
df.shape

```
# OUTPUT:
<img width="355" height="115" alt="image" src="https://github.com/user-attachments/assets/8c5db841-4273-466a-9b2e-807707017d49" />

```
df.set_index("PassengerId",inplace=True)
df.describe()

```
# OUTPUT:
<img width="963" height="474" alt="image" src="https://github.com/user-attachments/assets/f4dc2b6e-fb9d-458c-b07b-14afcd01303e" />

```
df.nunique()

```
# OUTPUT:
<img width="373" height="596" alt="image" src="https://github.com/user-attachments/assets/a80d262a-f727-44eb-b82e-e59eca49e3e4" />

```
 df["Survived"].value_counts()

```
# OUTPUT:
<img width="522" height="274" alt="image" src="https://github.com/user-attachments/assets/e61e4b76-c051-4877-9396-fe403def27c3" />

```
sns.countplot(data=df,x="Survived")

```
# OUTPUT:
<img width="912" height="648" alt="image" src="https://github.com/user-attachments/assets/370fed70-3534-4baa-be64-e7cab6fff518" />

```
df

```
# OUTPUT:
<img width="1430" height="673" alt="image" src="https://github.com/user-attachments/assets/16c3cdfe-0eb6-426c-bb00-55d7b1abcbf7" />

```
 df.Pclass.unique()

```
# OUTPUT:
<img width="412" height="104" alt="image" src="https://github.com/user-attachments/assets/e5e899a4-1899-4c27-b716-cf0592522f38" />

```
 df.rename(columns={'Sex':'Gender'},inplace=True)
 df

```
# OUTPUT:
<img width="1453" height="720" alt="image" src="https://github.com/user-attachments/assets/970e6565-28fc-424c-a4b3-3beb40244f95" />

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)

```
# OUTPUT:
<img width="1052" height="712" alt="image" src="https://github.com/user-attachments/assets/af5a83e6-e50d-4fcb-9f77-90e14202d3ca" />

```
 sns.catplot(x="Survived",hue="Gender",data=df,kind="count")

```
# OUTPUT:
<img width="942" height="719" alt="image" src="https://github.com/user-attachments/assets/62317463-b4fe-4496-b1e9-194185757389" />

```
 df.boxplot(column="Age",by="Survived")

```
# OUTPUT:
<img width="892" height="683" alt="image" src="https://github.com/user-attachments/assets/007fa86d-44fa-427b-b55f-7990f4fa445f" />

```
 sns.scatterplot(x=df["Age"],y=df["Fare"])

```
# OUTPUT:
<img width="953" height="638" alt="image" src="https://github.com/user-attachments/assets/66735da7-24b0-44a5-809c-eb83157ee1a4" />

```
 sns.jointplot(x="Age",y="Fare",data=df)

```
# OUTPUT:
<img width="968" height="823" alt="image" src="https://github.com/user-attachments/assets/5f776ad8-d676-45a3-8264-b6f78539a246" />

```
 fig, ax1 = plt.subplots(figsize=(8,5))
 plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)

```
# OUTPUT:
<img width="1025" height="664" alt="image" src="https://github.com/user-attachments/assets/abcd8817-e322-4558-ab07-59e68570d8e8" />

```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")

```
# OUTPUT:
<img width="1482" height="713" alt="image" src="https://github.com/user-attachments/assets/eb218131-72f7-46ba-9afd-b018ab2ad704" />

```
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True)

```
# OUTPUT:
<img width="895" height="678" alt="image" src="https://github.com/user-attachments/assets/39b77879-0f7f-4d31-881c-2c16147dfd1c" />


# RESULT
        Thus the Exploratory Data Analysis on the given data set was performed successfully.
