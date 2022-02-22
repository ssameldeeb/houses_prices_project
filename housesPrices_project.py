# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,median_absolute_error

# readings the data
data = pd.read_csv("houses_prices.csv")

# git information from the data file
print(data.shape)
print(data.columns.values)
print(data.head())
print(data.isnull().sum())

# drop the columns are not important
data = data.drop(["code"],axis=1)
print(data.head())

# clean data["houses_sold"]
data["houses_sold"].fillna(data["houses_sold"].mean() , inplace=True)
print(data.isnull().sum())

# clean data["no_of_crimes"]
data["no_of_crimes"].fillna(0, inplace=True)
print(data.isnull().sum())
data.loc[data["no_of_crimes"] > 0 , "no_of_crimes"] = 1
print(data["no_of_crimes"].tail(5))
print(data["no_of_crimes"].value_counts())
data.rename(columns={"no_of_crimes":"no_has_crimes"},inplace=True)
print(data.head())
sns.countplot(data["no_has_crimes"],palette="hot",alpha=0.5)
plt.show()

# clean data["date"]
data["date"]=pd.to_datetime(data["date"])
print(data.dtypes)
data["date"]=data["date"].dt.year
print(data.head())

# clean data["area"]
print(data["area"].value_counts())
print(data["area"].unique())
print(data.groupby("area")["average_price"].sum())
plt.figure(figsize=(10,6))
plt.plot(data["area"],data["average_price"])
plt.show()
Lo = LabelEncoder()
data["area"]=Lo.fit_transform(data["area"])
plt.figure(figsize=(10,6))
plt.scatter(data["area"],data["average_price"])
plt.show()

# show the effic columns together
sns.pairplot(data)
plt.show()

# show the correlation in the data
sns.heatmap(data.corr(),annot=True)
plt.show()

# split the data
x = data.drop("average_price",axis=1)
print(x.head())
y = data["average_price"]
print(y.head())

# split the data for testing and training
x_train , x_test , y_train , y_test = train_test_split(x , y , train_size=0.7)

# use different kind of regression to get the the best score and predict
# use LinearRegression 
m = LinearRegression()
m.fit(x_train,y_train)
print(m.score(x_train,y_train))
print(m.score(x_test,y_test))
y_pred = m.predict(x_test)
print(y_pred)
MS = mean_squared_error(y_test, y_pred)
print(MS)
MA = median_absolute_error(y_test, y_pred)
print(MA)

# use MLPRegressor 
m = MLPRegressor(activation="identity",max_iter=1000)
m.fit(x_train,y_train)
print(m.score(x_train,y_train))
print(m.score(x_test,y_test))

# use KNeighborsRegressor 
m = KNeighborsRegressor(n_neighbors=101)
m.fit(x_train,y_train)
print(m.score(x_train,y_train))
print(m.score(x_test,y_test))

# use RandomForestRegressor
print("The best score regression")
m = RandomForestRegressor(max_depth=15,n_estimators=30)
m.fit(x_train,y_train)
print(m.score(x_train,y_train))
print(m.score(x_test,y_test))
y_pred = m.predict(x_test)
print(y_pred)

# know the cost(j)
MS = mean_squared_error(y_test, y_pred)
print(MS)
MA = median_absolute_error(y_test, y_pred)
print(MA)

out = pd.DataFrame({"y_test":y_test,"y_pred":y_pred})
# out.to_csv("the predict.csv")
