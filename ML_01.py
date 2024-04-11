import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('ML_01.csv')
X=data[['bedrooms', 'bathrooms', 'sqft_living']]
y=data['price']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
print(f"Training R-squared: {train_score:.3f}")
print(f"Testing R-squared: {test_score:.3f}")

def predict_price(bedrooms, bathrooms, sqft_living):
    input_data=[[bedrooms, bathrooms, sqft_living]]
    input_data_scaled=scaler.transform(input_data)
    predicted_price=model.predict(input_data_scaled)
    return predicted_price[0]

bedrooms=int(input("Enter the number of bedrooms: "))
bathrooms=int(input("Enter the number of bathrooms: "))
sqft_living=int(input("Enter the square footage: "))
predicted_price=predict_price(bedrooms, bathrooms, sqft_living)
print("Predicted price: Rs.",int(predicted_price))
