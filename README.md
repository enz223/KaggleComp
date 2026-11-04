# KaggleComp

## Our group: Grace Liu, Helen Deng, Evelyn Zhang
We decided to use three features: neighborhood (because location is a big driver of home price), lot area (bigger lots would cost more), and pool area (because pools are premium features of a home). since neighborhood is a text category, we used OneHotEncoder to turn each neighborhood into a 0/1 column. then, we combined the encoding and model in the pipeline.

# Import libraries for data processing and analysis
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

# Load the training dataset containing house features and prices
df = pd.read_csv('/content/sample_data/train.csv')

# Preview the first 5 rows
df.head(5)

# Use Neighborhood, LotArea, and PoolArea for predictors. Compare it to SalePrice
X = df[['Neighborhood', 'LotArea', 'PoolArea']]
y = df['SalePrice']

# Load the test dataset to help with predictions
test = pd.read_csv('/content/sample_data/test.csv')

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# 
from sklearn.compose import make_column_transformer
transformer = make_column_transformer((OneHotEncoder(drop = 'first', handle_unknown = 'ignore'), ['Neighborhood']),
                                      remainder = 'passthrough')

#
pipe = Pipeline([('transform', transformer), ('model', LinearRegression())])

#
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

# Display RMSE
mean_squared_error(y_test, preds) ** 0.5
