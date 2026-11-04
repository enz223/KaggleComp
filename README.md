# KaggleComp

## Our group: Grace Liu, Helen Deng, Evelyn Zhang
For our original version, we decided to use three features: neighborhood (because location is a big driver of home price), lot area (bigger lots would cost more), and pool area (because pools are premium features of a home). Since neighborhood is a text category, we used OneHotEncoder to turn each neighborhood into a 0/1 column. Then, we combined the encoding and model in the pipeline.

For our updated version, (see below original version), we checked which numeric columns correlate most strongly with `SalePrice` and kept the top ones (overall quality, living area, basement and garage size, bathrooms, year built/remodeled, etc.). We also added quality ratings and key categories like `Neighborhood`, `MSZoning` and `SaleCondition`. We dropped `PoolQC`, `Fence`, `MiscFeature`, `FireplaceQu` and `Alley` because most of their values are missing.

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



**For our final submission**:
# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

# Create Dataframe
df = pd.read_csv('/content/sample_data/train.csv')
df.head(5)
test = pd.read_csv('/content/sample_data/test.csv')

from sklearn.compose import make_column_transformer

df.select_dtypes(['int', 'float']).corr()['SalePrice'].sort_values(ascending = False)

# Get all the columns with missing values
df.columns[df.isna().sum() > 0]

# Percentage of missing values
df[df.columns[df.isna().sum() > 0]].isna().sum()/len(df)

df = df.drop(columns = ['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley'])

# numeric columns
num_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars',
            'GarageArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
            'YearBuilt', 'YearRemodAdd', 'LotArea', 'MasVnrArea']

# quality columns that have an order
ord_cols = ['ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC', 'GarageQual']

# categories with no order
ohe_cols = ['Neighborhood', 'MSZoning', 'BldgType', 'Foundation',
            'GarageType', 'CentralAir', 'SaleCondition']

cols = num_cols + ord_cols + ohe_cols

vals = [['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
ohe_pipe = Pipeline([('impute', SimpleImputer(strategy = 'constant', fill_value = 'None')),
                     ('ohe', OneHotEncoder(drop = 'first', handle_unknown = 'ignore'))])


qual = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ord_vals = [qual, qual, qual, qual, qual, qual,
            ['None', 'No', 'Mn', 'Av', 'Gd'],    # BsmtExposure
            ['None', 'Unf', 'RFn', 'Fin']]       # GarageFinish

ord_pipe = Pipeline([('impute', SimpleImputer(strategy = 'constant', fill_value = 'None')),
                     ('ord', OrdinalEncoder(categories = vals*len(ord_cols)))])

transformer = make_column_transformer((ohe_pipe, ohe_cols),
                                      (ord_pipe, ord_cols),
                                      (SimpleImputer(), num_cols))


X = df[cols]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


model_pipeline = Pipeline([('transform', transformer), ('model', LinearRegression())])

model_pipeline.fit(X, y)
preds = model_pipeline.predict(X_test)

mean_squared_error(y_test, preds) ** 0.5

pdf = pd.DataFrame({'Id': test.index, 'SalePrice': test_preds})
pdf.to_csv('preds.csv', index = False)
