import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from arachnepy import WebRegressor
import numpy as np
from pathlib import Path

df_addr = Path(__file__).parent / "job_salary_prediction_dataset.csv"
df = pd.read_csv(df_addr)

x = df.drop(columns=["salary"])
y = df["salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state = 11,
    test_size = 0.2
)

transformations = ColumnTransformer([
    ("numerical", RobustScaler(), make_column_selector(dtype_include="int64")),
    ("one-hot", OneHotEncoder(drop='first', sparse_output=False), ["job_title", "industry", "remote_work", "location", "company_size"]),
    ("education", OrdinalEncoder(categories=[["High School", "Diploma", "Bachelor", "Master", "PhD"]]), ["education_level"]),
])

x_train = transformations.fit_transform(x_train)
x_test = transformations.transform(x_test)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)

y_test_scaled = y_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
)

web = WebRegressor([x_train.shape[1],32, 16, 1], learning_rate = 0.05, intialization_strength=0.1, epochs = 2000)

web.spin(x_train[:5000], y_train_scaled[:5000]) # training on first 5 samples because this dataset has 200k records

train_pred = web.predict(x_train)
train_mse = np.mean((train_pred.data - y_train_scaled.reshape(-1,1))**2)

test_pred = web.predict(x_test)
test_mse = np.mean((test_pred.data - y_test_scaled)**2)

print(f"Train MSE: {train_mse}"
      f"Test MSE: {test_mse}"
      f"Test R2 score: {web.score(x_test, y_test_scaled)}")