#%%
# Importação
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  BaggingRegressor
from  sklearn.model_selection import  train_test_split
import pandas as pd

#%%
# Obtendo o dado
seed:int = 100
data = pd.read_csv('data/BostonHousing.csv')
y = data['medv']
X = data.drop('medv', axis=1)
print(data.head())
#%%
# Criando Teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=seed)
#%%
# Normalização dos dados
scaler = StandardScaler()
X_train_t = scaler.fit_transform(X_train)
X_test_t = scaler.transform(X_test)
#%%
# Regressor único

model = KNeighborsRegressor(n_neighbors=7)
model.fit(X_train_t, y_train)
r2_simple = model.score(X_test_t,y_test)

#%%
base_model = KNeighborsRegressor(n_neighbors=7)
bagging = BaggingRegressor(base_estimator=base_model,
                           n_estimators=10,
                           random_state=seed)
bagging.fit(X_train_t,y_train)
r2_bag = bagging.score(X_test_t, y_test)
#%%
print(f'r2_simple:{r2_simple}\n')
print(f'r2_bag:{r2_bag}\n')

#

