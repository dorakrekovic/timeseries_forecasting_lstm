from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow import keras
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    # Izdvajanje značajki iz niza
    n_features = ts.shape[1]

    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

d = pd.read_csv('../data/allt2m_datetime.csv')
#print(f'Shape of data: {d.shape}')
#print(d.dtypes)

d['datetime'] = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in d['datetime']]
d.sort_values('datetime', inplace=True)

features = ['t2m']

# Satna razina
d = d.groupby('datetime', as_index=False)[features].mean()

d['date'] = [x.date() for x in d['datetime']]
d['hour'] = [x.hour for x in d['datetime']]
d['month'] = [x.month for x in d['datetime']]


d['day_cos'] = [np.cos(x * (2 * np.pi / 24)) for x in d['hour']]
d['day_sin'] = [np.sin(x * (2 * np.pi / 24)) for x in d['hour']]

dsin = d[['datetime', 't2m', 'hour', 'day_sin', 'day_cos']].head(25).copy()
dsin['day_sin'] = [round(x, 3) for x in dsin['day_sin']]
dsin['day_cos'] = [round(x, 3) for x in dsin['day_cos']]

d['timestamp'] = [x.timestamp() for x in d['datetime']]
# Sekunde u danu
s = 24 * 60 * 60
# Sekunde u godini
year = (365.25) * s
d['month_cos'] = [np.cos((x) * (2 * np.pi / year)) for x in d['timestamp']]
d['month_sin'] = [np.sin((x) * (2 * np.pi / year)) for x in d['timestamp']]

# hiperparametri
lag = 24
n_ahead = 1
test_share = 0.1

features_final = ['t2m', 'day_cos', 'day_sin', 'month_sin', 'month_cos']


# Koristeni stupci za predvidanje
ts = d[features_final]

nrows = ts.shape[0]

# Podjela u skup za treniranje i testiranje
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]

# Skaliranje podataka
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)

n_ft = X.shape[2]

# Podjela u skup za treniranje i testiranje
Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

# load model
from tensorflow.python.keras.saving.save import load_model

history = load_model('copernicus_1korak.h5')

# Usporedba predvidenih i stvarnih vrijednosti
yhat = [x[0] for x in history.predict(Xval)]
y = [y[0] for y in Yval]

# Spremanje predikcija
days = d['datetime'].values[-len(y):]

frame = pd.concat([
    pd.DataFrame({'day': days, 't2m': y, 'type': 'original'}),
    pd.DataFrame({'day': days, 't2m': yhat, 'type': 'forecast'})
])

# Invertirnanje skaliranja podataka
frame['t2m_unscaled'] = [(x * train_std['t2m']) + train_mean['t2m'] for x in frame['t2m']]

# Pivotiranje
pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['t2m_unscaled_original'] - pivoted['t2m_unscaled_forecast']
pivoted['res_unsc'] = [pow(x,2) for x in pivoted['res']]

pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['t2m_unscaled_original'] - pivoted['t2m_unscaled_forecast']
pivoted['res_mean'] = [pow(x,2) for x in pivoted['res']]
#print(pivoted.tail(10))

print(f"RMSE: {sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])} C")

#print(pivoted.res_mean.describe())
