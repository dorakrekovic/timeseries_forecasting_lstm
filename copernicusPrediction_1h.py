from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense


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

d = pd.read_csv('data/allt2m_datetime.csv')
#print(f'Shape of data: {d.shape}')
#print(d.dtypes)

d['datetime'] = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in d['datetime']]
d.sort_values('datetime', inplace=True)
#print(f"First date {min(d['datetime'])}")
#print(f"Most recent date {max(d['datetime'])}")

# Korištene značajke
features = ['t2m']

# Satna razina
d = d.groupby('datetime', as_index=False)[features].mean()
#print(d[features].describe())

d['date'] = [x.date() for x in d['datetime']]
d['hour'] = [x.hour for x in d['datetime']]
d['month'] = [x.month for x in d['datetime']]

#d.boxplot('t2m', by='hour', figsize=(12, 8), grid=False)
#plt.show()
#d.boxplot('t2m', by='month', figsize=(12, 8), grid=False)
#plt.show()

# Kreiranje cikličke dnevne značajke
d['day_cos'] = [np.cos(x * (2 * np.pi / 24)) for x in d['hour']]
d['day_sin'] = [np.sin(x * (2 * np.pi / 24)) for x in d['hour']]

dsin = d[['datetime', 't2m', 'hour', 'day_sin', 'day_cos']].head(25).copy()
dsin['day_sin'] = [round(x, 3) for x in dsin['day_sin']]
dsin['day_cos'] = [round(x, 3) for x in dsin['day_cos']]
#print(dsin)

d['timestamp'] = [x.timestamp() for x in d['datetime']]
# Sekunde u danu
s = 24 * 60 * 60
# Sekunde u godini
year = (365.25) * s
d['month_cos'] = [np.cos((x) * (2 * np.pi / year)) for x in d['timestamp']]
d['month_sin'] = [np.sin((x) * (2 * np.pi / year)) for x in d['timestamp']]

# Broj lagova koristenih u modelu
lag = 24

# Broj koraka unaprijed  koji se predvidaju
n_ahead = 1

# Udio skupa korišten za testiranje
test_share = 0.1

# Broj epoha za treniranje
epochs = 100

# Batch size
batch_size = 128

# Stopa ucenja
lr = 0.001

# Broj neurona u LSTM sloju
n_layer = 50


features_final = ['t2m', 'day_cos', 'day_sin', 'month_sin', 'month_cos']

#print(d[features_final].head(10))

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

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of the target data: {Ytrain.shape}")

print(f"Shape of validation data: {Xval.shape}")
print(f"Shape of the validation target data: {Yval.shape}")


class PredictionModel():
    def __init__( self, X, Y, n_outputs, n_lag, n_ft, n_layer, batch, epochs, lr, Xval=None, Yval=None,
                  min_delta=0.001, patience=10):
        lstm_input = Input(shape=(n_lag, n_ft))
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch
        self.epochs = epochs
        self.n_layer = n_layer
        self.lr = lr
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.min_delta = min_delta
        self.patience = patience

    def trainCallback(self):
        return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)

    #def modelSave(self):
        #return ModelCheckpoint('copernicus_1korak___test.h5', monitor='loss', mode='min', save_best_only=True)

    def train(self):
        empty_model = self.model
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        empty_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=optimizer)
        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.Xval, self.Yval),
                shuffle=False,
                callbacks=[self.trainCallback(), self.modelSave()])
        else:
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=False,
                callbacks=[self.trainCallback()])
        self.model = empty_model
        return history

    def predict(self, X):
        return self.model.predict(X)

model = PredictionModel(
    X=Xtrain,
    Y=Ytrain,
    n_outputs=n_ahead,
    n_lag=lag,
    n_ft=n_ft,
    n_layer=n_layer,
    batch=batch_size,
    epochs=epochs,
    lr=lr,
    Xval=Xval,
    Yval=Yval,
)
model.model.summary()

history = model.train()

loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

n_epochs = range(len(loss))

plt.figure(figsize=(30, 15))
plt.plot(n_epochs, loss, 'r', label='Train_loss', color='blue')
if val_loss is not None:
    plt.plot(n_epochs, val_loss, 'r', label='Val_loss', color='red')
plt.legend(loc=0)
plt.xlabel('Epoha')
plt.ylabel('RMSE')
plt.show()

# Usporedba predvidenih i stvarnih vrijednosti
yhat = [x[0] for x in model.predict(Xval)]
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


#print(pivoted.tail(10))

plt.figure(figsize=(30, 15))
plt.plot(pivoted.index, pivoted.t2m_original, color='blue', label='original')
plt.plot(pivoted.index, pivoted.t2m_forecast, color='red', label='forecast', alpha=0.6)
plt.title('Predviđanje temperature - skalirani podatci')
plt.legend()
plt.show()

plt.figure(figsize=(30, 15))
plt.plot(pivoted.index, pivoted.t2m_unscaled_original - 276.15, color='blue', label='original')
plt.plot(pivoted.index, pivoted.t2m_unscaled_forecast - 276.15, color='red', label='forecast', alpha=0.6)
plt.title('Predviđanje temperature')
plt.legend()
plt.show()

pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['t2m_unscaled_original'] - pivoted['t2m_unscaled_forecast']
pivoted['res_mean'] = [pow(x,2) for x in pivoted['res']]
#print(pivoted.tail(10))

print(f"RMSE: {sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])} C")

print(pivoted.res_mean.describe())
