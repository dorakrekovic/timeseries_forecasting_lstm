from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense
import wandb




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

d = pd.read_csv('data/Copernicus_skenerovci_2019_2021.csv')
d['datetime'] = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in d['datetime']]
d.sort_values('datetime', inplace=True)

# Korištene značajke
features = ['t2m']

d = d.groupby('datetime', as_index=False)[features].mean()


# Param
lag = 24
n_ahead = 1
test_share = 0.1
epochs = 50
batch_size = 32
lr = 0.001
n_layer = 50
model_path = 'model_inference_tensorflow.h5'

features_final = ['t2m']

# Koristeni stupci za predvidanje
ts = d[features_final]

nrows = ts.shape[0]

# Initialize a new W&B run
wandb.init(project="Inference", config={"bs": 32})

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


class PredictionModel():
    def __init__( self, X, Y, n_outputs, n_lag, n_ft, n_layer, batch, epochs, lr, Xval=None, Yval=None ):
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

    def load_model(self, model_path):
        self.model.load_weights(model_path)

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


model.load_model(model_path)

predictions = model.predict(Xval)


# Invert scaling to get the actual predictions
predicted_values = (predictions * train_std['t2m']) + train_mean['t2m']

wandb.log({
    "predictions": wandb.Histogram(predicted_values),
    "true_values": wandb.Histogram(Yval.flatten())
})



predicted_dates = d['datetime'].values[-len(predicted_values):]
predicted_df = pd.DataFrame({
    'datetime': predicted_dates,
    't2m_predicted': predicted_values.flatten()
})


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

pivoted.reset_index(inplace=True)
wandb.log({"temperature_data": wandb.Table(data=pivoted)})

plt.figure(figsize=(30, 15))
plt.plot(pivoted.index, pivoted.t2m_unscaled_original - 276.15, color='blue', label='original')
plt.plot(pivoted.index, pivoted.t2m_unscaled_forecast - 276.15, color='red', label='forecast', alpha=0.6)
plt.title('Real-predicted values')
plt.legend()
wandb.log({"temperature_plot": wandb.Image(plt)})


pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['t2m_unscaled_original'] - pivoted['t2m_unscaled_forecast']
pivoted['res_mean'] = [pow(x,2) for x in pivoted['res']]

print(f"RMSE: {sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])} C")

#print(pivoted.res_mean.describe())

wandb.finish()
