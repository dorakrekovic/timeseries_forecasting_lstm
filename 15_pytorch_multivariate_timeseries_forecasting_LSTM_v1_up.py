#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/duongtrung/Pytorch-tutorials/blob/main/15_pytorch_multivariate_timeseries_forecasting_LSTM_v1_up.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# dataset: https://finance.yahoo.com/quote/GE/history/
# select Max in the Time Period if you want to get all datasets until your current date


# In[2]:


# General design
# One-step-ahead
# Multivariate
# The first data's column is the target


# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
import wandb
# In[4]:


df = pd.read_csv("Skenderovci_3y_multivariate.csv")

# In[5]:


df.head(4)

# In[6]:


df.describe()

# In[7]:


df.info()

# In[8]:


# separate dates for future plotting
train_dates = pd.to_datetime(df['datetime'])

# In[9]:


train_dates.head(4)

# In[10]:


# variables for training
cols = list(df)[1:6]  # ['t2m', 'stl1', 'ssr', 'e', 'tp']
print(cols)

# In[11]:


number_of_features = len(cols)

# In[12]:


df_for_training = df[cols].astype(float)

# In[13]:


df_for_training.plot.line()

# In[14]:


scaler = StandardScaler()  # RobustScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

# In[15]:


df_for_training_scaled.shape

# In[16]:


plt.plot(df_for_training_scaled)
plt.show()

# In[17]:


np.savetxt("scaled_multivariate.csv", df_for_training_scaled, delimiter=',')


# In[18]:


##splitting dataset into train and test split for time series data
training_size = int(len(df_for_training_scaled) * 0.65)
test_size = len(df_for_training_scaled) - training_size
train_data, test_data = df_for_training_scaled[0:training_size, :], df_for_training_scaled[
                                                                    training_size:len(df_for_training_scaled), :]

# In[19]:


train_data.shape, test_data.shape

# In[20]:


train_data[0], test_data[0]

# In[21]:


train_data[1], test_data[1]

# In[ ]:


# In[22]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# In[23]:


past_observation = 24


# In[24]:


class MultiSequenceDataset(Dataset):
    def __init__(self, data, past_len=past_observation):
        self.data = data
        self.data = torch.from_numpy(data).float()  # .view(-1)
        self.past_len = past_len

    def __len__(self):
        return len(self.data) - self.past_len - 1

    def __getitem__(self, index):
        return self.data[index: index + self.past_len, :].reshape(-1, 1), self.data[index + self.past_len, :][0]


train_dataset = MultiSequenceDataset(train_data, past_len=past_observation)
test_dataset = MultiSequenceDataset(test_data, past_len=past_observation)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size, drop_last=True)


# In[ ]:


# In[30]:


class Lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(Lstm_model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=0,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))

        final_out = self.fc(out[-1])
        return final_out, hn, cn

    def predict(self, x):
        hn, cn = self.init()
        final_out = self.fc(out[-1])
        return final_out

    def init(self):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0

input_dim = 1
hidden_size = 30
num_layers = 5


model = Lstm_model(input_dim, hidden_size, num_layers).to(device)


loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

args = {
    'input_dim': 1,
    'hidden_size': 30,
    'num_layers': 5,
    'past_observation': 24,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'loss_function': 'MSELoss',
    'num_epochs': 50,
}
# In[35]:
wandb.init(project="multivariate", config=args)

model = Lstm_model(input_dim, hidden_size, num_layers).to(device)


def train(dataloader):
    hn, cn = model.init()
    model.train()
    losses = []
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        out, hn, cn = model(x.reshape(past_observation * number_of_features, batch_size, 1), hn, cn)
        loss = loss_fn(out.reshape(batch_size), y)
        losses.append(loss.item())
        hn = hn.detach()
        cn = cn.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train = np.sum(losses) / len(dataloader)
    # Logging the losses
    wandb.log({"epoch/epoch": epoch, "epoch/loss": loss_train})
    # print(f"train loss: {loss_train:>8f} ")
    return loss_train


def test(dataloader):
    hn, cn = model.init()
    model.eval()
    losses = []
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        out, hn, cn = model(x.reshape(past_observation * number_of_features, batch_size, 1), hn, cn)
        loss = loss_fn(out.reshape(batch_size), y)
        losses.append(loss.item())

    loss_test = np.sum(losses) / len(dataloader)
    wandb.log({"epoch/epoch": epoch, "epoch/val_loss": loss_test})
    # print(f"test loss: {loss_test:>8f} ")
    return loss_test


# In[37]:


epochs = 50
train_losses = []
test_losses = []
for epoch in range(epochs):
    train_losses.append(train(train_dataloader))
    test_losses.append(test(test_dataloader))
    if epoch % 5 == 0:
        print(f"epoch {epoch} ")
        print(train_losses[epoch], test_losses[epoch])

# In[38]:


plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()

# In[39]:


import math
from sklearn.metrics import mean_squared_error
import numpy as np


def calculate_metrics(data_loader):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        hn, cn = model.init()

        for batch, item in enumerate(data_loader):
            x, y = item
            y = y.reshape(batch_size, -1)
            x, y = x.to(device), y.to(device)
            x = x.view(past_observation * number_of_features, batch_size, 1)
            pred = model(x, hn, cn)[0]
            pred = np.repeat(pred.detach().cpu(), number_of_features, axis=-1)
            pred = scaler.inverse_transform(pred)  # .reshape(-1)
            y = np.repeat(y.detach().cpu(), number_of_features, axis=-1)
            y = scaler.inverse_transform(y)  # .reshape(-1)
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)
        # print(pred_arr[21],y_arr[21])
        return math.sqrt(mean_squared_error(y_arr, pred_arr))


# In[40]:


# calculating final loss metrics
print("final loss metrics after inversing scaling")
print(f"train mse loss {calculate_metrics(train_dataloader)}")
print(f"test mse loss {calculate_metrics(test_dataloader)}")

# In[ ]:


# In[ ]:




