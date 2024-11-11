import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Clean columns
def process_data(df):
    df['Close/Last'] = df['Close/Last'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    return df

# Prepare training
def TrainTest(df, prop=0.85):
    I = int(prop * len(df))
    train = df[:I]
    test = df[I:]
    return train, test

def Inputs(dataset, window=100, output=30):  
    n = len(dataset)
    training_data = []

    for w in range(window, n - output + 1):
        a1 = dataset['Open'][w - window:w].values.astype(np.float32)
        a2 = dataset['High'][w - window:w].values.astype(np.float32)
        a3 = dataset['Low'][w - window:w].values.astype(np.float32)
        a4 = dataset['Volume'][w - window:w].values.astype(np.float32)
        
        features = np.concatenate([a1, a2, a3, a4])
        
        b1 = dataset['Close/Last'][w:w + output].values.astype(np.float32)
        
        training_data.append([features, b1])
 
    IN = [torch.tensor(item[0], dtype=torch.float32) for item in training_data]
    OUT = [torch.tensor(item[1], dtype=torch.float32) for item in training_data]
 
    return torch.stack(IN), torch.stack(OUT)

def Outputs(dataset, window):
    a1 = dataset['Open'][-window:].values.astype(np.float32)
    a2 = dataset['High'][-window:].values.astype(np.float32)
    a3 = dataset['Low'][-window:].values.astype(np.float32)
    a4 = dataset['Volume'][-window:].values.astype(np.float32)
    
    X = np.concatenate([a1, a2, a3, a4])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0) 
    
    return X

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)  
        self.layer2 = nn.Linear(512, 256)  
        self.layer3 = nn.Linear(256, 128)  
        self.layer4 = nn.Linear(128, output_size)  
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

data = pd.read_csv('AAPL.csv')
data = process_data(data)

# Params
epochs = 5000  
window = 100  
output = 30  
learning_rate = 0.0001  

train, test = TrainTest(data)

X, Y = Inputs(train, window=window, output=output)

# Normalize
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / X_std 

Y_mean = Y.mean()
Y_std = Y.std()
Y = (Y - Y_mean) / Y_std  
model = Model(input_size=window * 4, output_size=output)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training 
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0: 
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

XX = Outputs(test, window)

XX = (XX - X_mean) / X_std  

with torch.no_grad():
    test_outputs = model(XX)

predictions = test_outputs.squeeze().detach().numpy()
predictions = predictions * Y_std.numpy() + Y_mean.numpy()

print("Predicted Prices for the next 30 days:")
print(predictions)
print("\nFirst 5 Predicted Prices for the next 30 days:")
print(predictions[:5])  

# Visualize 
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

history = test['Close/Last'].values  
xb = list(range(len(history)))  
ax.plot(xb, history, color='red', label='Historical Price')

xc = list(range(len(history), len(history) + len(predictions)))  
ax.plot(xc, predictions, color='green', label='Predicted Price')

plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()