import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
MAX_TIME_STEP = 30


from google.colab import drive
drive.mount('/content/drive')


df_raw = pd.read_csv("/content/drive/MyDrive/data/archive(1)/LeagueofLegends.csv")
df_raw = df_raw[df_raw['gamelength'] >= MAX_TIME_STEP]
df_raw.reset_index(drop = True, inplace = True)
print(len(df_raw))


df = pd.read_csv("/content/drive/MyDrive/data/archive(1)/LeagueofLegends.csv")
df = df[df['gamelength'] >= MAX_TIME_STEP]
df.reset_index(drop = True, inplace = True)
matches = len(df)
print(matches)


def count_item(items):
    count = np.zeros(MAX_TIME_STEP, dtype=np.int8)
    for timestep in range(MAX_TIME_STEP) :
        for item in items:
            if item[0] <= timestep + 1:
                count[timestep] += 1
    return count


from ast import literal_eval
df['golddiff'] = df['golddiff'].apply(literal_eval)
blue = ['bDragons', 'bBarons', 'bHeralds', 'bTowers', 'bInhibs', 'bKills']
red = ['rDragons', 'rBarons', 'rHeralds', 'rTowers', 'rInhibs', 'rKills']
diffs = ['dragondiff', 'barondiff', 'heralddiff', 'towerdiff', 'inhibitordiff', 'killdiff']
for r, b, d in zip(blue, red, diffs):
    df[b] = df[b].apply(literal_eval)
    df[r] = df[r].apply(literal_eval)

    df[b] = df[b].apply(count_item)
    df[r] = df[r].apply(count_item)
    df[d] = df[b] - df[r]


df_raw["bKills"] = df_raw["bKills"].apply(literal_eval)
df_raw["rKills"] = df_raw["rKills"].apply(literal_eval)


bkls = df_raw["bKills"]
btags = df_raw["blueTeamTag"]
bjng = df_raw["blueJungle"]
ganks_wtime = []
for jnglr, res, tag in zip(bjng, bkls, btags):
  tag = str(tag)
  jnglr = str(jnglr)
  arr = np.array(res)
  bool_arr = np.array([(tag + " " + jnglr) in res[i][3] and len(res[i][3]) < 3 for i in range(len(res))])
  if len(bool_arr) == 0:
    ganks_wtime.append(np.array([0]*MAX_TIME_STEP))
    continue
  if len(arr[bool_arr]) == 0:
    ganks_wtime.append(np.array([0]*MAX_TIME_STEP))
    continue
  ganks_wtime.append(arr[bool_arr])

times = []
for gank in ganks_wtime:
  temp = []
  if gank.shape[0] == 30:
    temp.append([0])
    times.append(temp)
    continue
  for k in range(gank.shape[0]):
    temp.append(gank[k][0])
  times.append(temp)

bgank_counts = []
for t in times:
  mins = [int(np.floor(a)) for a in t]
  counts = {a:mins.count(a) for a in range(1, MAX_TIME_STEP+1)}
  total = 0
  new = [0]*MAX_TIME_STEP
  for i in range(MAX_TIME_STEP):
    total += counts[i+1]
    new[i] = total
  bgank_counts.append(new)


rkls = df_raw["rKills"]
rtags = df_raw["redTeamTag"]
rjng = df_raw["redJungle"]
ganks_wtime = []
for jnglr, res, tag in zip(rjng, rkls, rtags):
  tag = str(tag)
  jnglr = str(jnglr)
  arr = np.array(res)
  bool_arr = np.array([(tag + " " + jnglr) in res[i][3] and len(res[i][3]) < 3 for i in range(len(res))])
  if len(bool_arr) == 0:
    ganks_wtime.append(np.array([0]*MAX_TIME_STEP))
    continue
  if len(arr[bool_arr]) == 0:
    ganks_wtime.append(np.array([0]*MAX_TIME_STEP))
    continue
  ganks_wtime.append(arr[bool_arr])

times = []
for gank in ganks_wtime:
  temp = []
  if gank.shape[0] == 30:
    temp.append([0])
    times.append(temp)
    continue
  for k in range(gank.shape[0]):
    temp.append(gank[k][0])
  times.append(temp)

rgank_counts = []
for t in times:
  mins = [int(np.floor(a)) for a in t]
  counts = {a:mins.count(a) for a in range(1, MAX_TIME_STEP+1)}
  total = 0
  new = [0]*MAX_TIME_STEP
  for i in range(MAX_TIME_STEP):
    total += counts[i+1]
    new[i] = total
  rgank_counts.append(new)


df["bGankTime"] = bgank_counts
df["rGankTime"] = rgank_counts
df["gankDiffs"] = df["bGankTime"].apply(np.array) - df["rGankTime"].apply(np.array)


stats = ['golddiff','dragondiff', 'barondiff', 'heralddiff', 'towerdiff', 'inhibitordiff', 'killdiff']
x = df[stats]
y = df['bResult']


arrs = x["golddiff"].apply(np.array)
first_30 = []
for a in arrs:
  first_30.append(a[0:30])

sum_ = [0]*MAX_TIME_STEP
for l in first_30:
  sum_ += l
sum_ = sum_/len(first_30)


stats = ['golddiff','dragondiff', 'barondiff', 'heralddiff', 'towerdiff', 'inhibitordiff', 'killdiff']
x = df[stats]
y = df['bResult']

from sklearn.preprocessing import StandardScaler

data = {}
scalers = {}
for stat in stats:
    scalers[stat] = StandardScaler()
    for row in df[stat]:
        scalers[stat].partial_fit(np.asanyarray(row).reshape(-1, 1))
    data[stat] = [scalers[stat].transform(np.asanyarray(row).reshape(-1, 1)).reshape(-1) for row in df[stat]]

num_features = len(data)
print(f'# of features per timestep: {num_features}')

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch
RANDOM_SEED = 0

class LOLDataset(Dataset):
    def __init__(self, data, stats, label):
        
        self.data =[]
        for t in range(MAX_TIME_STEP):
            self.data.append([[data[stat][i][t] for stat in stats] for i in range(matches)])
        self.label=[i for i in label]
        
    
    def __getitem__(self, item):
        return torch.tensor([ [torch.scalar_tensor(i) for i in x[item]] for x in self.data]), torch.tensor(self.label[item])

    def __len__(self):
        return len(self.label)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_size = 256
        
        self.rnn= nn.RNN(
            nonlinearity = 'relu',
            input_size = num_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.out = nn.Linear(self.hidden_size, 2)
    
    def forward(self,x):
        r_out, hn = self.rnn(x, torch.zeros(1, len(x), self.hidden_size))
        out = self.out(r_out[:, -1, :])
        return out

def train(dataloader, model, loss_fn, optimizer, mute = False):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = Variable(x), Variable(y)

        predict = model(x)
        loss = loss_fn(predict, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0 and not mute:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, validation = False):
    model.eval()
    size = len(dataloader.dataset)
    
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for step,(x,y) in enumerate(dataloader):
            x, y = Variable(x), Variable(y)
            predict = model(x)
            test_loss += loss_fn(predict, y).item()
            correct += (predict.argmax(1) == y).sum().item()
    
    print(f"{'Valid' if validation else 'Test'} Acc:{correct/size:>7f}, Avg Loss: {test_loss/size:>7f}")
    
    return correct/size


BATCH_SIZE = 32

dataset = LOLDataset(data, stats, df["bResult"])
test_size = valid_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size - valid_size

trainDataset, validDataset, testDataset = random_split(
    dataset = dataset,
    lengths = [train_size, valid_size, test_size],
    generator = torch.Generator().manual_seed(0)
)

trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle=True)
validLoader = DataLoader(validDataset, batch_size = BATCH_SIZE)
testLoader = DataLoader(testDataset, batch_size = BATCH_SIZE)

correct = 0
for x, y in dataset:
    if (x[-1][0] > 0) ^ (y == 1) == 0 : correct += 1
print(f'Baseline Accuracy = {correct/matches*100:>.2f}% ')

MUTE = False
EPOCH = 100
LR = 0.0001

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

model = RNN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

best_acc = 0
early_stopping = 0
early_stopping_threshold = 5

for epoch in range(1, EPOCH + 1):
    print(f"--------- Epoch #{epoch} ---------")
    train(trainLoader, model, loss_func, optimizer, mute = MUTE)
    valid_acc = test(validLoader, model, loss_func, validation = True)
    if valid_acc > best_acc :
        early_stopping = 0
        best_acc = valid_acc
        torch.save(model.state_dict(), f"./{MAX_TIME_STEP}.pt")
    else :
        early_stopping += 1
        if early_stopping == early_stopping_threshold :
            print(f"Early stopped at epoch #{epoch} with best validation accuracy {best_acc*100:.2f}get_ipython().run_line_magic(".")", "")
            break

model.load_state_dict(torch.load(f"./{MAX_TIME_STEP}.pt"))
acc_RNN = test(testLoader, model, loss_func)

print(f'Model Accuracy = {acc_RNN*100:>.2f}% ')


stats = ['golddiff','dragondiff', 'barondiff', 'heralddiff', 'towerdiff', 'inhibitordiff', 'killdiff', 'gankDiffs']
x = df[stats]
y = df['bResult']

from sklearn.preprocessing import StandardScaler

data = {}
scalers = {}
for stat in stats:
    scalers[stat] = StandardScaler()
    for row in df[stat]:
        scalers[stat].partial_fit(np.asanyarray(row).reshape(-1, 1))
    data[stat] = [scalers[stat].transform(np.asanyarray(row).reshape(-1, 1)).reshape(-1) for row in df[stat]]

num_features = len(data)
print(f'# of features per timestep: {num_features}')

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch
RANDOM_SEED = 0

class LOLDataset(Dataset):
    def __init__(self, data, stats, label):
        
        self.data =[]
        for t in range(MAX_TIME_STEP):
            self.data.append([[data[stat][i][t] for stat in stats] for i in range(matches)])
        self.label=[i for i in label]
        
    
    def __getitem__(self, item):
        return torch.tensor([ [torch.scalar_tensor(i) for i in x[item]] for x in self.data]), torch.tensor(self.label[item])

    def __len__(self):
        return len(self.label)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_size = 256
        
        self.rnn= nn.RNN(
            nonlinearity = 'relu',
            input_size = num_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.out = nn.Linear(self.hidden_size, 2)
    
    def forward(self,x):
        r_out, hn = self.rnn(x, torch.zeros(1, len(x), self.hidden_size))
        out = self.out(r_out[:, -1, :])
        return out

def train(dataloader, model, loss_fn, optimizer, mute = False):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = Variable(x), Variable(y)

        predict = model(x)
        loss = loss_fn(predict, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0 and not mute:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, validation = False):
    model.eval()
    size = len(dataloader.dataset)
    
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for step,(x,y) in enumerate(dataloader):
            x, y = Variable(x), Variable(y)
            predict = model(x)
            test_loss += loss_fn(predict, y).item()
            correct += (predict.argmax(1) == y).sum().item()
    
    print(f"{'Valid' if validation else 'Test'} Acc:{correct/size:>7f}, Avg Loss: {test_loss/size:>7f}")
    
    return correct/size


BATCH_SIZE = 32

dataset = LOLDataset(data, stats, df["bResult"])
test_size = valid_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size - valid_size

trainDataset, validDataset, testDataset = random_split(
    dataset = dataset,
    lengths = [train_size, valid_size, test_size],
    generator = torch.Generator().manual_seed(0)
)

trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle=True)
validLoader = DataLoader(validDataset, batch_size = BATCH_SIZE)
testLoader = DataLoader(testDataset, batch_size = BATCH_SIZE)

correct = 0
for x, y in dataset:
    if (x[-1][0] > 0) ^ (y == 1) == 0 : correct += 1
print(f'Baseline Accuracy = {correct/matches*100:>.2f}% ')

MUTE = False
EPOCH = 100
LR = 0.0001

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

model = RNN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

best_acc = 0
early_stopping = 0
early_stopping_threshold = 5

for epoch in range(1, EPOCH + 1):
    print(f"--------- Epoch #{epoch} ---------")
    train(trainLoader, model, loss_func, optimizer, mute = MUTE)
    valid_acc = test(validLoader, model, loss_func, validation = True)
    if valid_acc > best_acc :
        early_stopping = 0
        best_acc = valid_acc
        torch.save(model.state_dict(), f"./{MAX_TIME_STEP}.pt")
    else :
        early_stopping += 1
        if early_stopping == early_stopping_threshold :
            print(f"Early stopped at epoch #{epoch} with best validation accuracy {best_acc*100:.2f}get_ipython().run_line_magic(".")", "")
            break

model.load_state_dict(torch.load(f"./{MAX_TIME_STEP}.pt"))
acc_RNN = test(testLoader, model, loss_func)

print(f'Model Accuracy = {acc_RNN*100:>.2f}% ')



