import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        locs = torch.tensor(self.data.iloc[idx]['location_id'])
        times = torch.tensor(self.data.iloc[idx]['timestamp'])
        target = torch.tensor(self.data.iloc[idx]['user_id'])
        print(locs)
        return locs, times, target


class MyDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        # initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['user_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        locs = []
        for x in self.data.iloc[idx]['location_id'].split(','):
            try:
                locs.append(float(x.strip('[]')))
            except ValueError:
                pass
        times = []
        for x in self.data.iloc[idx]['timestamp'].split(','):
            try:
                times.append(float(x.strip()))
            except ValueError:
                pass
        target = torch.tensor(self.label_encoder.transform([self.data.iloc[idx]['user_id']])[0])
        return locs, times, target

    def collate_fn(self, batch):
        locs_batch, times_batch, target_batch = zip(*batch)
        max_len = max([len(locs) for locs in locs_batch + times_batch])
        locs_padded = [locs + [0] * (max_len - len(locs)) for locs in locs_batch]
        times_padded = [times + [0] * (max_len - len(times)) for times in times_batch]
        return torch.tensor(locs_padded), torch.tensor(times_padded), torch.tensor(target_batch)


# class MyDataset(Dataset):
#     def __init__(self, data_file):
#         self.data = pd.read_csv(data_file, dtype={'userid': int, 'location': int, 'timestamp': int})
#
#     def __len__(self):
#         return len(self.data['userid'].unique())
#
#     def __getitem__(self, index):
#         user_id = self.data['userid'].unique()[index]
#         user_data = self.data[self.data['userid'] == user_id]
#         locs = user_data['location'].tolist()
#         times = user_data['timestamp'].tolist()
#         return torch.tensor(locs), torch.tensor(times), torch.tensor(user_id)


# train_data = pd.read_csv('data/train_dataset.csv')
# test_data = pd.read_csv('data/test_dataset.csv')


import torch.nn as nn


# define the model
class MyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# define training function
def train(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    for i, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data.float())
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    return train_loss / len(dataloader.dataset)


# define validation function
def validate(model, dataloader, loss_fn):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data.float())
            val_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return val_loss, accuracy


# train_dataset = MyDataset(train_data)
train_data = MyDataset('data/train_dataset.csv')

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)
sample_batch = next(iter(train_loader))
locs, times, _ = sample_batch
input_size = len(locs[0]) + len(times[0])
print(f"Input size: {input_size}")
# num_users = len(train_data.label_encoder.classes_)
# test_dataset = MyDataset(test_data)
test_data = MyDataset('data/test_dataset.csv')

test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=test_data.collate_fn)


# set up model, loss function, and optimizer
input_size = len(train_data[0][0]) + len(train_data[0][1])
hidden_size = 64
num_classes = len(train_data.label_encoder.classes_)
model = MyModel(input_size, hidden_size, num_classes)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model = MyModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
#
for epoch in range(10):
    train_loss = 0.0
    model.train()
    for locs, times, target in train_loader:
        optimizer.zero_grad()
        output = model(locs, times)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for locs, times, target in test_loader:
            output = model(locs, times)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Epoch %d: Train loss: %.4f | Test accuracy: %.2f%%' %
          (epoch + 1, train_loss / len(train_loader),
           100 * correct / total))
