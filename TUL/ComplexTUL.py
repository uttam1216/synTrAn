import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


class MyDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file,converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
        # self.label_encoder = LabelEncoder()
        # self.label_encoder.fit(self.data['user_id'])

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
        target = torch.tensor([self.data.iloc[idx]['user_id']][0])
        return locs, times, target

    def collate_fn(self, batch):
        locs_batch, times_batch, target_batch = zip(*batch)
        max_len = max([len(locs) for locs in locs_batch + times_batch])
        locs_padded = [locs + [0] * (max_len - len(locs)) for locs in locs_batch]
        times_padded = [times + [0] * (max_len - len(times)) for times in times_batch]
        return torch.tensor(locs_padded), torch.tensor(times_padded), torch.tensor(target_batch)


def parse_list_col(value):
    return ast.literal_eval(value)


# Define the model
class MyModel(nn.Module):
    def __init__(self, num_locations, num_times, num_users):
        super(MyModel, self).__init__()
        self.embeddings_loc = nn.Embedding(num_locations, 16)
        self.embeddings_times = nn.Embedding(num_times, 16)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, num_users)

    def forward(self, locs, times):
        locs = torch.tensor(locs, dtype=torch.int64, device=device)
        times = torch.tensor(times, dtype=torch.int64, device=device)
        loc_embedded = self.embeddings_loc(locs)
        times_embedded = self.embeddings_times(times)
        x = torch.cat((loc_embedded, times_embedded), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (locs, times, targets) in enumerate(train_loader):
        locs = locs.to(device)
        times = times.to(device)
        targets = targets.to(torch.int64).to(device)
        optimizer.zero_grad()
        outputs = model(locs, times)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss


# Define the evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (locs, times, targets) in enumerate(val_loader):
            locs = locs.to(device)
            times = times.to(device)
            targets = targets.to(device)
            outputs = model(locs, times)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc


def main():
    batch_size = 32
    # set device

    # Define dataset and dataloader
    train_data = MyDataset('data/updated_train.csv')
    test_data = MyDataset('data/updated_test.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_data.collate_fn)

    # Calculate num_times, num_locations, num_users
    num_times = len(set([t for d in train_data.data['timestamp'].str.split(',') for t in d]))
    num_locations = len(set([l for d in train_data.data['location_id'].str.split(',') for l in d]))
    num_classes = len(np.unique(train_data['user_id']))

    # # create model
    # model = MyModel(num_locations=num_locations, num_times=num_times, num_users=num_users)
    # model.to(device)
    #
    # # create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    # create model
    model = MyModel(num_locations=num_locations, num_times=num_times, num_users=num_users)
    model.to(device)
    # create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train model
    num_epochs = 10
    for epoch in range(num_epochs):
        # train model for one epoch
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # evaluate model on validation set
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # print epoch statistics
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # save model
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
