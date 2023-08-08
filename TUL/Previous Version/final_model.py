import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import numpy as np
import ast
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


# class MyDataset(Dataset):
#     def __init__(self, data_file):
#         self.data = pd.read_csv(data_file, converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#         # self.label_encoder = LabelEncoder()
#         # self.label_encoder.fit(self.data['user_id'])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         locs = []
#         for x in self.data.iloc[idx]['location_id'].split(','):
#             try:
#                 locs.append(float(x.strip('[]')))
#             except ValueError:
#                 pass
#         times = []
#         for x in self.data.iloc[idx]['timestamp'].split(','):
#             try:
#                 times.append(float(x.strip()))
#             except ValueError:
#                 pass
#         target = torch.tensor([self.data.iloc[idx]['user_id']][0])
#         return locs, times, target
#
#     def collate_fn(self, batch):
#         locs_batch, times_batch, target_batch = zip(*batch)
#         max_len = max([len(locs) for locs in locs_batch + times_batch])
#         locs_padded = [locs + [0] * (max_len - len(locs)) for locs in locs_batch]
#         times_padded = [times + [0] * (max_len - len(times)) for times in times_batch]
#         return torch.tensor(locs_padded), torch.tensor(times_padded), torch.tensor(target_batch)


def parse_list_col(value):
    return ast.literal_eval(value)


#
# class LocationTimestampDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.locs = self.data['location_id'].apply(eval).tolist()
#         self.times = self.data['timestamp'].apply(eval).tolist()
#         self.targets = self.data['user_id'].tolist()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.locs[index], self.times[index], self.targets[index]
#
#
#     def collate_fn(self, batch):
#         locs, times, targets = zip(*batch)
#         locs_lengths = torch.tensor([len(x) for x in locs], dtype=torch.int64)
#         times_lengths = torch.tensor([len(x) for x in times], dtype=torch.int64)
#         locs = [torch.tensor(x, dtype=torch.int64) for x in locs]
#         times = [torch.tensor(x, dtype=torch.int64) for x in times]
#         locs_padded = torch.nn.utils.rnn.pad_sequence(locs, batch_first=True)
#         times_padded = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)
#         targets = torch.tensor(targets, dtype=torch.int64)
#         return locs_padded, times_padded, locs_lengths, times_lengths, targets

# class LocationTimestampDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.locs = self.data['location_id'].apply(eval).tolist()
#         self.times = self.data['timestamp'].apply(eval).tolist()
#         self.targets = self.data['user_id'].astype(int).tolist()
#         self.num_classes = len(set(self.targets))
#
#         # Create location to index dictionary
#         all_locs = [loc for seq in self.locs for loc in seq]
#         self.loc_to_ix = {'<pad>': 0, '<unk>': 1}
#         for loc in all_locs:
#             if loc not in self.loc_to_ix:
#                 self.loc_to_ix[loc] = len(self.loc_to_ix)
#         all_times=[time for seq in self.times for time in seq]
#         self.time_to_ix={'<pad>': 0, '<unk>': 1}
#         for time in all_times:
#             if time not in self.time_to_ix:
#                 self.time_to_ix[time]=len(self.time_to_ix)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.locs[index], self.times[index], self.targets[index]
#
#     def num_locations(self):
#         loc_set = set(tuple(loc) for loc in self.locs)
#         return len(loc_set)
#
#     def collate_fn(self, batch):
#         # Sort the batch by sequence length (in descending order) to use PackedSequence
#         batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
#         # Separate inputs and targets
#         locs, times, targets = zip(*batch)
#         # Convert lists to tensors and pad sequences
#         locs = nn.utils.rnn.pad_sequence([torch.tensor([self.loc_to_ix.get(l, self.loc_to_ix['<unk>']) for l in seq], dtype=torch.int64) for seq in locs], batch_first=True, padding_value=self.loc_to_ix['<pad>'])
#         times = nn.utils.rnn.pad_sequence([torch.tensor([self.time_to_ix.get(l,self.time_to_ix['<unk>']) for l in seq],dtype=torch.int64) for seq in times],batch_first=True,padding_value=self.time_to_ix['<pad>'])
#         # times = nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in times], batch_first=True, padding_value=0)
#         targets = torch.tensor(targets, dtype=torch.float32)
#         return locs, times, targets


class LocationTimestampDataset(Dataset):
    def __init__(self, csv_file, oversample=False):
        self.data = pd.read_csv(csv_file)
        print(self.data.columns)
        print(self.data.head(5))
        self.locs = self.data['location_id'].apply(eval).tolist()
        self.times = self.data['timestamp'].apply(eval).tolist()
        self.targets = self.data['user_id'].astype(int).tolist()
        self.num_classes = len(set(self.targets))

        # Create location to index dictionary
        all_locs = [loc for seq in self.locs for loc in seq]
        self.loc_to_ix = {'<pad>': 0, '<unk>': 1}
        for loc in all_locs:
            if loc not in self.loc_to_ix:
                self.loc_to_ix[loc] = len(self.loc_to_ix)
        all_times = [time for seq in self.times for time in seq]
        self.time_to_ix = {'<pad>': 0, '<unk>': 1}
        for time in all_times:
            if time not in self.time_to_ix:
                self.time_to_ix[time] = len(self.time_to_ix)
        if oversample:
            # Find the count of each target label
            target_counts = {}
            for target in self.targets:
                if target in target_counts:
                    target_counts[target] += 1
                else:
                    target_counts[target] = 1

            # Determine the maximum number of instances of a target label
            max_count = max(target_counts.values())

            # Oversample the data for each target label
            new_locs = []
            new_times = []
            new_targets = []
            for target, count in target_counts.items():
                locs = [seq for i, seq in enumerate(self.locs) if self.targets[i] == target]
                times = [seq for i, seq in enumerate(self.times) if self.targets[i] == target]
                indices = [i for i in range(count)]
                while count < max_count:
                    i = indices[count % len(indices)]
                    new_locs.append(locs[i])
                    new_times.append(times[i])
                    new_targets.append(target)
                    count += 1
            self.locs.extend(new_locs)
            self.times.extend(new_times)
            self.targets.extend(new_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.locs[index], self.times[index], self.targets[index]

    def num_locations(self):
        loc_set = set(tuple(loc) for loc in self.locs)
        return len(loc_set)

    def collate_fn(self, batch):
        # Sort the batch by sequence length (in descending order) to use PackedSequence
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        # Separate inputs and targets
        locs, times, targets = zip(*batch)
        # Calculate class frequencies
        class_freqs = np.bincount(targets)
        max_freq = class_freqs.max()

        # Oversample locs and times
        locs_c = list(locs)
        times_c = list(times)
        targets_c = list(targets)
        count = Counter(targets_c)
        max_count = max(count.values())
        locs_oversampled = []
        times_oversampled = []
        targets_oversampled = []
        for label, freq in count.items():
            locs_c_i = [locs[i] for i in range(len(locs)) if targets[i] == label]
            times_c_i = [times[i] for i in range(len(times)) if targets[i] == label]
            targets_c_i = [targets[i] for i in range(len(targets)) if targets[i] == label]
            if freq < max_count:
                num_samples = max_count - freq
                locs_oversampled.extend([locs_c_i[i % len(locs_c_i)] for i in range(num_samples)])
                times_oversampled.extend([times_c_i[i % len(times_c_i)] for i in range(num_samples)])
                targets_oversampled.extend([label] * num_samples)
        locs_c.extend(locs_oversampled)
        times_c.extend(times_oversampled)
        targets_c.extend(targets_oversampled)
        # Convert lists to tensors and pad sequences
        locs = nn.utils.rnn.pad_sequence(
            [torch.tensor([self.loc_to_ix.get(l, self.loc_to_ix['<unk>']) for l in seq], dtype=torch.int64) for seq in
             locs_c], batch_first=True, padding_value=self.loc_to_ix['<pad>'])
        times = nn.utils.rnn.pad_sequence(
            [torch.tensor([self.time_to_ix.get(l, self.time_to_ix['<unk>']) for l in seq], dtype=torch.int64) for seq in
             times_c], batch_first=True, padding_value=self.time_to_ix['<pad>'])
        # times = nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in times], batch_first=True, padding_value=0)
        targets = torch.tensor(targets_c, dtype=torch.float32)
        return locs, times, targets

    # def collate_fn(self,batch):
    #     locs = [torch.tensor(item[0]) for item in batch]
    #     times = [torch.tensor(item[1]) for item in batch]
    #     targets = [item[2] for item in batch]
    #
    #     # get sequence lengths
    #     locs_lengths = [len(seq) for seq in locs]
    #     times_lengths = [len(seq) for seq in times]
    #
    #     # pad sequences
    #     locs = torch.nn.utils.rnn.pad_sequence(locs, batch_first=True, padding_value=0)
    #     times = torch.nn.utils.rnn.pad_sequence(times, batch_first=True, padding_value=0)
    #     targets = torch.tensor(targets)
    #     return locs, times,targets, locs_lengths, times_lengths

    # def collate_fn(self, batch):
    #     locs, times, targets = zip(*batch)
    #     locs_lengths = torch.tensor([len(x) for x in locs], dtype=torch.int64)
    #     times_lengths = torch.tensor([len(x) for x in times], dtype=torch.int64)
    #     locs = [torch.tensor(x, dtype=torch.int64) for x in locs]
    #     times = [torch.tensor(x, dtype=torch.int64) for x in times]
    #     locs_padded = torch.nn.utils.rnn.pad_sequence(locs, batch_first=True)
    #     times_padded = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)
    #     targets = torch.tensor(targets, dtype=torch.int64)
    #     return locs_padded, times_padded, locs_lengths, times_lengths, targets


# class MyModel(nn.Module):
#     def __init__(self, num_locations, num_times, num_users, device):
#         super(MyModel, self).__init__()
#         self.device = device
#         self.embeddings_loc = nn.Embedding(num_locations, 16)
#         self.embeddings_times = nn.Embedding(num_times, 16)
#         self.lstm = nn.LSTM(16, 32, num_layers=1, batch_first=True)
#         self.fc1 = nn.Linear(32, 64)
#         self.fc2 = nn.Linear(64, num_users)
#
#     def forward(self, locs, times, locs_lengths, times_lengths):
#         locs = torch.tensor(locs, dtype=torch.int64, device=self.device)
#         times = torch.tensor(times, dtype=torch.int64, device=self.device)
#         loc_embedded = self.embeddings_loc(locs)
#         times_embedded = self.embeddings_times(times)
#
#         # pack sequences
#         loc_packed = nn.utils.rnn.pack_padded_sequence(loc_embedded, locs_lengths, batch_first=True,
#                                                        enforce_sorted=False)
#         times_packed = nn.utils.rnn.pack_padded_sequence(times_embedded, times_lengths, batch_first=True,
#                                                          enforce_sorted=False)
#
#         # pass through LSTM layer
#         lstm_input = torch.cat((loc_packed.unsqueeze(0), times_packed.unsqueeze(0)), dim=2)
#         lstm_output, _ = self.lstm(lstm_input)
#
#         # extract final hidden state of LSTM layer
#         x = lstm_output[:, -1, :]
#
#         # pass through fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         return x

# class MyModel(nn.Module):
#     def __init__(self, num_locations, num_times, num_users, device):
#         super(MyModel, self).__init__()
#         self.device = device
#         self.embeddings_loc = nn.Embedding(num_locations, 16)
#         self.embeddings_times = nn.Embedding(num_times, 16)
#         self.lstm = nn.LSTM(32, 32, num_layers=1, batch_first=True)
#         self.fc1 = nn.Linear(32, 64)
#         self.fc2 = nn.Linear(64, num_users)
#
#     def forward(self, locs, times):
#         # Embed the locations and times
#         embedded_locs = self.embeddings_loc(locs)
#         embedded_times = self.embeddings_times(times.long())
#
#         # Concatenate the embeddings and pass through the LSTM layer
#         embedded = torch.cat([embedded_locs, embedded_times], dim=2)
#         output, (hidden, cell) = self.lstm(embedded)
#
#         # Pass the final hidden state through the linear layer to get the logits
#         logits = self.fc1(hidden[-1])
#
#         return logits

# class MyModel(nn.Module):
#     def __init__(self, num_locations, num_times, num_users, device):
#         super(MyModel, self).__init__()
#         self.device = device
#         self.embeddings_loc = nn.Embedding(num_locations, 32)
#         self.embeddings_times = nn.Embedding(num_times, 32)
#         self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2)
#         self.fc1 = nn.Linear(128, 256)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(256, num_users)
#
#     def forward(self, locs, times):
#         # Embed the locations and times
#         embedded_locs = self.embeddings_loc(locs)
#         embedded_times = self.embeddings_times(times.long())
#
#         # Concatenate the embeddings and pass through the LSTM layer
#         embedded = torch.cat([embedded_locs, embedded_times], dim=2)
#         output, (hidden, cell) = self.lstm(embedded)
#
#         # Flatten the output and pass through the linear layers
#         x = self.dropout(hidden[-1])
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         logits = self.fc2(x)
#
#         return logits


# class MyModel(nn.Module):
#     def __init__(self, num_locations, num_times, num_users, device):
#         super(MyModel, self).__init__()
#         self.device = device
#         self.embeddings_loc = nn.Embedding(num_locations, 64)
#         self.embeddings_times = nn.Embedding(num_times, 64)
#         self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#         self.conv1d = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
#         self.max_pool1d = nn.MaxPool1d(kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(512, 256)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(256, num_users)
#
#     def forward(self, locs, times):
#         # Embed the locations and times
#         embedded_locs = self.embeddings_loc(locs)
#         embedded_times = self.embeddings_times(times.long())
#
#         # Concatenate the embeddings and pass through the LSTM layer
#         embedded = torch.cat([embedded_locs, embedded_times], dim=2)
#         output, (hidden, cell) = self.lstm(embedded)
#
#         # Use the hidden states from the LSTM to perform 1D convolution and max pooling
#         hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(2)
#         conv_output = self.conv1d(hidden)
#         pool_output = self.max_pool1d(conv_output)
#
#         # Flatten the output and pass through the linear layers
#         x = pool_output.view(pool_output.size(0), -1)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         logits = self.fc2(x)
#
#         return logits

# Next try this model
class MyModel(nn.Module):
    def __init__(self, num_locations, num_times, num_users, device):
        super(MyModel, self).__init__()
        self.device = device
        self.embedding_dim=128
        self.embeddings_loc = nn.Embedding(num_locations, self.embedding_dim)
        self.embeddings_times = nn.Embedding(num_times, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim*2, 256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.max_pool1d = nn.MaxPool1d(kernel_size=3, padding=1)
        self.attn = nn.Linear(512, 512)
        self.fc1 = nn.Linear(592, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_users)
        self.hidden_size = 512

    def forward(self, locs, times):
        # Embed the locations and times
        embedded_locs = self.embeddings_loc(locs)
        embedded_times = self.embeddings_times(times.long())

        # Concatenate the embeddings and pass through the LSTM layer
        embedded = torch.cat([embedded_locs, embedded_times], dim=2)
        output, (hidden, cell) = self.lstm(embedded)

        # Use the hidden states from the LSTM to perform 1D convolution and max pooling
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(2)
        conv_output = self.conv1d(hidden)
        pool_output = self.max_pool1d(conv_output)
        pool_output = pool_output.view(pool_output.size(0), -1)

        # Use a linear layer and softmax to compute the attention weights
        # attn_weights = F.softmax(self.attn(pool_output.view(pool_output.size(0), 1, -1)), dim=2)
        attn_weights = F.softmax(self.attn(pool_output.view(pool_output.size(0), 1, -1)), dim=2).view(
            pool_output.size(0), 1, -1).transpose(1, 2)

        # Reshape the attention weights tensor to perform batch matrix multiplication with output
        # attn_weights = attn_weights.unsqueeze(2)
        #
        # # Compute the weighted sum of the hidden states using the attention weights
        # attn_output = torch.bmm(output.transpose(1, 2), attn_weights).squeeze()

        # attn_weights = attn_weights.unsqueeze(2)
        attn_output = torch.bmm(output, attn_weights).squeeze()

        # Reduce the number of channels in pool_output to match the hidden_size of attn_output
        conv_output = self.conv1d(pool_output.unsqueeze(2))
        conv_output = self.max_pool1d(conv_output).squeeze()

        # Flatten the output and pass through the linear layers
        x = torch.cat([conv_output, attn_output], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


# Define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # print(batch)
        # locs, times, targets,locs_lengths,times_lengths = batch[0], batch[1], batch[2], batch[3],batch[4]
        locs, times, targets = batch[0], batch[1], batch[2]
        # print(locs.shape)
        # print(times.shape)
        locs = locs.to(device)
        times = times.to(device)
        targets = torch.tensor(targets, dtype=torch.long)
        targets = targets.to(device)
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
            targets = torch.tensor(targets, dtype=torch.long)
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
    batch_size = 16
    # set device

    # Define dataset and dataloader
    train_data = LocationTimestampDataset('data/merged_final_train.csv', True)
    test_data = LocationTimestampDataset('data/merged_final_test.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_data.collate_fn)

    # Calculate num_times, num_locations, num_users
    num_times = len(set([t for d in train_data.data['timestamp'].str.split(',') for t in d]))
    num_locations = len(set([l for d in train_data.data['location_id'].str.split(',') for l in d]))
    num_classes = train_data.num_classes
    # num_locations = train_data.num_locations()
    print(num_classes)
    print(num_locations)

    model = MyModel(num_locations=num_locations, num_times=num_times, num_users=num_classes, device=device)
    model.to(device)
    print(model)
    # create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train model
    num_epochs = 5
    for epoch in range(num_epochs):
        # train model for one epoch
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # evaluate model on validation set
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # print epoch statistics
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {(val_acc):.4f}')

    # save model
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
