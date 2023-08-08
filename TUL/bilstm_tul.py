import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn


# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         locs = torch.tensor(self.data.iloc[idx]['location_id'])
#         times = torch.tensor(self.data.iloc[idx]['timestamp'])
#         target = torch.tensor(self.data.iloc[idx]['user_id'])
#         print(locs)
#         return locs, times, target


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
        target = self.label_encoder.transform([self.data.iloc[idx]['user_id']])[0]
        return locs, times, target

    def collate_fn(self, batch):
        locs_batch, times_batch, target_batch = zip(*batch)
        max_len = max([len(locs) for locs in locs_batch + times_batch])
        max_len = 899
        locs_padded = [locs + [0] * (max_len - len(locs)) for locs in locs_batch]
        times_padded = [times + [0] * (max_len - len(times)) for times in times_batch]

        # Convert locs_padded and times_padded to tensors
        locs_tensor = torch.tensor(locs_padded)
        times_tensor = torch.tensor(times_padded)

        # Reshape locs_tensor and times_tensor to match the input size
        locs_tensor = locs_tensor.view(locs_tensor.size(0), locs_tensor.size(1), -1)
        times_tensor = times_tensor.view(times_tensor.size(0), times_tensor.size(1), -1)

        return locs_tensor, times_tensor, torch.tensor(target_batch)


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

# train_dataset = MyDataset(train_data)
# train_dataset = MyDataset('data/train_dataset.csv')
# Uncomment
train_dataset = MyDataset('data/StaticMap/merged_final_train.csv')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
sample_batch = next(iter(train_loader))
locs, times, _ = sample_batch
input_size = len(locs[0]) + len(times[0])
print(f"Input size: {input_size}")

# test_dataset = MyDataset(test_data)
test_dataset = MyDataset('data/StaticMap/merged_final_test.csv')

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)

privacy_dataset = MyDataset('data/k_anonymity/merged_final_privacy.csv')

privacy_loader = DataLoader(privacy_dataset, batch_size=32, shuffle=False, collate_fn=privacy_dataset.collate_fn)

synthetic_dataset = MyDataset('data/k_anonymity/merged_final_evaluation.csv')
synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=False, collate_fn=synthetic_dataset.collate_fn)


#
# for locs, times, target in eval_loader:
#     print(locs)
#     print(times)
#     print(target)


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()

        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, locs, times):
        x = torch.cat((locs.unsqueeze(2), times.unsqueeze(2)), dim=2)
        lstm_output, _ = self.bilstm(x)
        lstm_output = lstm_output[:, -1, :]
        output = self.fc(lstm_output)
        return output


# set up model, loss function, and optimizer
# input_size = len(train_dataset[0][0]) + len(train_dataset[0][1])
hidden_size = 64
num_classes = len(train_dataset.label_encoder.classes_)
model = MyModel(input_size,hidden_size, num_classes)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for locs, times, target in data_loader:
            output = model(locs, times)
            _, predicted_top5 = torch.topk(output, k=3, dim=1)  # Get top 5 predictions
            correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
            correct_top5 = correct_predictions.any(dim=1).sum().item()
            total += target.size(0)
            correct += correct_top5

    accuracy = 100 * correct / total
    return accuracy


def Synthetic(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for locs, times, target in data_loader:
            output = model(locs, times)
            _, predicted_top5 = torch.topk(output, k=5, dim=1)  # Get top 5 predictions
            correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
            correct_top5 = correct_predictions.any(dim=1).sum().item()
            total += target.size(0)
            correct += correct_top5

    accuracy = 100 * correct / total
    return accuracy


best_accuracy = 0.0  # Track the best accuracy achieved during training

for epoch in range(30):
    train_loss = 0.0
    model.train()
    for locs, times, target in train_loader:
        # print(locs)
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
            # print(locs)
            output = model(locs, times)
            # _, predicted = torch.max(output.data, 1)
            # total += target.size(0)
            # correct += (predicted == target).sum().item()
            _, predicted_top5 = torch.topk(output, k=5, dim=1)  # Get top 5 predictions
            correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
            correct_top5 = correct_predictions.any(dim=1).sum().item()
            total += target.size(0)
            correct += correct_top5

    accuracy = (100 * correct / total)
    print('Epoch %d: Train loss: %.4f | Test accuracy: %.2f%%' %
          (epoch + 1, train_loss / len(train_loader),
           accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'savedModel/BestModelfor2ndalgorithm_6_jun/simple_best_model.pth')  # Save the model with the best accuracy

# Load the best model
model.load_state_dict(torch.load('savedModel/BestModelfor2ndalgorithm_6_jun/simple_best_model.pth'))

# Test the best model
print("##############################################")
average_synthetic_accuracy=0.0
average_privacy_accuracy=0.0

for i in range(5):
    Synthetic_accuracy = Synthetic(model, synthetic_loader)
    average_synthetic_accuracy=average_synthetic_accuracy+Synthetic_accuracy
Synthetic_accuracy= float(average_synthetic_accuracy)/float(5.0)
print('Synthetic Trajectories accuracy: %.2f%%' % Synthetic_accuracy)
for i in range(5):
    privacy_accuracy = test(model, privacy_loader)
    average_privacy_accuracy=average_privacy_accuracy+privacy_accuracy
privacy_accuracy=float(average_privacy_accuracy)/float(5.0)
print('Privacy Preservation accuracy: %.2f%%' % privacy_accuracy)
print("##############################################")





















































# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# import torch.nn as nn
#
#
# # class MyDataset(Dataset):
# #     def __init__(self, data):
# #         self.data = data
# #
# #     def __len__(self):
# #         return len(self.data)
# #
# #     def __getitem__(self, idx):
# #         locs = torch.tensor(self.data.iloc[idx]['location_id'])
# #         times = torch.tensor(self.data.iloc[idx]['timestamp'])
# #         target = torch.tensor(self.data.iloc[idx]['user_id'])
# #         print(locs)
# #         return locs, times, target
#
#
# class MyDataset(Dataset):
#     def __init__(self, data_file):
#         self.data = pd.read_csv(data_file)
#         self.label_encoder = LabelEncoder()
#         self.label_encoder.fit(self.data['user_id'])
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
#         target = self.label_encoder.transform([self.data.iloc[idx]['user_id']])[0]
#         return locs, times, target
#
#     def collate_fn(self, batch):
#         locs_batch, times_batch, target_batch = zip(*batch)
#         max_len = max(len(locs) for locs in locs_batch)
#         locs_padded = [locs + [0] * (max_len - len(locs)) for locs in locs_batch]
#         times_padded = [times + [0] * (max_len - len(times)) for times in times_batch]
#         target_padded = [torch.tensor([target]) for target in target_batch]
#         target_padded = torch.stack(target_padded).expand(-1, max_len)
#         print("collate fn locs length:")
#         print(max_len)
#         print(len(locs_padded))
#         return torch.tensor(locs_padded), torch.tensor(times_padded), target_padded
#
#
# # class MyDataset(Dataset):
# #     def __init__(self, data_file):
# #         self.data = pd.read_csv(data_file, dtype={'userid': int, 'location': int, 'timestamp': int})
# #
# #     def __len__(self):
# #         return len(self.data['userid'].unique())
# #
# #     def __getitem__(self, index):
# #         user_id = self.data['userid'].unique()[index]
# #         user_data = self.data[self.data['userid'] == user_id]
# #         locs = user_data['location'].tolist()
# #         times = user_data['timestamp'].tolist()
# #         return torch.tensor(locs), torch.tensor(times), torch.tensor(user_id)
#
#
# # train_data = pd.read_csv('data/train_dataset.csv')


























# # test_data = pd.read_csv('data/test_dataset.csv')
#
# # train_dataset = MyDataset(train_data)
# # train_dataset = MyDataset('data/train_dataset.csv')
# # Uncomment
# train_dataset = MyDataset('data/StaticMap/merged_final_train.csv')
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
# sample_batch = next(iter(train_loader))
# locs, times, _ = sample_batch
# hidden_size = 64
# # input_size = len(locs[0]) + len(times[0])+1
# input_size = len(locs[0]) + len(times[0]) + 1
# num_embeddings = max(locs.max().item(), times.max().item()) + 1  # Calculate num_embeddings
# print(f"Input size: {input_size}")
# print(f"num_embeddings: {num_embeddings}")
# # input_size=2+hidden_size
# print(f"Input size: {input_size}")
#
# # test_dataset = MyDataset(test_data)
# test_dataset = MyDataset('data/StaticMap/merged_final_test.csv')
#
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)
#
# privacy_dataset = MyDataset('data/k_anonymity/merged_final_privacy.csv')
#
# privacy_loader = DataLoader(privacy_dataset, batch_size=32, shuffle=False, collate_fn=privacy_dataset.collate_fn)
#
# synthetic_dataset = MyDataset('data/k_anonymity/merged_final_evaluation.csv')
# synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=False, collate_fn=synthetic_dataset.collate_fn)
#
#
# #
# # for locs, times, target in eval_loader:
# #     print(locs)
# #     print(times)
# #     print(target)
#
#
# class MyModel(nn.Module):
#     def __init__(self, input_size,hidden_size, num_classes):
#         print("Inpput size in side model " + str(input_size))
#         super(MyModel, self).__init__()
#         print("hidden size "+str(hidden_size))
#         self.embedding = nn.Embedding(int(input_size), hidden_size)
#         self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
#         self.fc1 = nn.Linear(2176, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)
#
#     def forward(self, locs, times):
#         locs = locs.long()
#         times = times.long()
#
#         locs_embedded = self.embedding(locs)
#
#         max_index = times.max()
#         if max_index >= self.embedding.num_embeddings:
#             max_index = self.embedding.num_embeddings - 1
#         times_embedded = self.embedding(times.clamp_max(max_index)).transpose(0, 1)
#
#         _, (hidden, _) = self.bilstm(locs_embedded)
#         hidden = hidden.permute(1, 0, 2).reshape(hidden.shape[1], -1)
#         times_embedded = times_embedded.reshape(times_embedded.shape[0], -1)
#
#         x = torch.cat((hidden, times_embedded), dim=1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         x = nn.ReLU()(x)
#         x = self.fc4(x)
#         x = nn.ReLU()(x)
#         x = self.fc3(x)
#         return x
#
#
# # set up model, loss function, and optimizer
# # input_size = len(train_dataset[0][0]) + len(train_dataset[0][1])
#
# num_classes = len(train_dataset.label_encoder.classes_)
# model = MyModel(num_embeddings,hidden_size, num_classes)
# # print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# criterion = nn.CrossEntropyLoss()
#
#
# def test(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for locs, times, target in data_loader:
#             output = model(locs, times)
#             _, predicted_top5 = torch.topk(output, k=4, dim=1)  # Get top 5 predictions
#             correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
#             correct_top5 = correct_predictions.any(dim=1).sum().item()
#             total += target.size(0)
#             correct += correct_top5
#
#     accuracy = 100 * correct / total
#     return accuracy
#
#
# def Synthetic(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for locs, times, target in data_loader:
#             output = model(locs, times)
#             _, predicted_top5 = torch.topk(output, k=5, dim=1)  # Get top 5 predictions
#             correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
#             correct_top5 = correct_predictions.any(dim=1).sum().item()
#             total += target.size(0)
#             correct += correct_top5
#
#     accuracy = 100 * correct / total
#     return accuracy
#
#
# best_accuracy = 0.0  # Track the best accuracy achieved during training
#
# for epoch in range(30):
#     train_loss = 0.0
#     model.train()
#     for locs, times, target in train_loader:
#         # print(locs)
#         print("Inside training length of locs:")
#         print(len(locs))
#         optimizer.zero_grad()
#         output = model(locs, times)
#         # Modify the target tensor dimensions before repeating
#         # target = target.reshape(-1, 1).repeat(1, output.size(1))
#
#         # Rest of the code...
#
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for locs, times, target in test_loader:
#             # print(locs)
#             output = model(locs, times)
#             # _, predicted = torch.max(output.data, 1)
#             # total += target.size(0)
#             # correct += (predicted == target).sum().item()
#             _, predicted_top5 = torch.topk(output, k=5, dim=1)  # Get top 5 predictions
#             correct_predictions = predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5))
#             correct_top5 = correct_predictions.any(dim=1).sum().item()
#             total += target.size(0)
#             correct += correct_top5
#
#     accuracy = (100 * correct / total)
#     print('Epoch %d: Train loss: %.4f | Test accuracy: %.2f%%' %
#           (epoch + 1, train_loss / len(train_loader),
#            accuracy))
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), 'savedModel/simple_best_model.pth')  # Save the model with the best accuracy
#
# # Load the best model
# model.load_state_dict(torch.load('savedModel/simple_best_model.pth'))
#
# # Test the best model
# print("##############################################")
# average_synthetic_accuracy = 0.0
# average_privacy_accuracy = 0.0
#
# for i in range(5):
#     Synthetic_accuracy = Synthetic(model, synthetic_loader)
#     average_synthetic_accuracy = average_synthetic_accuracy + Synthetic_accuracy
# Synthetic_accuracy = float(average_synthetic_accuracy) / float(5.0)
# print('Synthetic Trajectories accuracy: %.2f%%' % Synthetic_accuracy)
# for i in range(5):
#     privacy_accuracy = test(model, privacy_loader)
#     average_privacy_accuracy = average_privacy_accuracy + privacy_accuracy
# privacy_accuracy = float(average_privacy_accuracy) / float(5.0)
# print('Privacy Preservation accuracy: %.2f%%' % privacy_accuracy)
# print("##############################################")
