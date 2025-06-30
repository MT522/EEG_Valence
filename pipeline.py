import numpy as np
import pandas as pd
from scipy import io as sio
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.feature_selection import SelectKBest
import torch
from torch import nn, optim
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

data = sio.loadmat('Project_data.mat')
channels = data['Channels']
test_data = data['TestData']
train_data = data['TrainData']
train_labels = data['TrainLabels'].ravel()
fs = data['fs']

var = np.load('var.npy')
amp_hist = np.load('amp_hist.npy')
ar_model = np.load('ar_model.npy')
cross_corr = np.load('cross_corr.npy')
max_freq = np.load('max_freq.npy')
mean_freq = np.load('mean_freq.npy')
med_freq = np.load('med_freq.npy')
rel_energy = np.load('rel_energy.npy')

var_test = np.load('var_test.npy')
amp_hist_test = np.load('amp_hist_test.npy')
ar_model_test = np.load('ar_model_test.npy')
cross_corr_test = np.load('cross_corr_test.npy')
max_freq_test = np.load('max_freq_test.npy')
mean_freq_test = np.load('mean_freq_test.npy')
med_freq_test = np.load('med_freq_test.npy')
rel_energy_test = np.load('rel_energy_test.npy')

features_tr = np.concatenate(
    (var, amp_hist, ar_model, cross_corr, max_freq, mean_freq, med_freq, rel_energy), axis=1)
features_te = np.concatenate((var_test, amp_hist_test, ar_model_test, cross_corr_test,
                             max_freq_test, mean_freq_test, med_freq_test, rel_energy_test), axis=1)

sc = StandardScaler()

features_tr = sc.fit_transform(features_tr)
features_te = sc.fit_transform(features_te)

np.save('features_train_scaled', features_tr)
np.save('features_test_scaled', features_te)

var_fea = ['variance_' + str(i) for i in range(var.shape[1])]
hist_fea = ['hist_' + str(i) for i in range(amp_hist.shape[1])]
ar_fea = ['ar_' + str(i) for i in range(ar_model.shape[1])]
cross_fea = ['correlation_' + str(i) for i in range(cross_corr.shape[1])]
max_fea = ['max_freq_' + str(i) for i in range(max_freq.shape[1])]
mean_fea = ['mean_freq_' + str(i) for i in range(mean_freq.shape[1])]
med_fea = ['median_freq_' + str(i) for i in range(med_freq.shape[1])]
energ_fea = ['rel_energy_' + str(i) for i in range(rel_energy.shape[1])]

all_fea = var_fea + hist_fea + ar_fea + cross_fea + \
    max_fea + mean_fea + med_fea + energ_fea


def get_feature_name(index):
    return all_fea[index]


def fisher_score(X, y):
    mean = np.mean(X, axis=0)

    sb1 = (np.sum(X[y == 1], axis=0) - mean)**2
    sb2 = (np.sum(X[y == -1], axis=0) - mean)**2

    sw1 = np.var(X[y == 1], axis=0)
    sw2 = np.var(X[y == -1], axis=0)

    return sb1 + sb2 / sw1 + sw2


selector = SelectKBest(score_func=fisher_score, k=50)
X_tr = selector.fit_transform(features_tr, train_labels)

# Get the indices of the selected features
selected_indices = np.where(selector.get_support())[0]
print(f'Selected feature indices: {selected_indices}')
print(
    f'Selected features: {[get_feature_name(index) for index in selected_indices]}')

X_te = features_te[:, selected_indices]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden2_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out


batch_size = 10
learning_rate = 0.01
input_size = 50
hidden1_size = 32
hidden2_size = 16
num_epochs = 10
num_total_steps = 550 // batch_size

model = MLPNet(input_size, hidden1_size, hidden2_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

X_tr_tensor = torch.tensor(np.float32(X_tr))
y_tr_tensor = torch.tensor(np.float32(np.where(train_labels == 1, 1.0, 0.0)))

kfold = KFold(n_splits=5, shuffle=True)

for fold, (train_ids, valid_ids) in enumerate(kfold.split(X_tr_tensor)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    train_loader = DataLoader(TensorDataset(
        X_tr_tensor, y_tr_tensor), batch_size=batch_size, sampler=train_subsampler)
    valid_loader = DataLoader(TensorDataset(
        X_tr_tensor, y_tr_tensor), batch_size=batch_size, sampler=valid_subsampler)

    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, torch.float32)
            targets = targets.to(device, torch.float32)

            outputs = model(inputs)

            loss = criterion(outputs.ravel(), targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 5 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_total_steps}], Loss: {loss.item():.4f}')

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')
    print('--------------------------------')
