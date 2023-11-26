import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

class LSTMSequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len_src=56, seq_len_tgt=8, lag_len=8):
        self.seq_len_src = seq_len_src
        self.lag_len = lag_len
        self.seq_len_tgt = seq_len_tgt
        self.df = torch.tensor(dataframe.values).float()
        self.len = math.floor((len(self.df) - self.seq_len_src - self.seq_len_tgt) / self.lag_len) + 1

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if type(i) == slice:
            raise Exception('Exception: Expected int, inputted slice!')

        X = self.df[i*self.lag_len:i*self.lag_len+self.seq_len_src]
        Y = self.df[i*self.lag_len+self.seq_len_src:i*self.lag_len+self.seq_len_src+self.seq_len_tgt]

        return X, Y

    def input_size(self):
        return self.df.shape[1]

# lstm - cat - fc -
class LSTM(nn.Module):
    def __init__(self, num_features_pred, num_features, hidden_units, num_layers, device):
        super().__init__()
        self.num_features_pred = num_features_pred  # this is the number of features
        self.num_features= num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=self.num_features_pred,
                            hidden_size=self.hidden_units,
                            batch_first=True,
                            num_layers=self.num_layers,
                            dropout = 0.3)
        self.fc1 = nn.Linear(in_features=self.hidden_units, out_features=self.num_features_pred)
        # self.fc2 = nn.Linear(in_features=128, out_features=self.num_features_pred)

        self.fc3 = nn.Linear(in_features=self.num_features, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=self.num_features_pred)

        self.relu = nn.ReLU()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(self, src):
        batch_size = src.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=self.device).requires_grad_()

        lstm_input = src

        # input only population into a lstm layer
        for t in range(lstm_input.shape[1]):
            output, (h0, c0) = self.lstm(lstm_input[:, :, :self.num_features_pred], (h0, c0))  # output size = (batch, sequence_length, hidden_size)
            output = self.fc1(output[:, -1:, :])
            output = self.relu(output)  # output size = (batch, 1, num_features_pred)

            # outputの一期分を付け加えて、最初の1期を消す
            output = torch.cat([lstm_input[:, 1:, :self.num_features_pred], output], dim=1)

            # concat output from lstm and dummies
            lstm_input = torch.cat((output[:, :, :self.num_features_pred], src[:, :, self.num_features_pred:]), dim=2)

        # output = self.fc2(output)
        # output = self.relu(output)

        # fc layers
        x = self.fc3(lstm_input)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)

        return x  # x.shape = [batch_size, 1(seq_len), num_features_pred]
    
def lstm_train(model, data_loader):
    total_losses = []
    model.train()
    for src, tgt in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)

        output = model(src)
        loss = model.criterion(output[:,:,0:model.num_features_pred], tgt[:,:,0:model.num_features_pred])

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_losses.append(loss.item())

    return np.average(total_losses)

# def test_model(data_loader, model, loss_function):
#     num_batches = len(data_loader)
#     total_loss = 0

#     model.eval()
#     with torch.no_grad():
#         for X, y in data_loader:
#             X = X.to(device)
#             y = y.to(device)
#             output = model(X[:,:,0:10], X[:,:,10:])
#             total_loss += math.sqrt(loss_function(output[:, :, 0:10], y[:, :, 0:10]).item())

#     avg_loss = total_loss / num_batches
#     test_losses.append(avg_loss)
#     # print(f"Test loss: {avg_loss}")


def lstm_predict(model, dataset):
    model.eval()
    with torch.no_grad():
        all_outputs = torch.zeros(1, model.num_features_pred).to(model.device)
        for i in tqdm(range(len(dataset))):  # (len(df_test)-8)/8 = 92回予測が必要
            src = dataset[i][0].float().to(model.device)
            tgt = dataset[i][1].float().to(model.device)

            # add a batch size of 1 for the encoder (= [1, :, :])
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)

            seq_len_src = src.shape[1]
            seq_len_tgt = tgt.shape[1]

            # initialize tensor for predictions
            outputs = torch.zeros(seq_len_tgt, model.num_features_pred).to(model.device)  # outputs.shape =  [seq_len, num_features_pred]
            # decode input_tensor
            lstm_input = src[:, -1:, :]

            for t in range(seq_len_tgt):
                output = model(lstm_input)   # output.shape = [batch_size, 1(seq_len), num_features_pred]
                outputs[i] = output[-1, :, :]
                # decoder input at the next period = (location output from prediction) + (acurate dummy input from test data)
                lstm_input = torch.cat((output[:, :, 0:model.num_features_pred], tgt[:, t:t+1, model.num_features_pred:]), dim=2)

            all_outputs = torch.cat([all_outputs, outputs], dim=0)

    return all_outputs[1:, :model.num_features_pred]
