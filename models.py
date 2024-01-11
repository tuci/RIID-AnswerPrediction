import torch
import numpy as np
import torch.nn as nn

# CNN model
class RIIDModel(nn.Module):
    def __init__(self):
        super(RIIDModel, self).__init__()
        self.cnn1 = nn.Conv1d(1, 32, 2)
        self.cnn2 = nn.Conv1d(32, 64, 2)
        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.cnn1(x))
        out = self.drop(self.relu(self.cnn2(out)))
        out = out.view(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.activation(self.fc2(out))
        return out

    # LSTM model


class RIIDModelLSTM(nn.Module):
    def __init__(self):
        super(RIIDModelLSTM, self).__init__()
        self.n_layers = 2
        self.lstm = nn.LSTM(6, 6, num_layers=self.n_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        c0 = torch.zeros((self.n_layers, batch_size, 6)).to(self.device)
        h0 = torch.zeros((self.n_layers, batch_size, 6)).to(self.device)
        return h0, c0

    def forward(self, x):
        batch_size = x.shape[0]
        h0, c0 = self.init_hidden(batch_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.drop(self.relu(self.fc1(out)))
        out = self.drop(self.relu(self.fc2(out)))
        return out


class RNN(nn.Module):
    def __init__(self, n_contineous_inputs, n_hidden, n_rnnlayers, n_outputs, embedding_sizes):
        super(RNN, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_contineous_inputs

        self.emb_drop = nn.Dropout(0.3)

        self.D = self.n_emb + self.n_cont
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        # print(f'RNN LSTM input size is: {self.D}')

        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True)
        self.fc = nn.Linear(self.M, self.K)
        self.activation = nn.Sigmoid()

    def forward(self, X, device):
        X_categorical, X_continuous = (*X,)

        # initial hidden states
        h0 = torch.zeros(self.L, X_categorical.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X_categorical.size(0), self.M).to(device)

        # print(f'categorical type: {X_categorical.dtype} & shape : {X_categorical.shape}')
        # print(f'continuous type: {X_continuous.dtype} & shape : {X_continuous.shape}')

        # below code is not needed
        #     x = [print(embedding) for col_idx,embedding in enumerate(self.embeddings)]
        #     x = [print(self.get_unique_categorical_data(X_categorical, col_idx).shape) for col_idx,embedding in enumerate(self.embeddings)]

        #     for col_idx,embedding in enumerate(self.embeddings):
        #         print(col_idx)
        #         print(f'input min & max: {torch.min(self.get_unique_categorical_data(X_categorical, col_idx))} & {torch.max(self.get_unique_categorical_data(X_categorical, col_idx))}')
        #         print(f'')
        #         t = embedding(self.get_unique_categorical_data(X_categorical, col_idx))
        # end of not needed code

        # categorial columns are first 2 columns in X
        # x = [embedding(self.get_unique_categorical_data(X_categorical, col_idx)) for col_idx,embedding in enumerate(self.embeddings)]
        x = [embedding(X_categorical[:, :, col_idx]) for col_idx, embedding in enumerate(self.embeddings)]
        # print(f'default shape: {x[0].shape} & {x[1].shape}')
        x = torch.cat(x, 2)
        # print(f'after merge shape: {x.shape}')
        x = self.emb_drop(x)  # I can remove dropout if this is unable to compile

        # concatentate last 2 columns (that are contineous columns - first two are categorical)
        x = torch.cat([x, X_continuous], 2)
        # print(f'RNN forward input size is: {x.shape}')

        # get RNN unit output
        out, _ = self.rnn(x.to(device), (h0, c0))

        # we only want h(T) at the final time step
        out = self.activation(self.fc(out[:, -1, :]))
        return out

    def get_unique_categorical_data(self, x, col_idx):
        x = x.reshape(-1, 2)
        _, idx = np.unique(x[:, col_idx], return_index=True)
        return x[np.sort(idx), col_idx]