import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import RNN
from data_loader import Riid_RNN_Dataset
from read_riid_file import read_riid_file

# train network
def train(model, loader, optimiser, criterion, device):
    # switch to train
    model.train()
    running_loss = 0.0
    for idx, (data, label) in enumerate(loader):
        label = label.to(device)
        data = [data[0].to(device), data[1].to(device)]
        optimiser.zero_grad()
        output = model([data[0].to(device), data[1].to(device)], device)
        loss = criterion(output, label)
        loss.backward()
        optimiser.step()
        running_loss += loss.detach().item()

    running_loss = running_loss / loader.__len__()
    return running_loss

# validate network
def validation(model, loader, optimiser, criterion, device):
    # switch to evaluation
    model.eval()
    running_loss = 0.0
    accuracy = 0.0
    samples = 0
    for idx, (data, label) in enumerate(loader):
        label = label.to(device)
        data = [data[0].to(device), data[1].to(device)]
        optimiser.zero_grad()
        output = model(data, device)
        loss = criterion(output, label)
        running_loss += loss.detach().item()
        accuracy += torch.sum((output > 0.5).type(torch.cuda.FloatTensor) == label).detach().item()
        samples += data[0].shape[0]

    running_loss = running_loss / loader.__len__()
    accuracy /= samples

    return running_loss, accuracy

def main():
    # save folder
    save_folder = './saves/models/CNN/'

    # train file
    train_file = './data/train.csv'
    columns = ['content_id', 'task_container_id', 'answered_correctly',
    'prior_question_elapsed_time', 'prior_question_had_explanation']
    trainset = read_riid_file(file=train_file)
    scale = 'prior_question_elapsed_time'
    trainset.prepare_data(columns, scale)
    split_train, split_validation = trainset.split_data()
    print('Train length: {}\tTrain True: {}'.format(len(split_train), np.sum(split_train['answered_correctly'])))
    init_loss = np.sum(split_train['answered_correctly'])/len(split_train)
    print('Loss if all predicted wrong: {:.4f}'.format(init_loss))
    print('Validation length: {}\tValidation True: {}'.format(len(split_validation),
                                                              np.sum(split_validation['answered_correctly'])))
    train_dataset = Riid_RNN_Dataset(split_train)
    validation_dataset = Riid_RNN_Dataset(split_validation)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, num_workers=0)
    val_loader = DataLoader(validation_dataset, batch_size=2048, shuffle=False, num_workers=0)

    # embeddings
    embedding_size = trainset.get_embeddings()

    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model parameters
    n_continous = 2
    n_hidden = 80
    n_layers = 3
    n_outputs = 1
    model = RNN(n_contineous_inputs=n_continous, n_hidden=n_hidden, n_rnnlayers=n_layers,
                n_outputs=n_outputs, embedding_sizes=embedding_size).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.BCELoss()

    # train
    num_epoch = 5
    train_loss = np.zeros((1, num_epoch))
    val_loss = np.zeros((1, num_epoch))
    val_acc = np.zeros((1, num_epoch))
    for epoch in range(num_epoch):
        train_epoch_loss = train(model, train_loader, optimiser, criterion, device)
        val_epoch_loss, val_epooch_acc = validation(model, val_loader, optimiser, criterion, device)
        np.append(train_loss, train_epoch_loss)
        np.append(val_loss, val_epoch_loss)
        np.append(val_acc, val_epooch_acc)

        print('Epoch {}/{}\tTrain Loss: {:.4f}\tValidation Loss: {:.4f}\tValidation Accuracy: {:.4f}\t'.format(
            epoch, num_epoch, train_epoch_loss, val_epoch_loss, val_epooch_acc))

        torch.save(model.state_dict(), save_folder + '/cnn_model_epoch{}.pt'.format(epoch))
        
if __name__ = '__main__':
    main()
        
