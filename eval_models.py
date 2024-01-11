import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from model import RIID_NN
from dataloader import dataLoader

# reverse one hot encoding for question difficulties
def part_to_numbers(part):
    if part == 'part_1':
        return 1
    if part == 'part_2':
        return 2
    if part == 'part_3':
        return 3
    if part == 'part_4':
        return 4
    if part == 'part_5':
        return 5
    
def community_to_number(community):
    if community == 'community_0':
        return 0.0
    if community == 'community_1':
        return 1.0
    if community == 'community_2':
        return 2.0
    if community == 'community_3':
        return 3.0

def IRT(skill, diff):
    return (torch.exp(skill-diff)/(1 + torch.exp(skill - diff)))

# evaluate model
def evaluate(model, loader, device, optimiser):
    accuracy = 0.0
    rash_acc = 0.0
    total = 0
    pred = []
    labels = []
    rash_pred = []
    difficulties = []
    model.eval()
    for idx, (feature, label, difficulty) in enumerate(loader):
        optimiser.zero_grad()
        out = model(feature.to(device))[:,0]
        labels.append(label)
        pred.append(out.detach().cpu().numpy())
        rash = IRT(out, difficulty.to(device))
        rash_pred.append(rash.detach().cpu().numpy())
        rash = (rash > 0.5).float()
        rash_acc += (label.to(device) == rash).sum().item()
        out = (out > 0.5).float()
        accuracy += (label.to(device) == out).sum().item()
        total += len(label)
    accuracy = accuracy / total
    rash_acc = rash_acc / total
    print('Rash accuracy: {:.4f}'.format(rash_acc))
    return accuracy, pred, rash_pred, labels

def evaluate_IRT(model, loader, device, optimiser):
    accuracy = 0.0
    total = 0
    pred = []
    labels = []
    model.eval()
    for idx, (feature, label, difficulty) in enumerate(loader):
        optimiser.zero_grad()
        out = model(feature.to(device))[:,0]
        labels.append(label)
        out = IRT(out, difficulty.to(device))
        pred.append(out.detach().cpu().numpy())
        out = (out > 0.5).float()
        accuracy += (label.to(device) == out).sum().item()
        total += len(label)
    accuracy = accuracy / total
    return accuracy, pred, labels


def question_difficulties(df):
    one_hot_encoded = df[['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6']]
    original_part = one_hot_encoded.idxmax(axis=1).reset_index()
    original_part['part'] = original_part[0].apply(part_to_numbers)
    one_hot_encoded = df[['community_0', 'community_1', 'community_2', 'community_3']]
    original_community = one_hot_encoded.idxmax(axis=1).reset_index()
    original_community['community'] = original_community[0].apply(community_to_number)
    df['part'] = original_part['part']
    df['community'] = original_community['community']

    # average correct answers per skill
    avr_ans_per_skill = df.groupby(['part', 'community'])['answered_correctly'].mean().reset_index()
    avr_ans_per_skill.rename(columns={'answered_correctly':'difficulty'}, inplace=True)
    df = df.merge(avr_ans_per_skill, how='inner', left_on=['part', 'community'], right_on=['part', 'community'])
    
    return df


# model saves
model_saves = './models/'

# read data and drop feature 'part_5.1'
data = pd.read_csv('./saves/pre_process_10000000.csv', nrows=1000)
data = data.drop('part_5.1', axis=1)

# add difficulty
data = question_difficulties(data)

# split train and validation
trainlen = (2 * len(data)) // 3
val_data = data.iloc[trainlen:]
len_val_data = len(val_data)
val_indices = np.random.permutation(len(val_data))
val_indices = val_indices[:200]
val_data = val_data.iloc[val_indices]

# data loader
valdata = dataLoader(val_data)
val_loader = DataLoader(valdata, batch_size=1, shuffle=True)

lr = 1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'GPU not found')
model = RIID_NN().to(device)
model.load_state_dict(torch.load(model_saves+'BCELogits/model_epoch15.pt'))
model_IRT = RIID_NN().to(device)
model_IRT.load_state_dict(torch.load(model_saves + 'IRT/model_epoch4.pt'))

optimiser = optim.Adam(model.parameters(), lr = lr)
criterion = nn.BCEWithLogitsLoss()

# evaluate on validation set
acc_base, pred_base, pred_rash, label_base = evaluate(model, val_loader, device, optimiser)
acc_irt, pred_irt, label_irt = evaluate_IRT(model_IRT, val_loader, device, optimiser)

# print accuracy of both models
print('Accuracy of base NN: {:.4f}\tAccuracy with IRT: {}'.format(acc_base, acc_irt))

# ROC curves
fpr, tpr, _ = roc_curve(label_base, pred_base, pos_label=1)
auc = roc_auc_score(label_base, pred_base)

pred = (np.array(pred_irt) > 0.5)
cf_matrix = confusion_matrix(label_base, pred)
sns.heatmap(cf_matrix, annot=True)
plt.show()
print('Neural Network')
print(cf_matrix)

fpr_irt, tpr_irt, _ = roc_curve(label_irt, pred_irt, pos_label=1)
auc_irt = roc_auc_score(label_irt, pred_irt)

pred = (np.array(pred_irt) > 0.5)
cf_matrix = confusion_matrix(label_irt, pred)
sns.heatmap(cf_matrix, annot=True)
plt.show()
print('\n\nIRT')
print(cf_matrix)

fpr_rash, tpr_rash, _ = roc_curve(label_base, pred_rash, pos_label=1)
auc_rash = roc_auc_score(label_base, pred_rash)

pred = (np.array(pred_irt) > 0.5)
cf_matrix = confusion_matrix(label_base, pred)
sns.heatmap(cf_matrix, annot=True)
plt.show()
print('\n\nRash')
print(cf_matrix)


# plot 
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='ROC curve(AUC={:.2f})'.format(auc))
ax.plot([0,1], [1,0], linestyle='--')
ax.set_xlabel('False Positive')
ax.set_ylabel('True Positive')
ax.legend(loc='lower right')
plt.title('ROC Çurve')
plt.savefig('./saves/roc_base.png')

figrt, axirt = plt.subplots()
axirt.plot(fpr_irt, tpr_irt, label='ROC curve(AUC={:.2f})'.format(auc_irt))
axirt.plot([0,1], [1,0], linestyle='--')
axirt.set_xlabel('False Positive')
axirt.set_ylabel('True Positive')
axirt.legend(loc='lower right')
plt.title('ROC Çurve')
plt.savefig('./saves/roc_irt.png')

figrash, axrash = plt.subplots()
axrash.plot(fpr_rash, tpr_rash, label='ROC curve(AUC={:.2f})'.format(auc_rash))
axrash.plot([0,1], [1,0], linestyle='--')
axrash.set_xlabel('False Positive')
axrash.set_ylabel('True Positive')
axrash.legend(loc='lower right')
plt.title('ROC Çurve')
plt.savefig('./saves/roc_base_rash.png')


figmerg, axmerge = plt.subplots()
axmerge.plot(fpr, tpr, color='b', label='ROC curve(AUC={:.2f}})'.format(auc))
axmerge.plot(fpr_irt, tpr_irt, color='r', label='ROC curve IRT(AUC={:.2f})'.format(auc_irt))
axmerge.plot(fpr_rash, tpr_rash, color='k', label='ROC Curve Rash(AUC={:.2f}'.format(auc_rash))
axmerge.plot([0,1], [1,0], linestyle='--')
axmerge.set_xlabel('False Positive')
axmerge.set_ylabel('True Positive')
axmerge.legend(loc='lower right')
plt.title('ROC Çurve')
plt.savefig('./saves/roc_base_rash_irt.png')

