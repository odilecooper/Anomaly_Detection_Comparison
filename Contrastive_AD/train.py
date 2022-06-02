import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import Contrastive_AD.helper_functions as helper_functions


class DatasetBuilder(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': self.data[idx], 'index': idx}
        return sample

class encoder_a(nn.Module):
    def __init__(self, kernel_size,hdn_size,d):
        super(encoder_a, self).__init__()
        self.fc1 = nn.Linear(d-kernel_size, hdn_size) #F network
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(hdn_size, hdn_size*2)
        self.activation2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hdn_size*2, hdn_size)
        self.activation3 = nn.LeakyReLU(0.2)
        self.batchnorm_1 = nn.BatchNorm1d(d-kernel_size+1)
        self.batchnorm_2 = nn.BatchNorm1d(d-kernel_size+1)
        self.fc1_y = nn.Linear(kernel_size, int(hdn_size/4)) #G network
        self.activation1_y = nn.LeakyReLU(0.2)
        self.fc2_y = nn.Linear(int(hdn_size/4), int(hdn_size/2))
        self.activation2_y = nn.LeakyReLU(0.2)
        self.fc3_y = nn.Linear(int(hdn_size/2), hdn_size)
        self.activation3_y = nn.LeakyReLU(0.2)
        self.kernel_size = kernel_size
        self.batchnorm1_y = nn.BatchNorm1d(d-kernel_size+1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y, x = helper_functions.positive_matrice_builder(x, self.kernel_size)
        x = self.activation1(self.fc1(x))
        x = self.batchnorm_1(x)
        x = self.activation2(self.fc2(x))
        x = self.batchnorm_2(x)
        x = self.activation3(self.fc3(x))
        y = self.activation1_y(self.fc1_y(y))
        y = self.batchnorm1_y(y)
        y = self.activation2_y(self.fc2_y(y))
        y = self.activation3_y(self.fc3_y(y))
        x = nn.functional.normalize(x,dim=1)
        y = nn.functional.normalize(y,dim=1)
        x = nn.functional.normalize(x,dim=2)
        y = nn.functional.normalize(y,dim=2)
        return (x, y)


class ContrastiveTrainer():
    def __init__(self, batch_size, device, faster_version=False):
        self.num_epochs = 2000
        self.no_btchs = batch_size
        self.no_negatives = 5
        self.temperature = 0.01
        self.lr = 0.01
        self.device = device
        self.faster_version = faster_version
        
    def train_and_evaluate(self, train, test, categories):
        train = torch.as_tensor(train, dtype=torch.float)
        test = torch.as_tensor(test, dtype=torch.float)
        test_losses_contrastloss = torch.zeros(test.shape[0],dtype=torch.float).to(self.device)
        d = train.shape[1]
        n = train.shape[0]
        if self.faster_version=='yes':
            num_permutations = min(int(np.floor(100 / (np.log(n) + d)) + 1),2)
        else:
            num_permutations=int(np.floor(100/(np.log(n)+d))+1)
        print("going to run for: ", num_permutations, ' permutations')
        hiddensize = 200
        if d <= 40:
            kernel_size = 2
            stop_crteria = 0.001
        if 40 < d and d <= 160:
            kernel_size = 10
            stop_crteria = 0.01
        if 160 < d:
            kernel_size = d - 150
            stop_crteria = 0.01
        for permutations in range(num_permutations):
            if num_permutations > 1:
                random_idx = torch.randperm(train.shape[1])
                train = train[:, random_idx]
                test = test[:, random_idx]
            dataset_test = DatasetBuilder(test)
            dataset_train = DatasetBuilder(train)
            model_a = encoder_a(kernel_size, hiddensize, d).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer_a = torch.optim.Adam(model_a.parameters(), lr=self.lr)
            trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                     shuffle=True, num_workers=0, pin_memory=True)
            testloader = DataLoader(dataset_test, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)

            ### training
            for epoch in range(self.num_epochs):
                model_a.train()
                running_loss = 0
                for i, sample in enumerate(trainloader, 0):
                    model_a.zero_grad()
                    pre_query = sample['data'].to(self.device)
                    pre_query = torch.unsqueeze(pre_query, 1)
                    pre_query, positives_matrice = model_a(pre_query)
                    scores_internal = helper_functions.scores_calc_internal(pre_query, positives_matrice, self.no_negatives, self.temperature).to(self.device)
                    scores_internal = scores_internal.permute(0, 2, 1)
                    correct_class = torch.zeros((np.shape(scores_internal)[0], np.shape(scores_internal)[2]),
                                                dtype=torch.long).to(self.device)
                    loss = criterion(scores_internal, correct_class).to(self.device)
                    loss.backward()
                    optimizer_a.step()
                    running_loss += loss.item()
                if (running_loss / (i + 1) < stop_crteria):
                    break
                if n<2000:
                    if (epoch + 1) % 100 == 0:
                        print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                else:
                    if (epoch + 1) % 10 == 0:
                        print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
            
            ### testing
            model_a.eval()
            criterion_test = nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                for i, sample in enumerate(testloader, 0):
                    pre_query = sample['data'].to(self.device)
                    indexes = sample['index'].to(self.device)
                    pre_query_test = torch.unsqueeze(pre_query, 1)  # batch X feature X 1
                    pre_query_test, positives_matrice_test = model_a(pre_query_test) # F(b)
                    scores_internal_test = helper_functions.scores_calc_internal(pre_query_test, positives_matrice_test, self.no_negatives, self.temperature).to(self.device)
                    scores_internal_test = scores_internal_test.permute(0, 2, 1)
                    correct_class = torch.zeros((np.shape(scores_internal_test)[0], np.shape(scores_internal_test)[2]),
                                                dtype=torch.long).to(self.device)
                    loss_test = criterion_test(scores_internal_test, correct_class).to(self.device)
                    test_losses_contrastloss[indexes] += loss_test.mean(dim=1).to(self.device)
        
        f1 = helper_functions.f1_calculator(categories, test_losses_contrastloss) # bigger the loss, more probability to be anomaly
        test_losses_contrastloss = test_losses_contrastloss.cpu()
        auc = roc_auc_score(categories, test_losses_contrastloss)
        score = average_precision_score(categories, test_losses_contrastloss)
        return (f1, auc, score)

