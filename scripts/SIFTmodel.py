import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import os
import argparse

# create the SIFTMatcher model
class SIFTnn(nn.Module):
    def __init__(self):
        super(SIFTnn, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # concatenate x1 and x2
        x = torch.cat((x1, x2)).double()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# class SIFTnn(nn.Module):
#     def __init__(self):
#         super(SIFTnn, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(256*2, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x1, x2):
#         out1 = self.feature_extractor(x1)
#         out2 = self.feature_extractor(x2)
#         out = torch.cat((out1, out2), dim=0)
#         out = self.classifier(out)
#         return out

def loadData(xdata_path, ydata_path):
    x = np.load(xdata_path, allow_pickle=True)
    y = np.load(ydata_path, allow_pickle=True)
    return x, y

def saveModel(model, path: str='model.pth'):
    torch.save(model.state_dict(), path)

def loadModel(path: str='.') -> torch.nn:
    model = SIFTnn()
    model.load_state_dict(torch.load(os.path.abspath(path)))
    model.eval()
    return model

def trainModel(**kwargs) -> torch.nn:

    model = kwargs['model']
    train_loader = kwargs['train_loader']
    test_loader = kwargs['test_loader']
    optimizer = kwargs['optimizer']
    criterion = kwargs['criterion']
    device = kwargs['device']
    # train the model
    epoch = 0
    running_loss = 0.0
    model.double()

    for x_batch, y_batch in train_loader:
        for i, x in enumerate(x_batch):
            # zero the parameter gradients
            optimizer.zero_grad()
            # x.to(device)
            # forward + backward + optimize
            outputs = model(x[:128], x[128:])
            # print(f'pred : {outputs[0]}\temp : {y_batch[i]}')
            loss = criterion(outputs[0], y_batch[i])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 32 == 31: # print every 100 batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            epoch+=1

    # evaluate the model on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            for i, x in enumerate(x_batch):
                outputs = model(x[:128], x[128:])
                predicted = (outputs[0] > 0.5).float()
                total += 1
                if predicted == y_batch[i]:
                    correct += 1

    print('Accuracy of the network on the %d test descriptor pairs: %d %%' % (total, 100 * correct / total))

    return model

if __name__ == '__main__':

    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="Path to model file (.pth). If no model is provide, a new one will be created and saved as model.pth") # path to model
    parser.add_argument('-f', '--features', help="Path to features to use in training")
    parser.add_argument('-l', '--labels', help="Validation/Labels set for use in training")
    parser.add_argument('-o', '--output', help="Model output path. WARNING this will overwrite any existing models with the same name")
    args = parser.parse_args()

    # Verify the required input
    if args.labels == None:
        print('Please specify path to labels (*.npy)')
        parser.print_help()
        exit(1)

    if args.features == None:
        print('Please specify path to features (*.npy)')
        parser.print_help()
        exit(1)

    if args.output == None:
        print("No specified output")
        parser.print_help()
        exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the features and the labels
    x,y = loadData(args.features, args.labels)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    dataset = TensorDataset(x, y)

    # Define train and test sizes
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Create the model, optimizer and loss criteria
    model = SIFTnn() if args.model == None else loadModel(args.model)
    print(f'Training on {device}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # binary cross entropy loss function

    # Train model
    model = trainModel(model=model, train_loader=train_loader,
                        test_loader=test_loader, optimizer=optimizer,
                        criterion=criterion, device=device)

    # Save the model
    saveModel(model, args.output)