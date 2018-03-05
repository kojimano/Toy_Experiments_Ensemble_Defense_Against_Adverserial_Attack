"""
Credit :  https://github.com/activatedgeek/LeNet-5
Note: Some modification has been done on the code provided above URL
"""
from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom

NUM_MODEL = 20

viz = visdom.Visdom()
data_train = MNIST('./pytorch_data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Scale((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./pytorch_data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Scale((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

net_ls = []
for i in range(NUM_MODEL):
    net_ls.append(LeNet5())

criterion = nn.CrossEntropyLoss()

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch, net=None, model_idx=0):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images), Variable(labels)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data[0])
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Model %d, Epoch %d, Batch: %d, Loss: %f' % (model_idx, epoch, i, loss.data[0]))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.FloatTensor(loss_list), torch.FloatTensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test(net=None):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data[0], float(total_correct) / len(data_test)))


def train_and_test(epoch, net, model_idx=0):
    train(epoch, net, model_idx)
    test(net)


def main():
    for model_idx in range(NUM_MODEL):
        for e in range(1, 16):
            train_and_test(e, net_ls[model_idx], model_idx)
        net_name = "models/LeNet-" + str(model_idx) + ".pt"
        torch.save(net_ls[model_idx].state_dict(), net_name)


if __name__ == '__main__':
    main()
