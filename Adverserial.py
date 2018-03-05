from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import copy

NUM_MODEL = 6
DEVIDE = 3

data_test = MNIST('./pytorch_data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Scale((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# TODO: check wether the precidction is correct label
def Attacker(images, labels, net_ls, num_steps =100, ganma=0):
  orig_images = copy.deepcopy(images)
  orig_images.requires_grad = False # TODO: fix
  optimizer = optim.Adam(images, lr=8)
  alpha = Variable(1. / len(net_ls), requires_grad=False)
  raw_loss = Variable(torch.zeros(1), requires_grad=False)
  for i in range(num_steps):
      for net in net_ls
          output = alpha*net(images)*labels
          raw_loss += torch.log(output)
      loss = T.sum(raw_loss)
      loss += ganma  / float(images.size()[0] * images.size()[1]) * (orig_images-images) ** 2 # TODO: devide by the shape of
      loss.backward()
      optimizer.step()
  return images


def Defender(images, labels, net_ls):
    output = Variable(torch.ones(labels.size()))
    for net in net_ls
        output*=net(images) # TODO: multiplication
    pred = output.data.max(1)[1]
    total_correct += pred.eq(labels.data.view_as(pred)).sum()
    return total_correct


def main():
    # Load Attacker Nets
    total_sample = 0
    total_correct = 0
    attacker_net_ls = []
    for model_idx in range(NUM_MODEL-DEVIDE):
        atatcker_net_ls.append(LeNet5())
    for model_idx in range(NUM_MODEL-DEVIDE):
        net_name = "models/LeNet-" + str(model_idx) + ".pt"
        torch.load(attacker_net_ls[model_idx].state_dict(), net_name)

    # Load Defender Nets
    defender_net_ls = []
    for model_idx in range(NUM_MODEL-DEVIDE):
        defender_net_ls.append(LeNet5())
    for model_idx in range(NUM_MODEL-DEVIDE):
        net_name = "models/LeNet-" + str(model_idx + DEVIDE) + ".pt"
        torch.load(defender_net_ls[model_idx].state_dict(), net_name)

    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = Variable(images, requires_grad=True), Variable(labels, requires_grad=False)
        adv_imgs = Attacker(images, labels, attacker_net_ls, num_steps =100, ganma=0)
        total_sample += 1
        attacker_total_correct += Defender(adv_imgs, labels, attacker_net_ls)
        defender_total_correct += Defender(adv_imgs, labels, defender_net_ls)

    print("Total Accuracy on Attacker Nertwork {}". format(attacker_total_correct / float(total_correct)))
    print("Total Accuracy on Defender Nertwork {}". format(defender_total_correct / float(total_correct)))

if __name__ == '__main__':
    main()
