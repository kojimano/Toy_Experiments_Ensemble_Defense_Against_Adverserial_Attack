"""
Implementation of Adverserial Attack/Defense Experiments
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
import copy
import os
import numpy as np
import argparse
from IPython import embed

parser.add_argument('--num_attacker_clone',  type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--num_defender_clone',  type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--attacker_logic',  type=string, default="mult", help='an integer for the accumulator')
parser.add_argument('--atacker_sample',  type=string, default="all", help='an integer for the accumulator')
parser.add_argument('--defender_logic',  type=string, default="mult", help='an integer for the accumulator')
parser.add_argument('--defender_sample',  type=string, default="all", help='an integer for the accumulator')
args = parser.parse_args()

NUM_MODEL = args.num_attacker_clone + args.num_defender_clone
DEVIDE = args.num_attacker_clone

data_test = MNIST('./pytorch_data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Scale((32, 32)),
                      transforms.ToTensor()]))
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# TODO: check wether the precidction is correct label
def Attacker(images, labels, net_ls, num_steps =100, ganma=0, sample="all"):
  orig_images = copy.deepcopy(images)
  orig_images.requires_grad = False # TODO: fix
  optimizer = optim.Adam([images], lr=8)
  alpha = Variable(torch.from_numpy(np.array([1. / len(net_ls)]).astype(np.float32)), requires_grad=False)
  raw_loss = Variable(torch.zeros(labels.size()), requires_grad=False)
  for i in range(num_steps):
      optimizer.zero_grad()
      loss = Variable(torch.zeros(1), requires_grad=False)
      if sample == "all":
          for net in net_ls:
              output = alpha*net(images)*labels
              raw_loss += output
      elif sample == "single":
          net_idx = np.random.randint(len(net_ls), size=1)
          net = net_ls[net_idx]
          alpha = 1.0
          output = alpha*net(images)*labels
          raw_loss += output
      loss = torch.sum(raw_loss)
      loss += ganma  / float(images.size()[2] * images.size()[2]) * torch.sum((orig_images-images) ** 2) # TODO: devide by the shape of
      loss.backward(retain_graph=True)
      #print("Iter {} Loss {}".format(i, loss))
      optimizer.step()
  return images

# TODO: Add options to do parameter ensemble
def Defender(images, labels, net_ls, logic="mult", sample="all", param_sample=None):
    if logic not in ["mult", "avg"]:
        assert(False)
    if sample not in ["single", "all", "random"]:
        assert(False)
    if torch.cuda.is_available():
        labels = Variable(torch.from_numpy(np.array([labels]).astype(np.int)).cuda())
    else:
        labels = Variable(torch.from_numpy(np.array([labels]).astype(np.int)))
    total_correct = 0
    if logic == "mult":
        output = Variable(torch.ones((1,10)))
    elif logic == "avg":
        output = Variable(torch.zeros((1,10)))
    if torch.cuda.is_available():
	output = output.cuda()
    else:
        assert(False)
    if sample == "all":
        for net in net_ls:
            if logic == "mult":
                output*=net(images)
            elif logic == "avg":
                output+=net(images)
            else:
                assert(False)
    elif sample == "single":
        net_idx = np.random.randint(len(net_ls), size=1)
        net = net_ls[net_idx]
        if logic == "mult":
            output*=net(images)
        elif logic == "avg":
            output+=net(images)
        else:
            assert(False)
    elif sample == "random":
        sample_num = np.random.randint(len(net_ls), size=1)
        sample_indcies = np.random.randint(len(net_ls), size=sample_num)
        net_ls = net_ls[sample_indcies]
        for net in net_ls:
            if logic == "mult":
                output*=net(images)
            elif logic == "avg":
                output+=net(images)
            else:
                assert(False)

    embed()
    pred = output.data.max(1)[1]
    total_correct += pred.eq(labels.data.view_as(pred)).sum()
    return total_correct


def main():
    # Load Attacker Nets
    total_sample = 0
    attacker_total_correct = 0
    defender_total_correct = 0
    attacker_net_ls = []
    for model_idx in range(NUM_MODEL-DEVIDE):

        attacker_net_ls.append(LeNet5())
    for model_idx in range(NUM_MODEL-DEVIDE):
        net_name = "models/LeNet-" + str(model_idx) + ".pt"
        torch.load(net_name, attacker_net_ls[model_idx].state_dict())


    # Load Defender Nets
    defender_net_ls = []
    for model_idx in range(NUM_MODEL-DEVIDE):
	net = LeNet5()
	if torch.cuda.is_available():
            net = net.cuda()
        defender_net_ls.append(net)
    for model_idx in range(NUM_MODEL-DEVIDE):
        net_name = "models/LeNet-" + str(model_idx + DEVIDE) + ".pt"
        torch.load(net_name, defender_net_ls[model_idx].state_dict())

    for i, (images, labels) in enumerate(data_test_loader):
        print("Itreation: {} ....".format(i))
        for raw_image, raw_label in zip(images, labels):
            label = np.zeros((1,10))
            label[:,raw_label] = 1
            label = label.astype(np.float32)
            raw_image = torch.unsqueeze(raw_image, 0)
            if torch.cuda.is_available():
                images, labels = Variable(raw_image.cuda(), requires_grad=True), Variable(torch.from_numpy(label).cuda(), requires_grad=False)
	    else:
                images, labels = Variable(raw_image, requires_grad=True), Variable(torch.from_numpy(label), requires_grad=False)
            adv_img = Attacker(images, labels, attacker_net_ls, num_steps =100, ganma=0)
            total_sample += 1
            attacker_total_correct += Defender(adv_img, raw_label, attacker_net_ls, logic=args.attacker_logic)
            defender_total_correct += Defender(adv_img, raw_label, defender_net_ls, logic=args.defender_logic, sample=args.defender_sample)

    print("Total Accuracy on Attacker Nertwork {}". format(attacker_total_correct / float(total_correct)))
    print("Total Accuracy on Defender Nertwork {}". format(defender_total_correct / float(total_correct)))

if __name__ == '__main__':
    main()
