from autoattack import AutoAttack
from models import *
import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
# load model
# loading the model
if __name__ == '__main__':
    path = "./cifar10model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18().to(device)
    modelDict = torch.load(path)
    epsilon = 0.031
    version = 'standard'
    save_dir = "./attackDir"
    norm = 'Linf'
    
    #for unkown reason, key expected to be in format of "conv1.weight", but actually to be in format of "module.conv1.weight"
    #so remove 'module.' for all keys
    my_dic_keys = list(modelDict.keys())
    for key in my_dic_keys:
        if "module." in key:
            newKey = key[7:]
            modelDict[newKey] = modelDict.pop(key) 

    # load data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    batch_size = 100
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    #finally load module
    model.load_state_dict(modelDict)
    model.eval()


    # create save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load attack    
    adversary = AutoAttack(model, norm = norm, eps=epsilon, version=version)

    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in testloader]
    y_test = torch.cat(l, 0)

    individual = False
    n_ex = 50000
    # run attack and save images
    with torch.no_grad():
        if not individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex], return_labels=True,
                bs=batch_size)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_{}_{}_eps_{:.5f}.pth'.format(
                save_dir, 'aa', version, n_ex, norm, epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:n_ex],
                y_test[:n_ex], return_labels=True, bs=batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                save_dir, 'aa', version, n_ex, epsilon))