import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn as nn

from net import Net, ResNet
from dataset import custom_MNIST_dataset

import argparse

parser = argparse.ArgumentParser(description='domain randomization excise')
parser.add_argument('--type', type=int, help='define domain_randomization_type')
parser.add_argument('--use_resnet', type=int, default=0, help='define domain_randomization_type')
args = parser.parse_args()

def main():
    batch_size = 32
    learning_rate = 0.001
    num_workers = 8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define data loader
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_mnist = custom_MNIST_dataset(root='./data', train=True, domain_randomziation_type = args.type,
                                            download=True)
    trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, drop_last = True)

    tesset_svhn = torchvision.datasets.SVHN(root='./data', split="test",
                                        download=True, transform=transform)
    testloader_svhn = torch.utils.data.DataLoader(tesset_svhn, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last = True)

    
    # define CNN
    if args.use_resnet == 1:
        print("use resnet as CNN model")
        net = ResNet()
    else:
        print("use simple CNN model")
        net = Net()
    net.to(device)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # start training
    for epoch in range(20):  
        running_loss = 0.0
        for i, data in enumerate(trainloader_mnist):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # test with target dataset
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader_svhn:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the target dataset: %d %%' % (
            100 * correct / total))

    print('Finished Training')


    


if __name__ == "__main__":
    main()