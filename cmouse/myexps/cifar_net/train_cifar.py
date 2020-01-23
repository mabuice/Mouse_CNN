import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
sys.path.append('../../')
from anatomy import *
from network import *
from config import *
import random
import wandb

# WandB – Initialize a new run
wandb.init(dir=WANDB_DIR)
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 128         # input batch size for training (default: 64)
config.test_batch_size = 128    # input batch size for testing (default: 1000)
config.epochs = 60             # number of epochs to train (default: 10)
config.momentum = 0.5          # SGD momentum (default: 0.5) 
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
init_lr = 0.1
config.lr = init_lr               # learning rate (default: 0.01)

def train(args, model, device, train_loader, optimizer, epoch):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    running_loss = 0.0
    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader):
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)
        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()
        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        output = model(data)
        # Define our loss function, and compute the loss
        loss = F.cross_entropy(output, target)
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        # Update the neural network weights
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 20 == 19:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx + 1, running_loss / 20))
            wandb.log({
            "running loss": running_loss/20})
            running_loss = 0.0
            

def test(args, model, device, test_loader):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            output = model(data)
            # Compute the loss sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))
    
    # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})

# prepare for data
class GBChannels():
    def __call__(self, tensor):
        return tensor[1:3,:,:]

transform = transforms.Compose([transforms.Resize(INPUT_SIZE[1:]),
                                transforms.ToTensor(),
                                GBChannels(),
                                transforms.Normalize((0.5, 0.5),(0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=DATA_FOLDER, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_FOLDER, train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get the shape of the input
for i, data in enumerate(train_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)
    print("The shape of input is %s"%str(inputs.shape))
    break


## main function
use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Set random seeds and deterministic pytorch for reproducibility
random.seed(config.seed)       # python random seed
torch.manual_seed(config.seed) # pytorch random seed
np.random.seed(config.seed)    # numpy random seed
torch.backends.cudnn.deterministic = True

# get the mouse network
net_name = 'network_(2,64,64)'
architecture = Architecture(data_folder=DATA_FOLDER)
net = gen_network(net_name, architecture)
mousenet = MouseNet(net, mask=0, bn=1)
mousenet.to(device)

optimizer = optim.SGD(mousenet.parameters(), lr=config.lr,
                          momentum=config.momentum)
 # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
wandb.watch(mousenet, log="all") 

def adjust_learning_rate(config, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = init_lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    config.update({'lr': lr}, allow_val_change=True)

for epoch in range(1, config.epochs + 1):  # loop over the dataset multiple times
    adjust_learning_rate(config, optimizer, epoch)
    train(config, mousenet, device, train_loader, optimizer, epoch)
    test(config, mousenet, device, test_loader)  
    print(epoch)
    print(config.lr)

# WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
torch.save(mousenet.state_dict(), "./myresults/model.h5")
wandb.save('./myresults/model.h5')

print('Finished Training')