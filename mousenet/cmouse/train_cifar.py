import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from config import *
from train_config import *
from anatomy import *
from network import *
from mousenet_complete_pool import MouseNetCompletePool
from fsimilarity import *
import random
import wandb
import argparse

parser = argparse.ArgumentParser(description='PyTorch %s Training' % DATASET)
parser.add_argument('--seed', default = 42, type=int, help='random seed')
parser.add_argument('--mask', default = 3, type=int, help='if use Gaussian mask')
args = parser.parse_args()
SEED = args.seed
MASK = args.mask
RUN_NAME_MASK = 'mask_%s_%s'%(MASK, RUN_NAME)
RUN_NAME_MASK_SEED = '%s_seed_%s'%(RUN_NAME_MASK, SEED)

def train(args, model, device, train_loader, optimizer, epoch):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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

     
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        running_loss += loss.item()
        if batch_idx % LOG_INTERVAL == LOG_INTERVAL-1:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx + 1, running_loss / LOG_INTERVAL))
            if USE_WANDB:
                wandb.log({
                "running loss": running_loss/LOG_INTERVAL,
                "running acc": 100.*correct/total})
            running_loss = 0.0            

def test(args, model, device, test_loader, epoch):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    #example_images = []
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
            # WandB - Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            #example_images.append(wandb.Image(
            #    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    # Save checkpoint.
    global best_acc
    acc = 100. * correct / len(test_loader.dataset)
    if epoch == 0 or acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': model.state_dict(),
            'best_acc1': acc,
            'epoch': epoch,
        }
        save_dir = RESULT_DIR + '/' + RUN_NAME_MASK +'/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if epoch == 0:
            torch.save(state, save_dir + '%s_init.pt'%(SEED))
        else:
            torch.save(state, save_dir + '%s_%s.pt'%(SEED, acc))
        best_acc = acc

    # WandB - wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    if USE_WANDB:
        wandb.log({
            #"Examples": example_images,
             "Test Accuracy": acc,
             "Test Loss": test_loss})

    #fsim = get_functional_similarity(model, layer_maps, device, is_mouse=1)
    #wandb.log(fsim)

        
config = None
if USE_WANDB:
    # WandB - Initialize a new run
    wandb.init(dir=WANDB_DIR, name=RUN_NAME_MASK_SEED, project=PROJECT_NAME)
    # WandB - Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.batch_size = BATCH_SIZE         # input batch size for training (default: 64)
    config.test_batch_size = BATCH_SIZE     # input batch size for testing (default: 1000)
    config.epochs = EPOCHS          # number of epochs to train (default: 10)
    config.momentum = MOMENTUM         # SGD momentum (default: 0.5) 
    config.no_cuda = False         # disables CUDA training
    config.seed = SEED             # random seed (default: 42)
    config.log_interval = LOG_INTERVAL   # how many batches to wait before logging training status
    config.lr = LR               # learning rate (default: 0.01)

best_acc = 0 
# preparing input transformation
if INPUT_SIZE[0]==3:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
                        transforms.Resize(INPUT_SIZE[1:]),transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
elif INPUT_SIZE[0]==2:
    class GBChannels():
        def __call__(self, tensor):
            return tensor[1:3,:,:]
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),
                            GBChannels(),
                            transforms.Normalize((0.4914, 0.4822), (0.2023, 0.1994))])
    transform_test = transforms.Compose([transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),
                            GBChannels(),
                            transforms.Normalize((0.4914, 0.4822), (0.2023, 0.1994))])
else:
    raise Exception('Number of input channel should be 2 or 3!')

# load dataset
if DATASET == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                    download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=transform_test)

elif DATASET == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                    download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                    download=True, transform=transform_test)
else:
    raise Exception('DATASET should be cifar10 or cifar100')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=2)


device = torch.device("cuda")
# print the shape of the input
for i, data in enumerate(train_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)
    print("The shape of input is %s"%str(inputs.shape))
    break

# Set random seeds and deterministic pytorch for reproducibility
random.seed(SEED)       # python random seed
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED)    # numpy random seed
torch.backends.cudnn.deterministic = True

# get the mouse network
net_name = 'network_(%s,%s,%s)'%(INPUT_SIZE[0],INPUT_SIZE[1],INPUT_SIZE[2])
architecture = Architecture()
net = gen_network(net_name, architecture)
mousenet = MouseNetCompletePool(net, mask=MASK)

mousenet.to(device)

optimizer = optim.SGD(mousenet.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
# WandB - wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
# Using log="all" log histograms of parameter values in addition to gradients
#if not WANDB_DRY:
#    wandb.watch(mousenet, log="all") 

def adjust_learning_rate(config, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every LR_EPOCHS"""
    lr = LR * (0.1 ** (epoch // LR_EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if USE_WANDB:
        config.update({'lr': lr}, allow_val_change=True)

test(config, mousenet, device, test_loader, 0)  
for epoch in range(1, EPOCHS + 1):  # loop over the dataset multiple times
    adjust_learning_rate(config, optimizer, epoch)
    print(epoch)     
    train(config, mousenet, device, train_loader, optimizer, epoch)
    test(config, mousenet, device, test_loader, epoch)  
    #break

# WandB - Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
if USE_WANDB:
    if not WANDB_DRY:
        wandb.save(WANDB_DIR+RUN_NAME_MASK_SEED+'.h5')

print('Finished Training')
