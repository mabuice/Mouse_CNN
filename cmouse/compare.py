from network import *
import simplenet
import xmetric_torch as xm
import torchvision.transforms as transforms
from config import INPUT_SIZE, DATA_DIR
import numpy as np
import pandas as pd
from mouse_cnn.architecture import *
import argparse
import random
parse = argparse.ArgumentParser()
parse.add_argument('--net', type=str, default='network_(2,64,64)', help='specifiy a network as yard stick')
parse.add_argument('--ismouse', type=int, default=1, help='is it comparing with mouse net?')
# parse.add_argument('--issimple', type=int, default=0, help='is it comparing with simple mouse net?')
parse.add_argument('--mask', type=int, default=1, help='add Gaussian mask')
parse.add_argument('--nchannels', type=int, default=2, help='number of input channels')
parse.add_argument('--size', type=int, default=64, help='size of input images')
parse.add_argument('--netw', type=str, default=None, help='specify a filename for network weights')
parse.add_argument('--method', type=str, default='ssm_pcc', help='metric')
parse.add_argument('--neuron', type=int, default=None, help='number of neurons')
parse.add_argument('--ndraws', type=int, default=1, help='number of draws of neurons')
parse.add_argument('--config', type=str, default='region')
parse.add_argument('--folder', type=str, default=None)
parse.add_argument('--seed', default=42, type=int, help='random seed')
args = parse.parse_args()

net_name = args.net
mask = args.mask
net_weight = args.netw
method = args.method
num_neuron = args.neuron
num_draws = args.ndraws
fconfig = args.config
seed = args.seed
size = INPUT_SIZE[1]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stim = xm.data.get_stim_raw()

def transform_one_image(img, size):
    t = transforms.ToTensor()(img) 
    t = t.expand(args.nchannels,t.size(1),t.size(2))
    t = transforms.ToPILImage()(t)
    t = transforms.Resize((args.size, args.size))(t)
    t = transforms.ToTensor()(t)
    if args.nchannels == 2:
        t = transforms.Normalize((0.5, 0.5),(0.5, 0.5))(t)
    elif args.nchannels == 3:
        t = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(t)
    return t

if args.ismouse == 1:
    # get the mouse network
    architecture = Architecture(data_folder=DATA_DIR)
    net = gen_network(net_name, architecture)
    mousenet = MouseNet(net, mask=mask, bn=1) 
    if net_weight is not None:
        f = torch.load(net_weight, map_location=device)
        # mousenet.load_state_dict(f['net'])
        mousenet.load_state_dict(f['state_dict'])
        best_acc1 = f['best_acc1']
    net1 = mousenet
    layer_maps = {0: 'LGNv',
                1: 'VISp4', 2:'VISp2/3', 3:'VISp5',
                4:'VISl4', 5:'VISl2/3', 6:'VISl5', 
                7:'VISal4', 8:'VISal2/3', 9:'VISal5', 
                10:'VISrl4', 11:'VISrl2/3', 12:'VISrl5',
                13:'VISpor4', 14:'VISpor2/3', 15:'VISpor5'}
    
    imgs = transform_one_image(stim[0], size)[None,:,:,:]
    for i in range(1, len(stim)):
        img_i = transform_one_image(stim[i], size)[None,:,:,:]
        imgs = torch.cat((imgs, img_i), 0)

# elif args.issimple == 1:
#     architecture = Architecture(data_folder=DATA_DIR)
#     net = simplenet.gen_simple_net(net_name, architecture)
#     simplenet = simplenet.SimpleNet(net, mask=mask, bn=1)
#     if net_weight is not None:
#         f = torch.load(net_weight, map_location=device)
#         simplenet.load_state_dict(f['net'])
#     net1 = simplenet
#     layer_maps = {0: 'LGNv',
#                 1: 'VISp4', 2:'VISp2/3', 3:'VISp5',
#                 4:'VISal4', 5:'VISal2/3', 6:'VISal5', 
#                 7:'VISpor4', 8:'VISpor2/3', 9:'VISpor5'}
    
#     imgs = transform_one_image(stim[0], size)[None,:,:,:]
#     for i in range(1, len(stim)):
#         img_i = transform_one_image(stim[i], size)[None,:,:,:]
#         imgs = torch.cat((imgs, img_i), 0)
else:
    random.seed(seed)       # python random seed
    torch.manual_seed(seed) # pytorch random seed
    np.random.seed(seed)    # numpy random seed

    torch.backends.cudnn.deterministic = True
    if net_name.split('_')[0] == 'VGG16':
        if net_name.split('_')[1] == 'pretrained':
            net1 = xm.nn.TorchModel(name='VGG16', is_pretrained=True, size=(size, size))
        if net_name.split('_')[1] == 'trained':
            net1 = xm.nn.TorchModel(name='VGG16', is_pretrained=False, size=(size, size))
            f = torch.load(net_weight, map_location=device)
            net1.net.load_state_dict(f['state_dict'])
        elif net_name.split('_')[1] == 'untrained':
            net1 = xm.nn.TorchModel(name='VGG16', is_pretrained=False, size=(size, size))
        else:
            assert('need to specify whether net is pretrained!')
        layer_maps = {0:-1, 1:1, 2:3, 3:4, 4:6, 5:8,
                    6:9, 7:11, 8:13, 9:15, 10:16, 11:18, 
                    12:20, 13:22, 14:23, 15:25, 16:27, 17:29, 18:30}
        imgs = stim
        
compare_param = {'num_shuffle': 0,
                 'num_bootstrap_col': 0}
compare_param['pca_dimension'] = 40 # for cca
compare_param['dmetric'] = 'pcc' # for ssm

comparator = xm.compare.get_comparator(method, **compare_param)

results = np.zeros([len(layer_maps), 6, num_draws])
def mice_net(data, stim, net1, comparator, num_neuron):
    if args.ismouse or args.issimple:
        net1.eval() # set the model to evaluation mode
    else:
        net1.net.eval()
    for i in layer_maps.keys():
        print(i)
        if args.ismouse or args.issimple:
            layer1 = net1.get_img_feature(stim, [layer_maps[i]])
            info = xm.NetInfo(layer_maps[i], i)
            layer1 = xm.Rep(layer1.detach().numpy(), info, 0)
        else:
            layer1 = net1.get_img_feature(stim, layer_maps[i])
        
        for j, region in enumerate(data.keys()):
            print(region)
            re = []
            for s in range(num_draws):
                d = data[region][:-1, :]
                if num_neuron == None:
                    layer2 = xm.rep.Rep(d, None, False)
                else:
                    layer2 = xm.rep.Rep(d[:, np.random.choice(d.shape[1], num_neuron)], None, False)
                re.append(comparator.compare(layer1, layer2)['score'][0])
            results[i,j, :] = re
            print(results[i,j])

# def mice_mice(data, comparator):
#     for i, region1 in enumerate(data.keys()):
#         d = data[region1][:-1, :]
#         layer1 = xm.rep.Rep(d, None, False)
                
#         for j, region2 in enumerate(data.keys()):
#             d = data[region2][:-1, :]
#             layer2 = xm.rep.Rep(d, None, False)
#             tmp = comparator.compare(layer1, layer2)['score'][0]
#             results[i,j] = tmp
#             print(region1, region2, tmp)
            
data = xm.data.read_neuro_file(DATA_DIR+'/ABO/bob_ns_response_dict_by_%s.pkl'%fconfig)
#mice_mice(data, comparator)
mice_net(data, imgs, net1, comparator, num_neuron)
if args.folder:
    save_folder = RESULT_DIR+'/comparison/%s/'% args.folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    np.save(save_folder + '%s_best_acc1_%s_%s.npy'% (net_weight.split('/')[-1], best_acc1, num_neuron), results)
else:
    save_folder = RESULT_DIR+'/comparison/init_nets/%s_default/'%net_name
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    np.save(save_folder + '%s_eval.npy'%seed, results)





