import torchvision
import torch

def resize_tensor(input_tensors, h, w):
    batch_size, channel, height, width = input_tensors.shape
    output = None
    for c in range(channel):
        re = resize_one_channel_tensor(
             torch.unsqueeze(input_tensors[:, c, :, :],1), h, w)
        if output is None:
            output = re
        else:
            output = torch.cat((output, re), 1)
    return output

def resize_one_channel_tensor(input_tensors, h, w):
    final_output = None
    batch_size, channel, height, width = input_tensors.shape
    input_tensors = torch.squeeze(input_tensors, 1)
  
    for img in input_tensors:
        img_PIL = torchvision.transforms.ToPILImage()(img)
        img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
        img_PIL = torchvision.transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
    return final_output

