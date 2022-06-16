import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = 'cuda'

result_dir = 'output_images'
checkpoint_path = 'cloth_segm_u2net_latest.pth'


def process_photo(pic_path):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    result = {}

    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.eval()

    img = Image.open(pic_path).convert('RGB')
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor)
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    cl_1 = [[[0 for _ in range(3)] for _ in range(len(output_arr))] for _ in range(len(output_arr))]
    cl_2 = [[[0 for _ in range(3)] for _ in range(len(output_arr))] for _ in range(len(output_arr))]
    cl_3 = [[[0 for _ in range(3)] for _ in range(len(output_arr))] for _ in range(len(output_arr))]
    res = []
    for i in range(len(output_arr)):
        for j in range(len(output_arr[i])):
            if output_arr[i][j] == 1:
                cl_1[i][j] = [1, 1, 1]
                res += [1]
            elif output_arr[i][j] == 2:
                cl_2[i][j] = [1, 1, 1]
                res += [2]
            elif output_arr[i][j] == 3:
                cl_3[i][j] = [1, 1, 1]
                res += [3]

    if 1 in res:
        mid_img = np.asanyarray(img)
        res_arr = np.asarray(cl_1)
        mid_img = np.multiply(mid_img, res_arr)
        output_img = Image.fromarray(mid_img.astype('uint8'))
        output_img = output_img.resize(img_size, Image.BICUBIC)
        result['top'] = output_img

    if 2 in res:
        mid_img = np.asanyarray(img)
        res_arr = np.asarray(cl_2)
        mid_img = np.multiply(mid_img, res_arr)
        output_img = Image.fromarray(mid_img.astype('uint8'))
        output_img = output_img.resize(img_size, Image.BICUBIC)
        result['bot'] = output_img

    if 3 in res:
        mid_img = np.asanyarray(img)
        res_arr = np.asarray(cl_3)
        mid_img = np.multiply(mid_img, res_arr)
        output_img = Image.fromarray(mid_img.astype('uint8'))
        output_img = output_img.resize(img_size, Image.BICUBIC)
        result['bot'] = output_img

    return result
