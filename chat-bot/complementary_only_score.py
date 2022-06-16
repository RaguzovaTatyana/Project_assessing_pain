import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


top_path = os.path.join('1.jpg')
bot_path = os.path.join('2.jpg')
shoe_path = os.path.join('3.jpg')
bag_path = os.path.join('4.jpg')
acc_path = os.path.join('5.jpg')
base_imgs = {'top': top_path, 'bot': bot_path, 'shoe': shoe_path, 'bag': bag_path, 'acc': acc_path}


def score_outfit(input_clothes, model):
    clothes = input_clothes
    elements = {'top', 'bot', 'shoe', 'bag', 'acc'}
    elements_to_add = elements - set(clothes.keys()) 
    for element in elements_to_add:
        clothes[element] = Image.fromarray(np.uint8(np.full((224, 224, 3), 255/2)), "RGB")
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    outfit_tensor = []
    for k, v in clothes.items():
        tensor = my_transforms(v)
        outfit_tensor.append(tensor.squeeze())
    outfit_tensor = torch.stack(outfit_tensor)
    outfit_tensor.unsqueeze_(0)
    with torch.no_grad():
        out = model._compute_score(outfit_tensor)
        score = out[0]
    return score.numpy()[0][0]

