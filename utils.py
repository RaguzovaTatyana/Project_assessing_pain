import zipfile
import random
import torch

def unzip_folder(folder_path, target_folder):
    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)


def open_file(name):
    f = open(name + '.txt', "r")
    scorestr = f.read()
    f.close()
    return scorestr


def get_frame(scorestr):
    return float(scorestr[0:scorestr.find('e')]) * (10 ** int(scorestr[scorestr.find('+') + 1:]))


def set_rseed(rseed):
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    np.random.seed(rseed)
    random.seed(rseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

