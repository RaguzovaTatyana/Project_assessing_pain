from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import cv2
import numpy as np
from utils import open_file, get_frame

def load_dataset(image_dir, label_dir):
    """
    Args:
        image_dir (string): Path to the image data "UNBCMcMaster_cropped/Images0.3"
        label_dir (string): Path to the label (pain level, etc.) "UNBCMcMaster"
        val_subj_id ([string]): list of paths containing validation data
        test_subj_id ([string]): list of paths containing test data
        subset (string): train, val, test
    """
    seqVASpath = os.path.join(label_dir, 'Sequence_Labels','VAS')
    frameVASpath = os.path.join(label_dir, 'Frame_Labels','PSPI')
    AUpath = os.path.join(label_dir, 'Frame_Labels', 'FACS')
    AAMpath = os.path.join(label_dir, 'AAM_landmarks')
    for root, dirs, files in os.walk(image_dir):
      print("root" + root)
      print("dirs" + dirs)
      print("files" + files)

def getItem(idx, image_files, seqVASpath, AUpath, AAMpath, frameVASpath, transform=None):
    """
    Return: sample
        an example of sample:
            sample['image'] = np.ndarray WxHx3
            sample['image_id'] = 'gf097t1aaaff001.png'
            sample['video_id'] = 'gf097t1aaaff'
            sample['au'] = np.ndarray (9,)
            sample['fraamelabel'] = 0
            sample['subj_id'] = '097-gf097'
            sample['videoVAS] = 5.0
            sample['framePSPI'] = 0.0
            sample['aam'] = np.ndarray (132,)
    """
    img_dir = image_files[idx][0]
    img_name = image_files[idx][1]
    image = cv2.imread(os.path.join(img_dir, img_name))
    video_dir = os.path.split(img_dir)
    video_id = video_dir[1]
    subj_dir = os.path.split(video_dir[0])
    subj_id = subj_dir[1]

    name = os.path.join(seqVASpath, subj_id, video_id)
    scorestr = open_file(name)
    videoVAS = get_frame(scorestr)

    name = os.path.join(os.path.split(seqVASpath)[0], 'SEN', subj_id,video_id)
    scorestr = open_file(name)
    videoSEN = get_frame(scorestr)
    
    name = os.path.join(os.path.split(seqVASpath)[0], 'OPR', subj_id,video_id)
    scorestr = open_file(name)
    videoOPR = get_frame(scorestr)
    
    name = os.path.join(os.path.split(seqVASpath)[0], 'AFF', subj_id, video_id)
    scorestr = open_file(name)
    videoAFF = get_frame(scorestr)

    # framePSPI
    name = os.path.join(frameVASpath, subj_id, video_id, img_name[:-4] + '_facs')
    scorestr = open_file(name)
    framePSPI = get_frame(scorestr)   
    framelabel = 0 + (framePSPI > 0)

    # frameAU
    name = os.path.join(AUpath, subj_id, video_id, img_name[:-4] + '_facs')
    scorestr = open_file(name)
    scorestr = [x.strip() for x in scorestr]
    au = np.zeros((64,))
    for line in scorestr:
        words = [x.strip() for x in line.split(' ') if x]
        aunumberstr = words[0]
        auintensitystr = words[1]
        aunumber = get_frame(auintensitystr)      
        auintensity = get_frame(aunumber)        
        au[int(aunumber) - 1] = auintensity

    au = au[[3, 5, 6, 9, 11, 19, 24, 25, 42]]

    # frameAAM
    name = os.path.join(AAMpath, subj_id, video_id, img_name[:-4] + '_aam')
    scorestr = open_file(name)
    aam = []
    for line in scorestr:
        words = [x.strip() for x in line.split(' ') if x]
        aam = aam + [float(words[0]), float(words[1])]
    aam = np.stack(aam)


    sample = {'image': image, 'image_id': img_name, 'video_id': video_id, 'au': au, 'aam': aam,
        'framelabel': framelabel, 'subj_id': subj_id, 'videoVAS': videoVAS, 'videoAFF': videoAFF, 
        'videoOPR': videoOPR, 'videoSEN': videoSEN,'framePSPI': framePSPI}
    if transform:
        sample['image'] = transform(image)

    return sample


def BGR2RGB(image):
    """
    Convert BGR image to RGB.
    """
    return image[:,:,::-1]
