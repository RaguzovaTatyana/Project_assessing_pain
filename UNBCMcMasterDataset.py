import torch
from torch.utils.data import Dataset
import torch
import os
import cv2
import numpy as np
# from utils import utils.open_file, utils.get_frame
import utils

class UNBCMcMasterDataset(Dataset):
    def __init__(self, image_dir, label_dir, val_subj_id, test_subj_id, subset, transform=None):
        """
        Args:
            image_dir (string): Path to the image data "detected"
            label_dir (string): Path to the label (pain level, etc.) (root dir)
            val_subj_id ([string]): list of paths containing validation data
            test_subj_id ([string]): list of paths containing test data
            subset (string): train, val, test
            transform (callable, optional): Optional transfomr to be applied on a sample
        """
        self.seqVASpath = os.path.join(label_dir, 'Sequence_Labels','VAS')
        self.frameVASpath = os.path.join(label_dir, 'Frame_Labels','PSPI')
        self.AUpath = os.path.join(label_dir, 'Frame_Labels', 'FACS')
        self.AAMpath = os.path.join(label_dir, 'AAM_landmarks')
        self.imagepath = image_dir
        self.image_files = []
        for root, dirs, files in os.walk(self.imagepath):
            for name in sorted(files):
                if name[-3:]=='png' and ((name[2:5] in test_subj_id and subset=='test') or (name[2:5] in val_subj_id and subset=='val') or (not(name[2:5] in val_subj_id+test_subj_id) and subset=='train')):
                    self.image_files.append((root, name))
        self.transform = transform

    def __len__(self):
        return sum([len(self.image_files)])

    def __getitem__(self, idx):
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
        img_dir = self.image_files[idx][0]
        img_name = self.image_files[idx][1]
        image = cv2.imread(os.path.join(img_dir, img_name))
        video_dir = os.path.split(img_dir)
        video_id = video_dir[1]
        subj_dir = os.path.split(video_dir[0])
        subj_id = subj_dir[1]

        name = os.path.join(self.seqVASpath, subj_id, video_id)
        scorestr = utils.open_file()
        videoVAS = utils.get_frame(scorestr)
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'SEN', subj_id, video_id)
        scorestr = utils.open_file()
        videoSEN = utils.get_frame(scorestr)
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'OPR', subj_id, video_id)
        scorestr = utils.open_file()
        videoOPR = utils.get_frame(scorestr)
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'AFF', subj_id, video_id)
        scorestr = utils.open_file()
        videoAFF = utils.get_frame(scorestr)
        # framePSPI
        name = os.path.join(self.frameVASpath, subj_id, video_id, img_name[:-4] + '_facs')
        scorestr = utils.open_file()
        framePSPI = utils.get_frame(scorestr)   
        framelabel = 0+(framePSPI > 0)

        # frameAU
        name = os.path.join(self.AUpath, subj_id, video_id, img_name[:-4] + '_facs')
        scorestr = utils.open_file()
        scorestr = [x.strip() for x in scorestr]
        au = np.zeros((64,))
        for line in scorestr:
            words = [x.strip() for x in line.split(' ') if x]
            aunumberstr = words[0]
            auintensitystr = words[1]
            aunumber = utils.get_frame(auintensitystr)
            auintensity = utils.get_frame(aunumber)
            au[int(aunumber)-1] = auintensity

        au = au[[3,5,6,9,11,19,24,25,42]]

        # frameAAM
        name = os.path.join(self.AAMpath, subj_id, video_id, img_name[:-4] + '_aam')
        scorestr = utils.open_file()
        aam = []
        for line in scorestr:
            words = [x.strip() for x in line.split(' ') if x]
            aam = aam + [float(words[0]), float(words[1])]
        aam = np.stack(aam)


        sample = {'image': image, 'image_id': img_name, 'video_id': video_id, 'au': au, 'aam': aam,
            'framelabel': framelabel, 'subj_id': subj_id, 'videoVAS': videoVAS, 
            'videoAFF': videoAFF, 'videoOPR': videoOPR, 
            'videoSEN': videoSEN,'framePSPI': framePSPI}
        if self.transform:
            sample['image'] = self.transform(image)

        return sample


class BGR2RGB(object):
    """Convert BGR image to RGB.
    """

    def __call__(self, image):
        image = image[:,:,::-1]

        return image
