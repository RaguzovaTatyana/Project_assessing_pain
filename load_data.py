import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import UNBCMcMasterDataset
import imp
imp.reload(UNBCMcMasterDataset)
from UNBCMcMasterDataset import *
import os
import utils

def load_dataset():
    for rseed in [0, 2, 4, 6, 8]:
        utils.set_rseed(rseed)

        image_dir = "./detected"
        label_dir = "./"
        if not os.path.isdir('./models_sf' + str(rseed)):
            os.mkdir('./models_sf' + str(rseed))
        if not os.path.isdir('./results_sf' + str(rseed)):
            os.mkdir('./results_sf' + str(rseed))

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets

        # load subject folder names from root directory
        subjects = []
        for directory in next(os.walk(image_dir))[1]:
            subjects.append(directory[:3])
        subjects = sorted(subjects)
        random.shuffle(subjects)
        print(subjects)

        fold_size = 5
        folders = []
        for i in range(5):
            folders += [subjects[i * fold_size: (i+1) * fold_size]]

        # start cross validation
        for subj_left_id, subj_left_out in enumerate(folders):
            utils.set_rseed(rseed)
            test_subj = subj_left_out
            train_id= range(len(folders))
            train_id.pop(subj_left_id)
            val_id = random.choice(train_id)
            val_subj = folders[val_id]

            if os.path.isfile('./results_sf' + str(rseed) + '/' + str(subj_left_id) + '.npz'):
                continue

            print('-'*10 + "cross-validation: " + "(" + str(subj_left_id+1) + "/5)" + '-' * 10)

            datasets = {x: UNBCMcMasterDataset(image_dir, label_dir, val_subj, test_subj, x, data_transforms[x]) for x in ['train', 'val', 'test']}
            
            # Create training and validation dataloaders
            weights = {}
            for phase in ['train', 'val', 'test']:
                labels = [x['framePSPI'] for x in datasets[phase]]
                labels = np.stack(labels)
                classes, classweights = np.unique(labels, return_counts=True)
                classweights = np.reciprocal(classweights.astype(float))
                sampleweights = classweights[np.searchsorted(classes, labels)]
                classweights = classweights * sampleweights.shape[0] / np.sum(sampleweights) 
                weights[phase] = {'classes':classes, 'classweights': classweights}
        
            shuffle = {'train': True, 'val': False, 'test': False}
            dataloaders_dict = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=4, worker_init_fn=lambda l: [np.random.seed((rseed + l)), random.seed(rseed + l), torch.manual_seed(rseed+ l)]) for x in ['train', 'val', 'test']}
    return dataloaders_dict