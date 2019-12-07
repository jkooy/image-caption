import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    '''
    This is a class for coco dataset that inheret from the torch dataset class
    '''
    def __init__(self, root, json, vocab, transform=None):
        '''
        Some properties of this dataset class
        '''
        self.root = root
        self.coco = COCO(json) # decode the caption json file
        self.ids = list(self.coco.anns.keys()) # the list of all the annotation keys
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        '''
        Define how each item in the dataset is formed
        '''
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption'] # the corresponding caption
        img_id = coco.anns[ann_id]['image_id'] # the image id
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB') # open the image
        # do the image transformation if possible
        if self.transform is not None:
            image = self.transform(image)
        # tokenize the caption
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        # perform word to index
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        '''
        Return the total length of this dataset
        '''
        return len(self.ids)


def collate_fn(data):
    '''
    This function customize the dataset, pads the caption to the same length and generate mini batches
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    images = torch.stack(images, 0) # stack all the image tensors

    lengths = [len(cap) for cap in captions]
    # padding the caption to the required length
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    '''
    The main function for setup the data loader
    '''
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader