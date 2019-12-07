import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    '''
    This is the function for load the testing image
    '''
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS) # we do the resize for the image
    
    if transform is not None:
        image = transform(image).unsqueeze(0) # do the transformation
    
    return image

def main(args):
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))]) # normalize the image according to the ImageNet guidance
    
    # read the vocab file
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    
    encoder = EncoderCNN(args.embed_size).eval() # setup the encoder network
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers) # the decoder network
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    image = load_image(args.image, transform) # load the image to tensor
    image_tensor = image.to(device)
    print(image_tensor.size())
    feature = encoder(image_tensor) # runing forward propagate through the encoder
    sampled_ids = decoder.sample(feature) # the output CNN feature goes through the sampling of the RNN network
    sampled_ids = sampled_ids[0].cpu().numpy()          
    # translate the index back to word and construct the sentence
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word != '<end>' and word != '<start>':
            sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print (sentence)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='resnet152+lstm+hidden_size512+lr_1e-3/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='resnet152+lstm+hidden_size512+lr_1e-3/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    
    
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)