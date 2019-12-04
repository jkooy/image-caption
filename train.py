import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import utils


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing and normalization
    transform = transforms.Compose([ 
        transforms.Resize((240, 240)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary list
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # data loader for train and validation
    train_data_loader = get_loader(args.train_image_dir, args.train_caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    val_data_loader = get_loader(args.val_image_dir, args.val_caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    losses_train = []
    losses_val = []
    
    ############ Load the trained model parameters
    if args.model_file:
        encoder_state_dict, decoder_state_dict, optimizer, *meta = utils.load_models(args.model_file)
        initial_step, initial_epoch, losses_train, losses_val = meta
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)
    else:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            print("Training loss: {}".format(loss.item()))
            
            losses_train.append(loss.item())
            
            loss.backward()
            optimizer.step()
                           
            ############################################################################################
            # check the performance on validation dataset
            if i % args.log_step == 0:
                # change to val mode
                encoder.eval()
                decoder.eval()

                batch_losses_val = []
                
                with torch.no_grad():
                    for val_step, (images, captions, lengths) in enumerate(val_data_loader):
                        
                        # forward 
                        if (val_step < 40): # only use the first 40 mini-batch to compute validation loss(to save time)
                            images = images.to(device)
                            captions = captions.to(device)
                            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                            features = encoder(images)
                            outputs = decoder(features, captions, lengths)
                            batch_loss_val = criterion(outputs, targets) 
                            
                            print("Val loss: {}".format(batch_loss_val.item()))
                            batch_losses_val.append(batch_loss_val.item()) 
                        else:
                            break

                    # compute the mean value across the whole batch
                    mean_losses_val = np.mean(batch_losses_val)
                    #print("Avg loss: {}".format(mean_losses_val))
                    losses_val.append(mean_losses_val)
                    

                # change to train mode
                encoder.train()
                decoder.train()
                
            ############################################################################################
            # Print log info
            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f},Validation Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, losses_train[-1],losses_val[-1])) 
            
           #############################################################################################              
            # Save the model
            if (i+1) % args.save_step == 0:
                utils.save_models(encoder, decoder, optimizer, i, epoch, losses_train, losses_val, args.model_path)
                
            # Save the model training loss and validation
            if (i+1) % args.log_step == 0:   # save the loss for each mini-batch
                utils.dump_losses(losses_train, losses_val, os.path.join(args.model_path, 'losses.pkl'))
                

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./demo_train_SavedModel' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=240 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='/datasets/COCO-2015/train2014', 
                        help='directory for resized images')
    parser.add_argument('--val_image_dir', type=str, default='/datasets/COCO-2015/val2014', 
                        help='directory for resized images')
    parser.add_argument('--train_caption_path', type=str,
                        default='/datasets/ee285f-public/COCO-Annotations/annotations_trainval2014/captions_train2014.json', 
                        help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str,
                        default='/datasets/ee285f-public/COCO-Annotations/annotations_trainval2014/captions_val2014.json', 
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=200, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=200, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)   
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    parser.add_argument('--model_file', type=str, default=None, help='path for trained encoder')
    args = parser.parse_args()
    print(args)
    main(args)