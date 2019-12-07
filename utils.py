import torch
import os


def save_models(encoder, decoder, optimizer, step, epoch, losses_train, losses_val, checkpoint_path):
    '''
    This function is for save the checkpoint during the training process
    '''
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file = os.path.join(checkpoint_path, 'model-%d-%d.ckpt' %(epoch+1, step+1))
    print('Saving model to:', checkpoint_file)
    torch.save({
        'encoder_state_dict': encoder.state_dict(), # save the encoder parameters
        'decoder_state_dict': decoder.state_dict(), # save the decoder parameters
        'optimizer': optimizer,
        'step': step,
        'epoch': epoch,
        'losses_train': losses_train, # save the training loss
        'losses_val': losses_val # save the validation loss
        }, checkpoint_file)

def load_models(checkpoint_file,sample=False):
    '''
    This function is for load the checkpoints into the model for further training
    '''
    if sample:
        checkpoint = torch.load(checkpoint_file,map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_file)
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    optimizer = checkpoint['optimizer']
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    return encoder_state_dict, decoder_state_dict, optimizer, step, epoch, losses_train, losses_val

def dump_losses(losses_train, losses_val, path):
    '''
    This function saves the train loss and validation loss to files
    '''
    import pickle
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'wb') as f:
        try:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f, protocol=2)
        except:
            pickle.dump({'losses_train': losses_train, 'losses_val': losses_val}, f)
