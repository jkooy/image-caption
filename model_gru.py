import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    '''
    The class for CNN network part, inhereted from the nn.Module
    '''
    def __init__(self, embed_size):
        
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        self.resnet = nn.Sequential(*modules) # setup our new resnet layer
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) # setup the new linear layer
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01) # add batch normalization
        
    def forward(self, images):
        
        with torch.no_grad():
            features = self.resnet(images) # the forward propagation (requires no gradient calculation)
        features = features.reshape(features.size(0), -1) # flatten 
        features = self.bn(self.linear(features)) # batch normalization
        return features


class DecoderRNN(nn.Module):
    '''
    The class for RNN network part, inhereted from the nn.Module
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # the embedding function 
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True) 
        # we just use the GRU network structure provided by pytorch
        self.linear = nn.Linear(hidden_size, vocab_size) # the linear 
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        '''
        this function is for running forward propagation through the whole network
        '''
        embeddings = self.embed(captions) # first embed the caption
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # concatenate the feature and the embeddings
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) # make them padded
        hiddens, _ = self.gru(packed) # get the hidden output of the gru unit
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        '''
        This function utilize sampling to get the sentence results from the feature input from the CNN
        '''
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.gru(inputs, states) # get the output from the gru unit         
            outputs = self.linear(hiddens.squeeze(1)) # run through the linear layer         
            _, predicted = outputs.max(1) # select the largest                 
            sampled_ids.append(predicted) # record the first word
            inputs = self.embed(predicted) # the input to the next lstm unit is generated                       
            inputs = inputs.unsqueeze(1)                         
        sampled_ids = torch.stack(sampled_ids, 1)                
        return sampled_ids