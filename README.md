# Image Captioning
## Description 
===========================<br> 
This is a image captioning project written in pytorch by team A+A+A+A+ composed of: 

*Xuehai He(A53303907)<br>*
*Yang Ni (A53295687)<br>*
*Zhijin Liang (A53311894)<br>*
*Xuechen Zhu(A53311894)<br>*

Our network used in this project has a structure of CNN+RNN, and it is based on the Show and Tell paper.
For the CNN part, we tried resnet152 and resnet50. For the RNN part, we tried LSTM and GRU.

## Requirements
===========================<br>
- Python 3.6+
- PyTorch 1.3.1
- Matplotlib
- Pillow
- Numpy

- These could be installed by:
```
python -m pip install --user torch
python -m pip install --user matplotlib
python -m pip install --user numpy
```

## Code organization
===========================<br>
### example_pics
folder for pictures used in inference demo
### notebook_plot_loss
in this folder, there's a notebook for plot the validation and training loss
### folders for trained models and results
- resnet152+gru+hidden_size512+lr_1e-3
- resnet152+lstm+hidden_size512+lr_1e-3
- resnet152+lstm+hidden_size512+lr_1e-2
- resnet152+lstm+hidden_size1024+lr_1e-3
- resnet50+lstm+hidden_size1024+lr_1e-3
- We change the CNN, RNN, hidden size, learning rate
### Result_generate_testset.ipynb
notebook for run the trained model on whole test set
### Result_generate_valset.ipynb
notebook for run the trained model on whole validation set
### checkDataset.ipynb
notebook for check the dataset&dataloader performance, it shows some of the pictures and captions
### demo_for_inference.ipynb
this is the demo for inference, it will test the 7 example pictures and generate the captions
### demo_for_train.ipynb
this is the demo for training the 'resnet152+lstm+hidden_size512+lr_1e-3' network
### vocab.pkl
this is the dictionary that encodes the word
### train.py
main file for training
### model.py
lstm+CNN model construction
### model_gru.py
gru+CNN model construction
### utils.py
some useful functions
### data_loader.py
load the training and validation set
### sample.py
get captions when running forward through the network
### build_vocab.py
build the dictionary
