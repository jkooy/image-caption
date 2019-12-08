# Image Captioning
## Description 
===========================<br> 
This is a image captioning project written in pytorch by team A+A+A+A+ composed of: 

***Xuehai He(A53303907)<br>***
***Yang Ni (A53295687)<br>***
***Zhijin Liang (A53311894)<br>***
***Xuechen Zhu(A53314896)<br>***

Our network used in this project has a structure of CNN+RNN, and it is based on the Show and Tell paper.
For the CNN part, we tried resnet152 and resnet50. For the RNN part, we tried LSTM and GRU.

## Results
===========================<br>
- The best result is produced by the second model below


resnet152+lstm+hidden_size1024+lr_1e3: Bleu_4_C5 = 0.235378 CIDEr_C5 = 0.748013

**resnet152+lstm+hidden_size512+lr_1e3: Bleu_4_C5 = 0.242659 CIDEr_C5 = 0.772517**

resnet152+gru+hidden_size512+lr_1e3: Bleu_4_C5 = 0.235254 CIDEr_C5 = 0.750898

resnet152+gru+hidden_size1024+lr_1e3: Bleu_4_C5 = 0.234776 CIDEr_C5 = 0.749187

resnet50+lstm+hidden_size1024+lr_1e3: Bleu_4_C5 = 0.237044 CIDEr_C5 = 0.749605

resnet152+lstm+hidden_size512+lr_1e-2: Bleu_4_C5 = 0.222249 CIDEr_C5 = 0.695748

## Requirements
===========================<br>
- Python 3.6+
- PyTorch 1.3.1
- Matplotlib
- Pillow
- Numpy
- NLTK
- pycocotools
- And some other library included in regular python

- These could be installed on DSMLP server by:
```
python -m pip install --user torch
python -m pip install --user matplotlib
python -m pip install --user numpy
python -m pip install --user pycocotools
python -m pip install --user nltk
```

## Code organization
===========================<br>
### example_pics
folder for pictures used in inference demo
### notebook_plot_loss
in this folder, there's a notebook for plot the validation and training loss
### folders for trained models and results
- Some models may not include the trained model files because the storage limit of GitHub
- But all models has the result and the eval_score, and we have DropBox link for the model files that exceed limit

- resnet152+gru+hidden_size512+lr_1e-3
- resnet152+gru+hidden_size1024+lr_1e-3
- resnet152+lstm+hidden_size512+lr_1e-3
- resnet152+lstm+hidden_size512+lr_1e-2
- resnet152+lstm+hidden_size1024+lr_1e-3
- resnet50+lstm+hidden_size1024+lr_1e-3
	- We change the CNN (resnet152 and resnet50), RNN (LSTM and GRU), hidden size (512 and 1024), learning rate (1e-2 and 1e-3)
	- Under each folder, there might be three folders (models, result_json and eval_score) for: 
		- storing trained models
		- storing results
		- storing scores

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
