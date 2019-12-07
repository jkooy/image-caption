import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    
    def __init__(self):
        '''
        This sets up the basic property of the Vocabulary class
        '''
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        '''
        This is the function for adding words and corresponding encodings to the dictionary
        '''
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        '''
        This is in effect when the class is called, and the encoded index will be returned
        '''
        if not word in self.word2idx:
            # if the word is not in the dictionary, then we assign a 'unk' to it
            return self.word2idx['<unk>']
        # or we return the regular word
        return self.word2idx[word]

    def __len__(self):
        '''
        This function will return the length
        '''
        return len(self.word2idx)

def build_vocab(json, threshold):
    '''
    This is the function to build the vocabulary dictionary used to encode the word shown up in the dataset
    '''
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption']) # read the caption out of the json file
        tokens = nltk.tokenize.word_tokenize(caption.lower()) # slice the caption into words 
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # if the word frequency is above the threshold, record it
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    
    vocab = Vocabulary()

    # add all the regular words
    for i, word in enumerate(words):
        vocab.add_word(word)
    # add four special words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    return vocab

def main(args):
    '''
    the main function for building the vocab.json file
    '''
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/datasets/ee285f-public/COCO-Annotations/annotations_trainval2014/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)