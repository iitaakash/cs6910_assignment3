import torch
import pandas as pd


MAX_OUTPUT_CHARS = 32
class Aksharantar(torch.utils.data.Dataset):
    tamil_ascii = [ 2946, 2947, 2949, 2950, 2951, 2952, 2953, 2954, 2958, 2959, 2960, 2962, 2963, 
                    2965, 2969, 2970, 2972, 2974, 2975, 2979, 2980, 2984, 2985, 2986, 2990, 
                    2991, 2992, 2993, 2994, 2995, 2996, 2997, 2999, 3000, 3001, 3006, 3007, 
                    3008, 3009, 3010, 3014, 3015, 3016, 3018, 3019, 3020, 3021]
 

    def __init__(self, dataset_dir = "../aksharantar_sampled/tam/", mode = 'train'):
        self.dataset_dir = dataset_dir
        self.mode = mode
    
        if self.mode == 'train' :
            self.dataset_dir = self.dataset_dir + "/tam_train.csv"
        elif self.mode == 'val' or self.mode == 'valid':
            self.dataset_dir = self.dataset_dir + "/tam_valid.csv"
        else:
            self.dataset_dir = self.dataset_dir + "/tam_test.csv"

        self.data = pd.read_csv(self.dataset_dir, sep=',', header= None, names=['english', 'tamil'])
        self.data = self.data.sample(frac = 1)

        self.english = [[*str(dat)] for dat in self.data['english'].tolist()]
        self.tamil = [[*dat] for dat in self.data['tamil'].tolist()]

        
    def __len__(self):
        return len(self.english)


    def __getitem__(self, index):

        english_word = self.english[index]
        tamil_word = self.tamil[index]

        # encoding ground truth words
        eng_out = torch.zeros((MAX_OUTPUT_CHARS ,26 + 3), dtype= torch.float32)
        len_eng_word = len(english_word)
        for i  in range(MAX_OUTPUT_CHARS):
            if i < len_eng_word:
                val = ord(english_word[i]) - 97 + 3
            elif i == len_eng_word:
                val = 1
            else:
                val = 2
            eng_out[i,val] = 1

        # encoding input tamil words
        tamil_in = torch.zeros((MAX_OUTPUT_CHARS ,len(Aksharantar.tamil_ascii) + 1), dtype= torch.float32)
        padding = MAX_OUTPUT_CHARS - len(tamil_word)

        for i in range(MAX_OUTPUT_CHARS):
            if i < padding:
                val = 0
            else:
                index = i - padding
                val = Aksharantar.tamil_ascii.index(ord(tamil_word[index])) + 1
            tamil_in[i,val] = 1
            
        return tamil_in, eng_out

def decode_input(tamil_in):
    if tamil_in.shape == (1, MAX_OUTPUT_CHARS, len(Aksharantar.tamil_ascii) + 1):
        tamil_in = tamil_in.squeeze(0)

    out = ""
    for i in range(MAX_OUTPUT_CHARS):
        word_vec =  tamil_in[i,:]
        val = torch.argmax(word_vec)
        if val == 0:
            continue
        out = out + chr(Aksharantar.tamil_ascii[val - 1])
    return out

def encode_input(tamil):
    tamil_letters = [*tamil]
    tamil_in = torch.zeros((MAX_OUTPUT_CHARS ,len(Aksharantar.tamil_ascii) + 1), dtype= torch.float32)
    padding = MAX_OUTPUT_CHARS - len(tamil_letters)
    for i in range(MAX_OUTPUT_CHARS):
        if i < padding:
            val = 0
        else:
            index = i - padding
            val = Aksharantar.tamil_ascii.index(ord(tamil_letters[index])) + 1
        tamil_in[i,val] = 1
    
    tamil_in = tamil_in.unsqueeze(0)
    return tamil_in


    
def decode_out(english_out):
    if english_out.shape == (1, MAX_OUTPUT_CHARS, 29):
        english_out = english_out.squeeze(0)

    out = ""
    for i in range(MAX_OUTPUT_CHARS):
        word_vec =  english_out[i,:]
        val = torch.argmax(word_vec)
        if val == 1 or val == 2 or val == 0:
            break
        out = out + chr(val + 97 - 3)
    return out


if __name__ == "__main__":
    dataset = Aksharantar(mode = 'valid')
    print(len(dataset))
    for tamil, english in dataset:
        print(english.shape)
        print(tamil.shape)
        break


