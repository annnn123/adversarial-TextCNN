import os
import re
import json
import tqdm
import random
from itertools import chain
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StringProcess(object):
    """clean raw texts
    """

    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9():;,\.!\?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        self.url = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):

        string = re.sub(r"\\n", " ", string)
        string = re.sub(r"\\t", " ", string)

        string = re.sub(self.other_char, " ", string)
        string = re.sub(r" \'", " ", string)
        string = re.sub(r"\' ", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def clean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

def remove_less_word(string, word_st):
    return " ".join([word for word in string.split() if word in word_st])


class DataProcess:
    def __init__(self, data_path, clean_data_path, dataset_name, encoding=None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.clean_data_path = clean_data_path
        self.context_dct = defaultdict(dict)

        self.encoding = encoding
        self.clean()

    def clean(self):
        sp = StringProcess()
        word_list = list()

        with open(self.data_path, 'rb', encoding=self.encoding) as fin:
            for indx, item in tqdm.tqdm(enumerate(fin), desc="clean the text"):
                data = item.strip().decode('latin1')
                data = sp.clean_str(data)
                if self.dataset_name.lower() not in {"mr"}:
                    data = sp.remove_stopword(data)
                word_list.extend(data.split())

        word_st = set()
        if self.dataset_name.lower() not in {"mr"}:
            for word, value in Counter(word_list).items():
                if value < 5:
                    continue
                word_st.add(word)
        else:
            word_st = set(word_list)

        doc_len_list = list()
        with open(self.clean_data_path, mode='w') as fout:
            with open(self.data_path, mode='rb', encoding=self.encoding) as fin:
                for line in tqdm.tqdm(fin):
                    line_str = line.strip().decode('latin1')
                    line_str = sp.clean_str(line_str)
                    if self.dataset_name.lower() not in {"mr"}:
                        line_str = sp.remove_stopword(line_str)
                        line_str = remove_less_word(line_str, word_st)

                    fout.write(line_str)
                    fout.write(" \n")

                    doc_len_list.append(len(line_str.split()))

        print("Average length:", np.mean(doc_len_list))
        print("doc count:", len(doc_len_list))
        print("Total number of words:", len(word_st))


class DataSplit():
    def __init__(self, data_path, label_path, save_dir, encoding=None):

        self.data_path = data_path
        self.label_path = label_path
        self.save_dir = save_dir
        self.encoding = encoding

        self.split()

    def split(self):

        texts = []
        with open(self.data_path, 'r', encoding=self.encoding) as f:
            for line in f.readlines():
                texts.append(line.strip())

        label_info_ls = []
        with open(self.label_path, 'r', encoding=self.encoding) as f:
            for line in f.readlines():
                label_info_ls.append(line.strip().split())

        assert len(texts) == len(label_info_ls)

        dataset_type = ['train', 'dev', 'test'] # set([label_info[1] for label_info in label_info_ls])
        for tp in dataset_type:
            tmp_dataset = []
            for text, (idx, _tp, label) in zip(texts, label_info_ls):
                if tp in _tp:
                    tmp_dataset.append({'text': text, 'label': label})

            if len(tmp_dataset)>0:
                with open(os.path.join(self.save_dir, tp+'.json'), 'w', encoding=self.encoding) as f:
                    json.dump(tmp_dataset, f, indent=2)

        return






######################################################################################################


class Vocab():
    """ Vocabulary, i.e. structure containing language terms.
    """
    def __init__(self, word2id=None, lower=True):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words to indices
        @param lower (bool): whether to lowercase the text, default True
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = {}
            self.word2id['<pad>'] = 0   #Padding Token
            self.word2id['<s>'] = 1     #Start Token
            self.word2id['</s>'] = 2    #End Token
            self.word2id['<unk>'] = 3   #Unknown Token
        self.unk_idx = self.word2id['<unk>']
        self.bos_idx = self.word2id['<s>']
        self.eos_idx = self.word2id['</s>']
        self.padding_idx = self.word2id['<pad>']
        self.id2word = {w_id: w for w, w_id in self.word2id.items()}

        self.lower = lower

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        if self.lower:
            word = word.lower()
        return self.word2id.get(word, self.unk_idx)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        if self.lower:
            word = word.lower()
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return f"Vocabulary[size={len(self)}]"

    def get_word_by_id(self, word_id):
        """ Return mapping of index to word.
        @param word_id (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word.get(word_id)

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if self.lower:
            word = word.lower()
        if word not in self.word2id:
            word_id = self.word2id[word] = len(self)
            self.id2word[word_id] = word
            return word_id
        else:
            return self[word]

    def encode(self, text):
        """ Convert list of words or string into list of indices.
        @param sent (list[str] or string): sentence in words
        @return word_ids (list[int]): sentence in indices
        """
        if type(text) == str:
            if self.lower:
                text = text.lower()
            text = text.strip().split()

        return [self[word] for word in text]

    def decode(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sent (list[str]): list of words
        """
        return [self.id2word.get(w_id) for w_id in word_ids]

    @staticmethod
    def build_from_corpus(corpus, vocab_size=5000, freq_cutoff=1, lower=True):
        """ Given a corpus construct a Vocabulary.
        @param corpus (list[list[str]]): corpus of text produced by read_corpus function
        @param vocab_size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab (Vocab): Vocab instance produced from provided corpus
        """
        vocab = Vocab(word2id=None, lower=lower)

        if len(corpus) == 0:
            return vocab

        if type(corpus[0]) == str:
            corpus = [text.strip().split() for text in corpus]

        word_freq_counter = Counter(chain(*corpus))
        valid_words = [word for word, freq in word_freq_counter.items() if freq >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda x: word_freq_counter[x], reverse=True)[: vocab_size]
        for word in top_k_words:
            vocab.add(word)
        return vocab

    def save(self, vocab_save_path):
        with open(vocab_save_path, 'w') as f:
            json.dump({'word2id': self.word2id, 'lower': self.lower}, f, indent=2)
        print(F"save vocab dict to {vocab_save_path}")

    @staticmethod
    def load(vocab_save_path):
        with open(vocab_save_path, 'r') as f:
            save_dict = json.load(f)
        return Vocab(save_dict['word2id'], save_dict['lower'])


class Label():
    def __init__(self, label2id=None):
        if label2id:
            self.label2id = label2id
        else:
            self.label2id = {}
        self.id2label = {label_id: label for label, label_id in self.label2id.items()}

    def __getitem__(self, label):
        try:
            return self.label2id.get(label)
        except:
            raise ValueError('label does not exist')

    def __contains__(self, label):
        return label in self.label2id

    def __len__(self):
        return len(self.label2id)

    def __setitem__(self, key, value):
        raise ValueError('Label is readonly')

    def __repr__(self):
        return f"Label[size={len(self)}]"

    def id2word(self, label_id):
        return self.id2word.get(label_id)

    def add(self, label):
        if label not in self.label2id:
            label_id = self.label2id[label] = len(self)
            self.id2label[label_id] = label
            return label_id
        else:
            return self[label]

    @staticmethod
    def build(labels):
        label2id = Label()
        label_set = set(labels)
        for label in label_set:
            label2id.add(label)
        return label2id

    def save(self, label_save_path):
        with open(label_save_path, 'w') as f:
            json.dump(self.label2id, f, indent=2)

    @staticmethod
    def load(label_save_path):
        with open(label_save_path, 'r') as f:
            label2id = json.load(f)
        return Label(label2id)


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def train_dev_split(data, dev_ratio=0.1):
    train_size = len(data)
    random.shuffle(data)
    train_data = data[: int(train_size * (1 - dev_ratio))]
    dev_data = data[int(train_size * (1 - dev_ratio)):]

    return train_data, dev_data



def pad_to_max_len(input_ids, max_len, pad_token=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[: max_len]
    else:
        input_ids = input_ids + [pad_token] * (max_len - len(input_ids))
    return input_ids


class CLSDataset(Dataset):
    def __init__(self, data, vocab, label2id, max_seq_len):
        self.data = data
        self.vocab = vocab
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.vocab.encode(item['text'])
        attention_mask = [1] * len(text)
        label = self.label2id[item['label']]

        text = pad_to_max_len(text, self.max_seq_len, self.vocab.padding_idx)
        attention_mask = pad_to_max_len(attention_mask, self.max_seq_len, 0)

        sample = {'input_ids': torch.tensor(text, dtype=torch.long),
                  'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                  'label': torch.tensor(label, dtype=torch.long)}

        return sample

    def __len__(self):
        return len(self.data)


def build_dataloader_for_cls(data, vocab, label2id, batch_size, max_seq_len, shuffle=True):
    dataset = CLSDataset(data, vocab, label2id, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle)
    return dataloader


def load_word2vec(w2v_path):
    w2v = {}
    with open(w2v_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            if line:
                w2v[line[0]] = [float(i) for i in line[1:]]
    return w2v


def load_pretrained_embeddings(vocab, w2v_path, embedding_size):

    w2v = load_word2vec(w2v_path)

    pretrained_embeddings = []
    for word_id in range(len(vocab)):
        word = vocab.id2word[word_id]
        if word in w2v:
            pretrained_embeddings.append(w2v[word])
        else:
            pretrained_embeddings.append([0.] * embedding_size)

    pretrained_embeddings = torch.tensor(np.array(pretrained_embeddings), dtype=torch.float)

    return pretrained_embeddings


if __name__ == '__main__':


    dataset = 'Ohsumed'
    data_path = f"./dataset/{dataset}/{dataset}.txt"
    clean_data_path = f"./dataset/{dataset}/{dataset}_clean.txt"
    label_path = f"./dataset/{dataset}/{dataset}_label.txt"
    data_dave_dir = f"./dataset/{dataset}/"

    # data process
    DataProcess(data_path=data_path, clean_data_path=clean_data_path, dataset_name=dataset)


    # split data into train/dev/test
    DataSplit(data_path=clean_data_path, label_path=label_path, save_dir=data_dave_dir)


    # load data
    with open(f'./dataset/{dataset}/train.json', 'r') as f:
        train_data = json.load(f)
    text_length_ls = [len(item['text'].split()) for item in train_data]
    import numpy as np
    print('mean length:', np.mean(text_length_ls))
    print('max length:', np.max(text_length_ls))
    for p in [75, 85, 95]:
        print(p, np.percentile(text_length_ls, p))

    vocab = Vocab.build_from_corpus([item['text'] for item in train_data], vocab_size=50000, freq_cutoff=3, lower=True)
    print(vocab)
    print(len(vocab))
    vocab.save(f'./dataset/{dataset}/vocab.json')
    print(vocab.word2id)

    vocab = Vocab.load(f'./dataset/{dataset}/vocab.json')
    print(vocab)


    label2id = Label.build([item['label'] for item in train_data])
    print(label2id)
    print(len(label2id))
    print(label2id.label2id)
    label2id.save(f'./dataset/{dataset}/labels.json')

    label2id = Label.load(f'./dataset/{dataset}/labels.json')
    print(label2id)