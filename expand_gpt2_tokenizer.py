import os
import json
import shutil
import numpy as np
from glob import glob
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, GPT2TokenizerFast

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source_tokenizer', type=str, default='facebook/opt-350m')
parser.add_argument('--datadir', type=str, default='/mnt/lm-ko')
parser.add_argument('--num_training_steps', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--vocab_size', type=int, default=30000)
args = parser.parse_args()


def prepare_data(datadir):
    datalist = glob(f'{datadir}/*.txt')
    
    data = []
    for d in tqdm(datalist):
        d = load_dataset('text', data_files=d)['train']
        data.append(d)
    data = concatenate_datasets(data)
    return data


def sample_batch_idxs(batch_size, maxlen):
    idxs = []
    while len(idxs) < batch_size:
        idx = np.random.randint(0, maxlen)
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def prepare_training_corpus(data, num_training_steps=10000, batch_size=1000):
    for _ in range(num_training_steps):
        batch_idxs = sample_batch_idxs(batch_size, len(data))
        yield data[batch_idxs]['text']
        
        
def read_json(fpath):
    return json.load(open(fpath, 'r'))


def write_json(obj, fpath):
    return json.dump(obj, open(fpath, 'w'))


def main(args):
    print('Preparing corpus...')
    data = prepare_data(args.datadir)
    training_corpus = prepare_training_corpus(data, num_training_steps=args.num_training_steps, batch_size=args.batch_size)
    print('Finished preparing corpus!!')

    print('Training tokenizer...')
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    source_tokenizer.save_pretrained('source')
    proxy_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    target_tokenizer = proxy_tokenizer.train_new_from_iterator(
        training_corpus, 
        vocab_size = args.vocab_size,
        special_tokens_map = source_tokenizer.special_tokens_map
    )
    print('Finished training tokenizer!!')

    print('Expanding tokenizer...')
    os.makedirs('target', exist_ok=True)
    shutil.copy('source/special_tokens_map.json', 'target')
    shutil.copy('source/tokenizer_config.json', 'target')

    target_tokenizer_config = json.loads(target_tokenizer._tokenizer.to_str())
    new_vocab = target_tokenizer_config['model']['vocab']
    new_merges = target_tokenizer_config['model']['merges']

    vocab = read_json('source/vocab.json')
    for v in new_vocab:
        if v not in vocab:
            vocab[v] = len(vocab)
    write_json(vocab, 'target/vocab.json')

    merges = open('source/merges.txt', 'r').read().strip().split('\n')
    for m in new_merges:
        if m not in merges:
            merges.append(m)
    open('target/merges.txt', 'w').write('\n'.join(merges))

    print('Finished expading tokenizer')



if __name__ == '__main__':
    main(args)