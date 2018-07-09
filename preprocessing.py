import os
import pickle

if not os.path.isfile('./data/c2i') or not os.path.isfile('./data/i2c'):
    i2c = ['<pad>', '<unk>', '<sos>', '<eos>', '<num>']
    c2i = {'<pad>': 0, '<unk>': 1, '<sow>': 2, '<eow>': 3, '<num>': 4}

    with open('./data/corpus.txt', encoding='utf-8') as f:
        corpus = f.read()

    idx = 0
    for c in corpus:
        try:
            c2i[c]
        except KeyError:
            i2c.append(c)
            c2i[c] = idx
            idx += 1

    with open('./data/i2c', 'wb') as f:
        pickle.dump(i2c, f)

    with open('./data/c2i', 'wb') as f:
        pickle.dump(c2i, f)

if not os.path.isfile('./data/pcorpus'):
    i2c = pickle.load(open('./data/i2c', 'rb'))
    c2i = pickle.load(open('./data/c2i', 'rb'))
    data = []
    length = []
    with open('./data/corpus.txt', encoding='utf-8') as f:
        corpus = f.read().split()

    for word in corpus:
        tw = []
        if word.isdigit():
            tw.append(4)
        for char in word:
            tw.append(c2i[char])
        tw.append(3)
        data.append(tw)
        length.append(len(tw) if tw[0] != 4 else len(tw)-1)

    with open('./data/pcorpus', 'wb') as f:
        pickle.dump(data, f)
    with open('./data/length', 'wb') as f:
        pickle.dump(length, f)