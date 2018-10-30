
import os
import re
import sys
from tqdm import tqdm
from typing import Counter
from gensim.models import FastText
import pandas
import sqlite3
import csv
import json

REGEX_MATCH_PATTERN = os.getenv("REGEX_MATCH_PATTERN", r"[\w']+|[,\.\?!;\-\(\)]")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 100_000))
EMBED_DIM = int(os.getenv("EMBED_DIM", 300))
LIMIT = int(os.getenv("LIMIT", 1_000_000_000))


def main(inpath: str):
    regex = re.compile(REGEX_MATCH_PATTERN)
    token_counts = Counter()
    corpus = []
    with open(inpath) as infile:
        for idx, line in enumerate(tqdm(infile, desc="Loading corpus")):
            if idx >= LIMIT:
                break
            doc = regex.findall(line)
            corpus.append(doc)
            for token in doc:
                token_counts[token] += 1
    
    if os.path.exists('vectors_out.sqlite'):
        os.unlink('vectors_out.sqlite')

    conn = sqlite3.connect('vectors_out.sqlite')
    conn.execute('''
        CREATE TABLE vector_meta (
            vector_float_bytes integer, 
            embedding_dimensions integer,
            vocab_size integer
        )
    ''')
    conn.execute('INSERT INTO vector_meta VALUES (?, ?, ?)',
                 ('float32', EMBED_DIM, VOCAB_SIZE))
    conn.execute('CREATE TABLE vectors (token text primary key, vector_bytes blob);')
    conn.execute('CREATE TABLE frequencies (token text primary key, cout integer);')

    print(f"Loaded {sum(token_counts.values())} words.  Training model...")

    ft_model = FastText(
        corpus, 
        size=EMBED_DIM, 
        window=5, 
        min_count=10, 
        max_vocab_size=VOCAB_SIZE,
        alpha=0.1,
        sg=1,
        sorted_vocab=1,
        iter=10,
        workers=8,
    )
    
    vectors = {}
    vocab = {}
    sorted_counts = sorted(token_counts.items(), key=lambda p: -p[1])
    idx = 2
    progress = tqdm(desc="Writing Vectors", total=VOCAB_SIZE)
    for token, count in sorted_counts:
        try:
            vector = ft_model.wv.get_vector(token)
        except:
            # print("Model miss on " + token)
            continue
        vectors[token] = vector
        vocab[token] = idx
        vector_bytes = vector.astype('float32').tobytes()
        conn.execute('INSERT INTO vectors VALUES (?, ?)', (token, vector_bytes))
        conn.execute('INSERT INTO frequencies VALUES (?, ?)', (token, count))
        progress.update(1)
        idx += 1
        if idx >= VOCAB_SIZE:
            break

    progress.close()
    print("Writing vocab.json...")
    with open('vocab.json', 'w') as outfile:
        json.dump(vocab, outfile)
    print("Writing vectors_out.sqlite...")
    conn.commit()
    print("Writing embeddings_out.txt...")
    pandas.DataFrame.from_dict(vectors, orient='index') \
        .to_csv('embeddings_out.txt', header=None, sep=" ", quoting=csv.QUOTE_NONE)
    print("Done!")


if __name__ == '__main__':
    main(sys.argv[1])
