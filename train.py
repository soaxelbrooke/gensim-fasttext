
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
EPOCHS = int(os.getenv("EPOCHS", 1))
SKIPGRAM = int(os.getenv("SKIPGRAM", 1))
WINDOW_LEN = int(os.getenv("WINDOW_LEN", 5))
MIN_COUNT = int(os.getenv("MIN_COUNT", 20))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.1))
MIN_LEARNING_RATE = float(os.getenv("MIN_LEARNING_RATE", 0.0001))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
NEGATIVE_SAMPLES = int(os.getenv("NEGATIVE_SAMPLES", 10))


def jsonify_build_params() -> str:
    return json.dumps(
        {
            "REGEX_MATCH_PATTERN": REGEX_MATCH_PATTERN,
            "VOCAB_SIZE": VOCAB_SIZE,
            "EMBED_DIM": EMBED_DIM,
            "LIMIT": LIMIT,
            "EPOCHS": EPOCHS,
            "SKIPGRAM": SKIPGRAM,
            "WINDOW_LEN": WINDOW_LEN,
            "MIN_COUNT": MIN_COUNT,
            "LEARNING_RATE": LEARNING_RATE,
            "MIN_LEARNING_RATE": MIN_LEARNING_RATE,
            "NUM_WORKERS": NUM_WORKERS,
        }
    )


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

    if os.path.exists("vectors_out.sqlite"):
        os.unlink("vectors_out.sqlite")

    conn = sqlite3.connect("vectors_out.sqlite")
    conn.execute(
        """
        CREATE TABLE vector_meta (
            vector_float_bytes integer, 
            embedding_dimensions integer,
            vocab_size integer,
            build_parameters text
        )
    """
    )
    conn.execute(
        "INSERT INTO vector_meta VALUES (?, ?, ?, ?)",
        ("float32", EMBED_DIM, VOCAB_SIZE, jsonify_build_params()),
    )
    conn.execute("CREATE TABLE vectors (token text primary key, vector_bytes blob);")
    conn.execute("CREATE TABLE frequencies (token text primary key, count integer);")

    print(f"Loaded {sum(token_counts.values())} words.  Training model...")

    ft_model = FastText(
        corpus,
        size=EMBED_DIM,
        window=WINDOW_LEN,
        min_count=MIN_COUNT,
        max_vocab_size=VOCAB_SIZE,
        alpha=LEARNING_RATE,
        sg=SKIPGRAM,
        sorted_vocab=1,
        iter=EPOCHS,
        workers=NUM_WORKERS,
        negative=NEGATIVE_SAMPLES,
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
        vector_bytes = vector.astype("float32").tobytes()
        conn.execute("INSERT INTO vectors VALUES (?, ?)", (token, vector_bytes))
        conn.execute("INSERT INTO frequencies VALUES (?, ?)", (token, count))
        progress.update(1)
        idx += 1
        if idx >= VOCAB_SIZE:
            break

    progress.close()
    print("Writing vocab.json...")
    with open("vocab.json", "w") as outfile:
        json.dump(vocab, outfile)
    print("Writing vectors_out.sqlite...")
    conn.commit()
    print("Writing embeddings_out.txt...")
    pandas.DataFrame.from_dict(vectors, orient="index").to_csv(
        "embeddings_out.txt", header=None, sep=" ", quoting=csv.QUOTE_NONE
    )
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1])
