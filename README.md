# LearningWithPony
NTU project - Learning language with pony

## Download Word2vec weights

from gensim.models import KeyedVectors

import gensim.downloader as api

model = api.load("word2vec-google-news-300")

model.save("word2vec.model")

It's should be save the word2vec.model and word2vec.model.vecors.npy

## Download YOLO weights.

https://ppt.cc/fLvgmx


## Weights location

Put the word2vec.model, word2vec.model.vecors.npy and yolov4.h5 in the root dictionary (same path with Main.py)




