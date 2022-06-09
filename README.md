# LearningWithPony
NTU project - Learning language with pony

## Download Word2vec weights

from gensim.models import KeyedVectors
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
model.save("word2vec.model")

## Download YOLO weights.
