# LearningWithPony
NTU project - Learning language with pony

1. Download Word2vec weights

from gensim.models import KeyedVectors
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
model.save("word2vec.model")

2. Download YOLO weights.
