MAX_LENGTH = 200
EPOCHS = 100
# EPOCHS = 120    # imdb
LEARNING_RATE = 0.001

SOS_token = 0
EOS_token = 1
hidden_size = 256
# hidden_size = 512   # imdb
BATCH_SIZE = 32

# DATASETS = './data/new_imdb_cypher.json'
DATASETS = './data/restaurants.json'
# DATASETS = './data/new_advising_cypher.json'
OUTPUT_FOLDER = './output/restaurants'
# OUTPUT_FOLDER = './output/advising'
# OUTPUT_FOLDER = './output/imdb'
