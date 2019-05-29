# Done: implement backward LSTM
# Done: Logistic regression layer in the end
# TODO: implement attention layer
# TODO: use BERT embeddings

import csv
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas
import bcolz
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###
# glove word embeddings initialization
###


glove_dimensions = 200

# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'data/glove/6B.'+str(glove_dimensions)+'.dat', mode='w')

# with open(f'data/glove/glove.6B.'+str(glove_dimensions)+'d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
    
# vectors = bcolz.carray(vectors[1:].reshape((400000, glove_dimensions)), rootdir=f'data/glove/6B.'+str(glove_dimensions)+'.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'data/glove/6B.'+str(glove_dimensions)+'_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'data/glove/6B.'+str(glove_dimensions)+'_idx.pkl', 'wb'))



torch.manual_seed(1)

df_training = pandas.read_csv("./data/data_prep/guilty_training.csv")

data = df_training.values.tolist()
# print(data[0:3])
random.shuffle(data)
dev_data = data[0:15]
training_data = data[15:len(data)]
# training_data.append(["she is guilty",1.0])
# training_data.append(["he is guilty",1.0])
# training_data.append(["they are guilty",1.0])
# training_data.append(["she is not guilty",0.0])
# training_data.append(["he is not guilty",0.0])
# training_data.append(["they are not guilty",0.0])
random.shuffle(training_data)

# compute mean target for training data for baseline
targets = []
for story, target in training_data:
    targets.append(target)
mean_training_targets = torch.tensor(np.mean(targets))


testing_data = [
    ("they were not guilty", ["NotGuilty"]),
    ("they were guilty", ["Guilty"])
]

# print(training_data)


# vectors = bcolz.open(f'data/glove/6B.50.dat')[:]
# words = pickle.load(open(f'data/glove/6B.50_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'data/glove/6B.50_idx.pkl', 'rb'))

vectors = bcolz.open(f'data/glove/6B.'+str(glove_dimensions)+'.dat')[:]
words = pickle.load(open(f'data/glove/6B.'+str(glove_dimensions)+'_words.pkl', 'rb'))
word2idx = pickle.load(open(f'data/glove/6B.'+str(glove_dimensions)+'_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# print(glove['the'])


# split up sentence to words and save their indices
# those indices will be used for GloVe embedding look up later on
def prepare_sequence(seq, w_embed_dict):
    idxs = [w_embed_dict[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# create a dictionary with a unique word & index pairing
# i.e., every token can be uniquely identified by an index number
w_embed_dict = {}
for sentence, target in training_data + testing_data + dev_data:
    # print("sentence: {}".format(sentence))
    # print("target: {}".format(target))
    split_sentence = sentence.split()
    # print("split_sentence: {}".format(split_sentence))
    for word in split_sentence:
        # print("word: {}".format(word))
        if word not in w_embed_dict:
            w_embed_dict[word] = len(w_embed_dict)

# print(w_embed_dict)
print(len(w_embed_dict))

# number of unique words
matrix_len = len(w_embed_dict)
# initialize embedding matrix with size (dataset’s vocabulary length, word vectors dimension)
wordembedding_matrix = np.zeros((matrix_len, glove_dimensions))
# words_found = 0

# go through dictionary and assign a glove embedding to each word
# the position of the word in the tensor is the index of the word in the dictionary
for word,index in w_embed_dict.items():
    try:
        # print("word: {}".format(word))
        # print("index: {}".format(index))
        wordembedding_matrix[index] = glove[word]
        # words_found += 1
    except KeyError:
        # if word is not in glove, initialize randomly
        # print("This word was not in the dictionary: {}".format(word))
        wordembedding_matrix[index] = np.random.normal(scale=0.6, size=(glove_dimensions, ))

# convert numpy array to tensor
wordembedding_matrix = torch.from_numpy(wordembedding_matrix)

# create initial embedding layer for LSTM that assigns 
# GloVe embeddings for each word
def create_emb_layer(wordembedding_matrix, non_trainable=False):
    num_embeddings, embedding_dim = wordembedding_matrix.size()
    # print(num_embeddings)
    # print(embedding_dim)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': wordembedding_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def avg_layers(hidden_matrix,last_layers):
    i=1
    last_hidden_layers = []
    while i < last_layers:
        try:
            last_hidden_layers.append(hidden_matrix[-i])
            i = i+1
        except:
            pass
    avg_layer = torch.mean(torch.stack(last_hidden_layers),dim=0,keepdim=True)
    return avg_layer

# hidden dimensions
# HIDDEN_DIM = 350
HIDDEN_DIM = 200

class LSTM_Forward(nn.Module):

    def __init__(self, wordembedding_matrix, hidden_dim):
        super(LSTM_Forward, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(wordembedding_matrix, True)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim*2, 1)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # lstm_out contains all hidden layers
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # print("lstm_out: {}".format(lstm_out))
        # pick last 4 layers and average them to get one vector
        last_4_layers = avg_layers(lstm_out.view(len(sentence), -1), 4)
        # print("last_4_layers: {}".format(last_4_layers))
        # linear layer on top of last hidden layer
        prediction = torch.sigmoid(self.linear(last_4_layers))
        # prediction = self.linear(last_4_layers)
        # print("prediction: {}".format(prediction))
        return prediction

model = LSTM_Forward(wordembedding_matrix, HIDDEN_DIM)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# PREDICTIONS BEFORE TRAINING
# See what the scores are before training
with torch.no_grad():
    print("")
    print(testing_data[0][0])
    inputs = prepare_sequence(testing_data[0][0].split(), w_embed_dict)
    prediction = model(inputs)
    # The sentence is "...".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    if prediction > 0.5:
        print("Prediction: {} -- GUILTY!".format(prediction))
    else:
        print("Prediction: {} -- NOT GUILTY!".format(prediction))

    print("")
    print(testing_data[1][0])
    inputs = prepare_sequence(testing_data[1][0].split(), w_embed_dict)
    prediction = model(inputs)
    if prediction > 0.5:
        print("Prediction: {} -- GUILTY!".format(prediction))
    else:
        print("Prediction: {} -- NOT GUILTY!".format(prediction))

# TRAINING
num_epochs = 30
mean_dev_losses = []
mean_baseline_losses = []
mean_training_losses = []
epochs_toplot = []
csvData = [['Data_type', 'Epoch', 'Loss']]
for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    training_losses = []
    epochs_toplot.append(epoch+1)
    count = 0 # I think this can be erased
    for sentence, target in training_data:
        count += 1
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # transform sentence into tensor of embeddings
        sentence_in = prepare_sequence(sentence.split(), w_embed_dict)
        # look up target "embedding"
        # targets = prepare_sequence(target, tag_to_ix)
        targets = torch.tensor(target)

        # Step 3. Run our forward pass.
        prediction = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(prediction, targets)
        training_losses.append(loss)
        csvData.append(['training',epoch,loss.detach().numpy()])
        loss.backward()
        optimizer.step()

    print("Epoch: {}/{}; Training Loss: {}".format(epoch+1, num_epochs, loss))

    with torch.no_grad():
        losses = []
        baseline_losses = []
        for sentence, target in dev_data:
            # transform sentence into tensor of embeddings
            sentence_in = prepare_sequence(sentence.split(), w_embed_dict)
            # look up target "embedding"
            # targets = prepare_sequence(target, tag_to_ix)
            targets = torch.tensor(target)

            # Step 3. Run our forward pass.
            prediction = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(prediction, targets)
            losses.append(loss)
            csvData.append(['testing',epoch,loss.detach().numpy()])
            baseline_loss = loss_function(mean_training_targets, targets)
            baseline_losses.append(baseline_loss)
            csvData.append(['baseline',epoch,baseline_loss.detach().numpy()])

            # print("Prediction: {}; Target: {}".format(prediction,targets))

        mean_dev_losses.append(torch.mean(torch.stack(losses)))
        mean_baseline_losses.append(torch.mean(torch.stack(baseline_losses)))
        mean_training_losses.append(torch.mean(torch.stack(training_losses)))
        # fig, ax = plt.subplots()
        # ax.plot(epochs_toplot, mean_dev_losses,color="green")
        # ax.plot(epochs_toplot, mean_baseline_losses,color="red")
        # ax.plot(epochs_toplot, mean_training_losses,color="blue")
        # ax.set(xlabel='epoch', ylabel='loss',
        #        title='Loss on dev_data')
        # ax.grid()
        # fig.savefig("test.png")

        with open('losses.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

        csvFile.close()

        # print("Losses: {}; Mean loss: {}".format(losses, torch.mean(torch.stack(losses))))
        print("Mean testing loss: {}".format(torch.mean(torch.stack(losses))))


# PREDICTIONS AFTER TRAINING
# See what the scores are after training
with torch.no_grad():
    print("")
    print(testing_data[0][0])
    inputs = prepare_sequence(testing_data[0][0].split(), w_embed_dict)
    prediction = model(inputs)
    # The sentence is "...".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    if prediction > 0.5:
        print("Prediction: {} -- GUILTY!".format(prediction))
    else:
        print("Prediction: {} -- NOT GUILTY!".format(prediction))

    print("")
    print(testing_data[1][0])
    inputs = prepare_sequence(testing_data[1][0].split(), w_embed_dict)
    prediction = model(inputs)
    if prediction > 0.5:
        print("Prediction: {} -- GUILTY!".format(prediction))
    else:
        print("Prediction: {} -- NOT GUILTY!".format(prediction))











