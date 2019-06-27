import csv
import copy
import math
import random
import pandas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel, BertTokenizer

ling_measure = "suspect_guilt"

torch.manual_seed(1)

# Print iterations progress
def print_progress_bar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, progress_bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# BERT
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

df_training = pandas.read_csv("./data/data_prep/"+ling_measure+"_training.csv")

data = df_training.values.tolist()
# print(type(data[0:3]))
random.shuffle(data)


cv_fold = 10

split_data = [[] for i in range(cv_fold)]

for data_id in range(len(data)):
    bucket = math.floor(data_id/(len(data)/cv_fold))
    tokenized_text = tokenizer.tokenize(data[data_id][0])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    split_data[bucket].append([torch.tensor([indexed_tokens]),data[data_id][1]])


for i in range(cv_fold):
    print("")
    print("crossvalidation step: ",i)

    dev_data = split_data[i]
    run_id = "cv"+str(i)

    training_data_prep = copy.deepcopy(split_data)
    training_data_prep.pop(i)

    training_data = []
    for fold_list in training_data_prep:
        for training_sample in fold_list:
            training_data.append(training_sample)

    # compute mean target for training data for baseline
    targets = []
    for story, target in training_data:
        targets.append(target)
    mean_training_targets = torch.tensor(np.mean(targets))

    # hidden dimensions
    HIDDEN_DIM = 200

    class LSTM_Forward(nn.Module):

        def __init__(self, hidden_dim):
            super(LSTM_Forward, self).__init__()
            self.hidden_dim = hidden_dim

            # self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(wordembedding_matrix, True)
            self.Bert = BertModel.from_pretrained('bert-base-uncased')


            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
            self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True)

            # this only works for sentences of length 4
            self.attention1 = nn.Linear(hidden_dim*2,50)
            self.attention2 = nn.Linear(50,1)

            # The linear layer that maps from hidden state space to tag space
            self.linear = nn.Linear(hidden_dim*2, 1)

        def forward(self, sentence):
            # embeds = self.word_embeddings(sentence)
            encoded_layers, _ = self.Bert(sentence,output_all_encoded_layers=False)

            # lstm_out contains all hidden layers
            # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))
            # print("lstm_out.size(): {}".format(lstm_out.size()))

            attention1 = torch.tanh(self.attention1(lstm_out))
            # print("attention1.size(): {}".format(attention1.size()))

            attention2 = torch.softmax(self.attention2(attention1),dim=0)
            # attention2 = self.attention2(attention1)
            # print("attention2.size(): {}".format(attention2.size()))
            # print("attention2: {}".format(attention2))

            dot_product = torch.mm(torch.transpose(attention2.view(len(sentence[0]), -1),0,1),lstm_out.view(len(sentence[0]), -1))
            # dot_product = torch.mm(torch.transpose(attention2.view(len(sentence), -1),0,1),lstm_out.view(len(sentence), -1))
            # print("dot_product.size(): {}".format(dot_product.size()))
            # print("torch.transpose(attention2.view(len(sentence[0]), -1),0,1).size()",torch.transpose(attention2.view(len(sentence[0]), -1),0,1).size())
            # print("lstm_out.view(len(sentence[0]), -1).size()",lstm_out.view(len(sentence[0]), -1).size())

            # linear layer on top of last hidden layer
            prediction = torch.sigmoid(self.linear(dot_product))

            return prediction

    model = LSTM_Forward(HIDDEN_DIM)
    loss_function = nn.MSELoss()
    # loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # TRAINING
    num_epochs = 30
    mean_dev_losses = []
    mean_baseline_losses = []
    mean_training_losses = []
    epochs_toplot = []
    csvData = [['Data_type', 'Epoch', 'Prediction', 'Target', 'Loss']]
    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        training_losses = []
        epochs_toplot.append(epoch+1)
        count = 0
        for sentence, target in training_data:
            count += 1
            printProgressBar(count, len(training_data), prefix = 'Progress:', suffix = 'Complete', length = 50)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # transform sentence into tensor of embeddings
            # sentence_in = prepare_sequence(sentence)
            sentence_in = sentence
            # look up target "embedding"
            # targets = prepare_sequence(target, tag_to_ix)
            targets = torch.tensor(target)

            # Step 3. Run our forward pass.
            prediction = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(prediction, targets)
            training_losses.append(loss)
            csvData.append(['training',epoch,prediction.detach().numpy()[0][0],targets.detach().numpy(),loss.detach().numpy()])
            loss.backward()

            optimizer.step()

        print("Epoch: {}/{}; Training Loss: {}".format(epoch+1, num_epochs, loss))

        with torch.no_grad():
            losses = []
            baseline_losses = []
            for sentence, target in dev_data:
                # transform sentence into tensor of embeddings
                # sentence_in = prepare_sequence(sentence.split(), w_embed_dict)
                sentence_in = sentence
                # look up target "embedding"
                # targets = prepare_sequence(target, tag_to_ix)
                targets = torch.tensor(target)

                # Step 3. Run our forward pass.
                prediction = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(prediction, targets)
                losses.append(loss)
                csvData.append(['testing',epoch,prediction.detach().numpy()[0][0],targets.detach().numpy(),loss.detach().numpy()])
                baseline_loss = loss_function(mean_training_targets, targets)
                baseline_losses.append(baseline_loss)
                csvData.append(['baseline',epoch,mean_training_targets.numpy(),targets.detach().numpy(),baseline_loss.detach().numpy()])

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

            with open('losses_'+ling_measure+'_'+run_id+'.csv', 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(csvData)

            csvFile.close()

            # print("Losses: {}; Mean loss: {}".format(losses, torch.mean(torch.stack(losses))))
            print("Mean testing loss: {}".format(torch.mean(torch.stack(losses))))

    torch.save(model.state_dict(), "model_weights/"+ling_measure+'_'+run_id+".pt")
