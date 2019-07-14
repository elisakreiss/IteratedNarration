import csv
import random
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel, BertTokenizer

ling_measure = "suspect_guilt"
# ling_measure = "author_belief"

torch.manual_seed(1)


# BERT
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)



df_testing = pandas.read_csv("./data/data_prep/"+ling_measure+"_testing.csv")

data = df_testing.values.tolist()
# print(type(data[0:3]))
# print(data[0:3])
random.shuffle(data)
# dev_data = data[0:15]

run_id="testing"

testing_data = []
csvDataStory = [['story_tokenized', 'tensor', 'story_id', 'story', 'token_id']]
for data_id in range(len(data)):
    tokenized_text = tokenizer.tokenize(data[data_id][0])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    testing_data.append([torch.tensor([indexed_tokens]), data[data_id][1]])

    # csvFileCreation
    for token_id in range(len(tokenized_text)):
        csvDataStory.append([tokenized_text[token_id], indexed_tokens[token_id], data_id, data[data_id][0], token_id])

with open('tokenized_stories.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvDataStory)
    csvFile.close()


# testing_data = [
#     ("they were not guilty", ["NotGuilty"]),
#     ("they were guilty", ["Guilty"])
# ]


# hidden dimensions
# HIDDEN_DIM = 350
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

    def forward(self, sentence, sen_id):
        # embeds = self.word_embeddings(sentence)
        encoded_layers, _ = self.Bert(sentence,output_all_encoded_layers=False)

        # lstm_out contains all hidden layers
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))

        attention1 = torch.tanh(self.attention1(lstm_out))
        # print("attention1.size(): {}".format(attention1.size()))

        attention2 = torch.softmax(self.attention2(attention1),dim=0)
        # CSVFILECREATION
        for token_id in range(len(attention2.squeeze().tolist())):
            csvDataTensor.append([sentence.squeeze().tolist()[token_id], attention2.squeeze().tolist()[token_id], sen_id, token_id])
        # csvDataTensor.append([sentence.squeeze().tolist(), attention2.squeeze().tolist()])

        dot_product = torch.mm(torch.transpose(attention2.view(len(sentence[0]), -1),0,1),lstm_out.view(len(sentence[0]), -1))
        # dot_product = torch.mm(torch.transpose(attention2.view(len(sentence), -1),0,1),lstm_out.view(len(sentence), -1))

        # linear layer on top of last hidden layer
        prediction = torch.sigmoid(self.linear(dot_product))

        return prediction

model = LSTM_Forward(HIDDEN_DIM)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

model.load_state_dict(torch.load("data/model_weights/suspect_guilt_cv0.pt"))
model.eval()
print("loaded weights")



with torch.no_grad():
    losses = []
    # csvData = [['Data_type', 'Prediction', 'Target', 'Loss']]
    csvDataTensor = [['tensor', 'attention', 'story_id', 'token_id']]
    csvDataLoss = [['story_id', 'target', 'prediction', 'loss']]
    sen_id = 0 # for csv file creation
    for sentence, target in testing_data:
        # transform sentence into tensor of embeddings
        # sentence_in = prepare_sequence(sentence.split(), w_embed_dict)
        sentence_in = sentence
        # look up target "embedding"
        # targets = prepare_sequence(target, tag_to_ix)
        targets = torch.tensor(target)
        print("target: ", target)

        # Step 3. Run our forward pass.
        prediction = model(sentence_in, sen_id)

        # Step 4. Compute the loss, gradients, and update the parameters by
        loss = loss_function(prediction, targets)
        # losses.append(loss)
        # csvData.append(['final_testing', prediction.detach().numpy()[0][0], targets.detach().numpy(), loss.detach().numpy()])
        csvDataLoss.append([sen_id, targets.detach().numpy(), prediction.detach().numpy()[0][0], loss.detach().numpy()])

        # with open('losses_'+ling_measure+'_'+run_id+'.csv', 'w') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerows(csvData)

        # csvFile.close()

        sen_id += 1
    
    with open('attention.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvDataTensor)

    csvFile.close()

    with open('loss.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvDataLoss)

    csvFile.close()








