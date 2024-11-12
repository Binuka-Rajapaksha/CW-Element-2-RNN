# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import random

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNNModel(nn.Module):
    """
    PyTorch RNN model for classifying sequences as being followed by consonants or vowels.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x) # shape: (batch_size, seq_len, embed_dim)
        _, hidden = self.rnn(embedded) # hidden shape: (1, batch_size, hidden_dim)
        return self.fc(hidden[-1])  # shape: (batch_size, output_dim) | Take the last hidden state

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def predict(self, context):
        with torch.no_grad():
            # Convert each character in the context to its index
            context_indices = [self.vocab_index.index_of(char) for char in context]
            context_tensor = torch.tensor(context_indices).unsqueeze(0) # Add batch dimension | # shape: (1, seq_len)
            output = self.model(context_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
        
    # Hyperparameters
    vocab_size = len(vocab_index)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # 0: consonant, 1: vowel (Binary classification)
    learning_rate = 0.0005
    epochs = 10

    # Initialize model, optimizer and loss
    model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare and shuffle training data
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle data at the start of each epoch
        random.shuffle(train_data)

        for ex, label in train_data:
                ex_indices = [vocab_index.index_of(char) for char in ex]
                input_tensor = torch.tensor(ex_indices).unsqueeze(0)  # shape: (1, seq_len)
                target_tensor = torch.tensor([label])  # shape: (1)

                optimizer.zero_grad()
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    return RNNClassifier(model, vocab_index)
# # models.py

# import numpy as np
# import collections
# import torch
# import torch.nn as nn
# import torch.optim as optim

# #####################
# # MODELS FOR PART 1 #
# #####################

# class ConsonantVowelClassifier(object):
#     def predict(self, context):
#         """
#         :param context:
#         :return: 1 if vowel, 0 if consonant
#         """
#         raise Exception("Only implemented in subclasses")


# class FrequencyBasedClassifier(ConsonantVowelClassifier):
#     """
#     Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
#     classify as consonant, otherwise as vowel.
#     """
#     def __init__(self, consonant_counts, vowel_counts):
#         self.consonant_counts = consonant_counts
#         self.vowel_counts = vowel_counts

#     def predict(self, context):
#         # Look two back to find the letter before the space
#         if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
#             return 0
#         else:
#             return 1

# class RNNModel(nn.Module):
#     """
#     PyTorch RNN model for classifying sequences as being followed by consonants or vowels.
#     """
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(RNNModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)



#     def forward(self, x):
#         embedded = self.embedding(x)
#         _, (hidden, _) = self.rnn(embedded)
#         hidden = self.dropout(hidden[-1])  # Apply dropout to the last hidden state
#         return self.fc(hidden)



# class RNNClassifier(ConsonantVowelClassifier):
#     def __init__(self, model, vocab_index):
#         self.model = model
#         self.vocab_index = vocab_index

#     def predict(self, context):
#         # Convert context into indices
#         input_indices = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
#         output = self.model(input_indices)
#         prediction = torch.argmax(output, dim=1).item()  # 0 for consonant, 1 for vowel
#         return prediction


# def train_frequency_based_classifier(cons_exs, vowel_exs):
#     consonant_counts = collections.Counter()
#     vowel_counts = collections.Counter()
#     for ex in cons_exs:
#         consonant_counts[ex[-1]] += 1
#     for ex in vowel_exs:
#         vowel_counts[ex[-1]] += 1
#     return FrequencyBasedClassifier(consonant_counts, vowel_counts)


# def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
#     # Hyperparameters
#     embedding_dim = 50
#     hidden_dim = 50
#     output_dim = 2  # 0: consonant, 1: vowel
#     learning_rate = 0.001
#     epochs = 10

#     # Initialize model, loss, optimizer, and scheduler
#     model = RNNModel(len(vocab_index), embedding_dim, hidden_dim, output_dim)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Halve LR every 5 epochs

#     # Convert examples to tensors
#     def prepare_data(examples, label):
#         data = []
#         for ex in examples:
#             indices = [vocab_index.index_of(c) for c in ex]
#             data.append((torch.tensor(indices, dtype=torch.long), label))
#         return data

#     train_data = prepare_data(train_cons_exs, 0) + prepare_data(train_vowel_exs, 1)
#     dev_data = prepare_data(dev_cons_exs, 0) + prepare_data(dev_vowel_exs, 1)

#     # Training loop
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for inputs, label in train_data:
#             inputs = inputs.unsqueeze(0)  # Add batch dimension
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, torch.tensor([label]))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")

#         # Step the scheduler after each epoch
#         scheduler.step()

#     # Evaluation on the development set
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for inputs, label in dev_data:
#             inputs = inputs.unsqueeze(0)  # Add batch dimension
#             outputs = model(inputs)
#             prediction = torch.argmax(outputs, dim=1).item()
#             if prediction == label:
#                 correct += 1
#     accuracy = correct / len(dev_data)
#     print(f"Development set accuracy: {accuracy * 100:.2f}%")

#     return RNNClassifier(model, vocab_index)



#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
