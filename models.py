# models.py

from matplotlib import pyplot as plt
import numpy as np
import collections
from sklearn.metrics import classification_report, confusion_matrix
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
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2, dropout=0.1, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) # shape: (batch_size, seq_len, embed_dim)
        _, hidden = self.rnn(embedded) # hidden shape: (1, batch_size, hidden_dim)
        return self.output_layer(hidden[-1])  # shape: (batch_size, output_dim) | Take the last hidden state

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def predict(self, context):
        with torch.no_grad():
            # Convert each character in the context to its index
            context_indices = [self.vocab_index.index_of(char) for char in context]
            # context_tensor = torch.from_numpy(np.asarray(context_indices)).unsqueeze(0)
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0) # Add batch dimension | # shape: (1, seq_len)
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


def evaluate_rnn_classifier(model, dev_data, vocab_index):
    """
    Evaluates the RNNClassifier on the development set.
    """
    # model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ex, label in dev_data:
            ex_indices = [vocab_index.index_of(char) for char in ex]
            input_tensor = torch.tensor(ex_indices, dtype=torch.long).unsqueeze(0)
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            correct += (predicted == label)
            total += 1

    return correct / total * 100.0

def display_classification_report(y_true, y_pred):
    """
    Displays a classification report and confusion matrix.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Consonant", "Vowel"]))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Consonant", "Vowel"], yticklabels=["Consonant", "Vowel"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_metrics(train_losses, val_accuracies):
    """
    Plots training loss and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_predictions(model, data, vocab_index):
    """
    Gets true and predicted labels for a dataset.
    """
    y_true = []
    y_pred = []

    with torch.no_grad():
        for ex, label in data:
            ex_indices = [vocab_index.index_of(char) for char in ex]
            input_tensor = torch.tensor(ex_indices, dtype=torch.long).unsqueeze(0)
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()

            y_true.append(label)
            y_pred.append(predicted)

    return y_true, y_pred


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
    embedding_dim = 10
    hidden_dim = 30
    output_dim = 2  # 0: consonant, 1: vowel (Binary classification)
    learning_rate = 0.0006
    epochs = 10

    # Initialize model, optimizer and loss
    model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    dev_data = [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]

    # Lists to store metrics
    train_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle data at the start of each epoch
        random.shuffle(train_data)

        for ex, label in train_data:
                ex_indices = [vocab_index.index_of(char) for char in ex]
                # input_tensor = torch.from_numpy(np.asarray(ex_indices)).unsqueeze(0)
                input_tensor = torch.tensor(ex_indices, dtype=torch.long).unsqueeze(0)  # shape: (1, seq_len)
                target_tensor = torch.tensor([label], dtype=torch.long)  # shape: (1)

                # Forward pass
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
        

        epoch_loss /= len(train_data)

        # Validation phase
        val_accuracy = evaluate_rnn_classifier(model, dev_data, vocab_index)
        train_losses.append(epoch_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot training metrics
    plot_training_metrics(train_losses, val_accuracies)
 
    # Generate final performance report on dev data
    y_true, y_pred = get_predictions(model, dev_data, vocab_index)
    display_classification_report(y_true, y_pred)

    return RNNClassifier(model, vocab_index)


import numpy as np
import matplotlib.pyplot as plt

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
    

class RNNLanguageModelModule(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type="LSTM"):
        super(RNNLanguageModelModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

        # Use LSTM or GRU based on `rnn_type`
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=2, batch_first=True)
        else:
            raise ValueError("Invalid rnn_type. Choose 'LSTM' or 'GRU'.")
        
        # If you using bidirectional=True, you need to adjust the output layer ( hidden_size * 2)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, inputs, hidden_state=None):
        # Embed the input characters
        embedded = self.embedding(inputs)  # Shape: (batch_size, seq_len, embed_size)
        # Pass through the RNN
        rnn_out, hidden_state = self.rnn(embedded, hidden_state)  # Shape: (batch_size, seq_len, hidden_size)
        # Generate logits for each character in the sequence
        logits = self.output_layer(rnn_out)  # Shape: (batch_size, seq_len, vocab_size)
        return logits, hidden_state
    

class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        # Convert context and next_char to tensor indices
        context_idx = torch.tensor([[self.vocab_index.index_of(c) for c in context]], dtype=torch.long)
        next_char_idx = self.vocab_index.index_of(next_char)

        # Forward pass through the model
        logits, self.hidden_state = self.model_emb(context_idx, self.hidden_state)
        probs = nn.functional.log_softmax(logits, dim=-1)

        # Return the log probability of next_char
        return probs[0, -1, next_char_idx].item()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        self.hidden_state = None  # Reset hidden state for a new sequence

        for char in next_chars:
            total_log_prob += self.get_log_prob_single(char, context)
            context += char  # Update context with the predicted character

        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    # Hyperparameters
    embed_size = 64
    hidden_size = 128
    seq_len = 40 # chunk size
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Use "LSTM" or "GRU"
    rnn_type = "GRU"

    # Create the PyTorch model and optimizer
    model = RNNLanguageModelModule(len(vocab_index), embed_size, hidden_size, rnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    def create_batches(text, seq_len, batch_size):
        data = [vocab_index.index_of(c) for c in text]

        # Calculate the usable size (truncate to fit batch_size and seq_len)
        num_tokens = len(data)
        usable_tokens = (num_tokens - 1) // (batch_size * seq_len) * (batch_size * seq_len)

        # Ensure inputs and targets are the same length
        data = data[:usable_tokens + 1]  # Include one more token for targets

        # Prepare inputs and targets
        inputs = torch.tensor(data[:-1], dtype=torch.long).view(batch_size, -1)
        targets = torch.tensor(data[1:], dtype=torch.long).view(batch_size, -1)
        return inputs, targets

    train_inputs, train_targets = create_batches(train_text, seq_len, batch_size)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        hidden_state = None

        for i in range(0, train_inputs.size(1), seq_len):
            inputs = train_inputs[:, i:i + seq_len]
            targets = train_targets[:, i:i + seq_len]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits, hidden_state = model(inputs, hidden_state)

            # Detach hidden state to prevent gradient buildup
            if isinstance(hidden_state, tuple):  # For LSTM
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:  # For RNN or GRU
                hidden_state = hidden_state.detach()

            # Compute loss
            loss = criterion(logits.reshape(-1, len(vocab_index)), targets.reshape(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Return the trained model wrapped in RNNLanguageModel
    return RNNLanguageModel(model, None, vocab_index)



def compute_perplexity(model, text, vocab_index, seq_len=40):
    """
    Computes perplexity of the model on the given text.
    :param model: Trained RNNLanguageModelModule
    :param text: Text sequence to evaluate
    :param vocab_index: Indexer for the vocabulary
    :param seq_len: Length of input chunks for evaluation
    :return: Perplexity of the model on the given text
    """
    model.eval()
    data = [vocab_index.index_of(c) for c in text]
    num_tokens = len(data) - 1  # Exclude the last token for input

    inputs = torch.tensor(data[:-1], dtype=torch.long).unsqueeze(0)  # Shape: (1, num_tokens)
    targets = torch.tensor(data[1:], dtype=torch.long).unsqueeze(0)  # Shape: (1, num_tokens)

    total_log_prob = 0.0
    total_chars = 0

    with torch.no_grad():
        hidden_state = None
        for i in range(0, inputs.size(1), seq_len):
            input_seq = inputs[:, i:i + seq_len]
            target_seq = targets[:, i:i + seq_len]
            
            logits, hidden_state = model(input_seq, hidden_state)

            # Detach hidden state to prevent gradient buildup
            if isinstance(hidden_state, tuple):  # For LSTM
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:  # For RNN or GRU
                hidden_state = hidden_state.detach()

            log_probs = nn.functional.log_softmax(logits, dim=-1)
            total_log_prob += log_probs.gather(2, target_seq.unsqueeze(-1)).sum().item()
            total_chars += target_seq.numel()

    # Compute perplexity
    avg_log_prob = total_log_prob / total_chars
    perplexity = np.exp(-avg_log_prob)
    return perplexity




