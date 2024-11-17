
# models.py
 
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
 
 
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, vocab_index, embed_dim=50, hidden_dim=100, output_dim=2):
        super(RNNClassifier, self).__init__()
        self.vocab_index = vocab_index  # Store vocab_index for later use
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Use log-softmax for stability with NLLLoss
 
    def forward(self, input_seq):
        # Embedding
        embedded = self.embedding(input_seq)
        # RNN
        _, (hidden, _) = self.rnn(embedded)
        # Feedforward and activation
        output = self.fc(hidden[-1])
        return self.softmax(output)
 
class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
 
    def predict(self, context):
        with torch.no_grad():
            # Convert each character in the context to its index
            context_indices = [self.vocab_index.index_of(char) for char in context]
            # context_tensor = torch.from_numpy(np.asarray(context_indices)).unsqueeze(0)
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
    embedding_dim = 10
    hidden_dim = 30
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
                # input_tensor = torch.from_numpy(np.asarray(ex_indices)).unsqueeze(0)
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
 
 
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, vocab_index, embed_dim=50, hidden_dim=100):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_index = vocab_index  # Store vocab_index for later use
 
    def forward(self, input_seq, hidden=None):
        # Embed the input sequence
        embedded = self.embedding(input_seq)
        # Process through RNN; provide hidden state if carrying over
        output, hidden = self.rnn(embedded, hidden)
        # Predict next character at each position
        logits = self.fc(output)
        return logits, hidden
 
    def init_hidden(self, batch_size):
        # Initialize hidden state for RNN (two tensors for LSTM)
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))
 
    def get_log_prob_single(self, next_char, context):
        # Prepare context tensor
        context_indices = [self.vocab_index.index_of(c) for c in context]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(context_tensor)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            # Get log prob of next_char at final position
            next_char_index = self.vocab_index.index_of(next_char)
            return log_probs[0, -1, next_char_index].item()
 
    def get_log_prob_sequence(self, next_chars, context=" "):
        # Compute log probability for a sequence of characters
        log_prob = 0.0
        for next_char in next_chars:
            log_prob += self.get_log_prob_single(next_char, context)
            context += next_char  # Append char to context for the next prediction
        return log_prob
 
 
def train_lm(args, train_text, dev_text, vocab_index):
    # Hyperparameters
    embed_dim = 50
    hidden_dim = 100
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10
    seq_length = 30  # Number of characters per input sequence
 
    # Model, loss, optimizer
# Model, loss, optimizer
    model = RNNLanguageModel(vocab_size=len(vocab_index), vocab_index=vocab_index, embed_dim=embed_dim, hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
    # Prepare training data
    train_indices = [vocab_index.index_of(c) for c in train_text if vocab_index.contains(c)]
    train_data = [train_indices[i:i+seq_length+1] for i in range(0, len(train_indices) - seq_length)]
 
 
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            inputs = [torch.tensor(seq[:-1], dtype=torch.long) for seq in batch]
            targets = [torch.tensor(seq[1:], dtype=torch.long) for seq in batch]
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
 
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, len(vocab_index)), targets.view(-1))
            total_loss += loss.item()
           
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
       
        avg_loss = total_loss / len(train_data)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
 
    # Evaluation
    model.eval()
    dev_indices = [vocab_index.index_of(c) for c in dev_text if vocab_index.contains(c)]
    dev_sequences = [dev_indices[i:i+seq_length+1] for i in range(0, len(dev_indices) - seq_length)]
    total_log_prob = 0.0
    total_characters = 0
 
    with torch.no_grad():
        for seq in dev_sequences:
            inputs = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0)
            targets = torch.tensor(seq[1:], dtype=torch.long).unsqueeze(0)
            logits, _ = model(inputs)  # Extract only logits
            log_probs = nn.functional.log_softmax(logits, dim=2)
            # Sum log probabilities for each target character
            for t in range(len(targets[0])):
                total_log_prob += log_probs[0, t, targets[0, t]].item()
            total_characters += len(targets[0])
 
    avg_log_prob = total_log_prob / total_characters
    perplexity = np.exp(-avg_log_prob)
    print(f"Log prob of dev set: {total_log_prob:.4f}")
    print(f"Avg log prob: {avg_log_prob:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
 
    return model
 