# ner_model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_processing import preprocess_data

class NERModel(nn.Module):
    def __init__(self, embedding_layer, hidden_size, output_size):
        super(NERModel, self).__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_layer.embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output

    def predict_entities(self, sentence, vocab):
        # Preprocess the input sentence
        tokenized_sentence, _ = preprocess_data([{"sentence": sentence, "entities": []}])
        indexed_sentence = [[vocab[token] for token in tokenized_sentence[0]]]

        # Convert to PyTorch tensor
        input_data = torch.tensor(indexed_sentence)

        # Run the model
        self.eval()
        with torch.no_grad():
            output = self(input_data, [len(indexed_sentence[0])])

        # Convert the output to predictions (assuming binary classification)
        predictions = torch.argmax(output, dim=-1).numpy().flatten()

        # Map predictions back to entity types
        idx_to_entity = {0: "O", 1: "ENTITY"}  # Adjust as per your entity types
        predicted_entities = [idx_to_entity[idx] for idx in predictions[0]]

        return predicted_entities
