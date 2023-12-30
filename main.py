# train.py
import torch
import torch.optim as optim
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score
from data_processing import preprocess_data
from ner_model import NERModel

# train.py
def train(model, optimizer, criterion, input_data, target_data, vocab):
    if vocab is None:
        vocab = {} # Create an empty vocabulary if not provided
        
    model.train()
    optimizer.zero_grad()

    lengths = [len(sentence) for sentence in input_data]
    output = model(input_data, lengths)

    # Modify the target_data during preprocessing to match the shape of the output
    flat_target_data = target_data.view(-1)

    # Reshape the output tensor to have the same shape as flat_target_data
    output_reshaped = output.view(-1, output.size(-1))  # Get the last dimension size as output_size

    # Ensure both tensors have the same size
    min_size = min(output_reshaped.size(0), flat_target_data.size(0))
    output_reshaped = output_reshaped[:min_size, :]
    flat_target_data = flat_target_data[:min_size]

    loss = criterion(output_reshaped, flat_target_data)
    loss.backward()
    optimizer.step()

    return model, vocab


def inference(model, sentence, vocab):
    if vocab is None:
        raise ValueError("Vocabulary must be provided for inference.")

    # Preprocess the input sentence
    tokenized_sentence, _ = preprocess_data([{"sentence": sentence, "entities": []}])

    # Update vocabulary with new words from the sentence
    for token in tokenized_sentence[0]:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Map words to their indices in the vocabulary
    indexed_sentence = [[vocab[token] for token in tokenized_sentence[0]]]

    # Convert to PyTorch tensor
    input_data = torch.tensor(indexed_sentence)

    # Run the model
    model.eval()
    with torch.no_grad():
        output = model(input_data, [len(indexed_sentence[0])])

    # Convert the output to predictions (assuming binary classification)
    predictions = torch.argmax(output, dim=-1).numpy().flatten()

    # Map predictions back to entity types
    idx_to_entity = {0: "O", 1: "ENTITY"}
    predicted_entities = [idx_to_entity[prediction] for prediction in predictions.tolist()]


    # Debug information
    print("Tokenized Sentence:", tokenized_sentence)
    print("Indexed Sentence:", indexed_sentence)
    print("Vocabulary:", vocab)
    print("Predictions:", predictions)
    print("Predicted Entities:", predicted_entities)

    return predicted_entities

if __name__ == "__main__":
    # Example dataset (We can replace this with an actual dataset)
    training_data = [
        {"sentence": "John works at Google.", "entities": [(0, 4, "PERSON"), (16, 22, "ORG")]},
        {"sentence": "Apple is planning to build a new office in New York.", "entities": [(0, 5, "ORG"), (42, 51, "LOC")]},
        {"sentence": "Microsoft Corporation is a technology company.", "entities": [(0, 9, "ORG")]},
        {"sentence": "New York City is a vibrant metropolis.", "entities": [(0, 8, "LOC")]},
        # Add more examples...
    ]

    # Data Preparation
    tokenized_sentences, entities = preprocess_data(training_data)

    # Model Building
    vocab_size = len(set(token for sentence in tokenized_sentences for token in sentence))
    embedding_dim = 50
    hidden_size = 50
    output_size = 2  # Change this based on the number of entity types

    # Use the provided NERModel
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    model = NERModel(embedding_layer, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize vocabulary
    vocab = {}

    # Convert sentences to indices
    indexed_sentences = [[hash(token) % vocab_size for token in sentence] for sentence in tokenized_sentences]

    # Padding sequences
    max_len = max(len(sentence) for sentence in indexed_sentences)
    padded_sentences = [sentence + [0] * (max_len - len(sentence)) for sentence in indexed_sentences]

    # Convert to PyTorch tensors
    input_data = torch.tensor(padded_sentences)

    # Flatten the entities list
    flat_entities = [label for sublist in entities for label in sublist]

    # Convert to PyTorch tensor
    target_data = torch.tensor(flat_entities)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        lengths = [len(sentence) for sentence in indexed_sentences]
        output = model(input_data, lengths)

        # Modify the target_data during preprocessing to match the shape of the output
        flat_target_data = target_data.view(-1)

        # Reshape the output tensor to have the same shape as flat_target_data
        output_reshaped = output.view(-1, output_size)

        # Ensure both tensors have the same size
        min_size = min(output_reshaped.size(0), flat_target_data.size(0))
        output_reshaped = output_reshaped[:min_size, :]
        flat_target_data = flat_target_data[:min_size]

        loss = criterion(output_reshaped, flat_target_data)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Print shapes for debugging
        print("Output shape:", output.shape)
        print("Flat Target shape:", flat_target_data.shape)

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        # Example validation dataset (We can replace this with an actual validation dataset)
        validation_data = [
            {"sentence": "Mary is a software engineer.", "entities": [(0, 4, "PERSON"), (12, 32, "JOB")]},
            {"sentence": "Microsoft announces a partnership with Tesla.", "entities": [(0, 9, "ORG"), (31, 36, "ORG")]}
            # Add more examples...
        ]

        tokenized_val_sentences, val_entities = preprocess_data(validation_data)

        indexed_val_sentences = [[hash(token) % vocab_size for token in sentence] for sentence in tokenized_val_sentences]
        padded_val_sentences = [sentence + [0] * (max_len - len(sentence)) for sentence in indexed_val_sentences]
        input_val_data = torch.tensor(padded_val_sentences)

        # Flatten the true labels
        true_labels = [label for sublist in val_entities for label in sublist]

        val_lengths = [len(sentence) for sentence in indexed_val_sentences]
        val_output = model(input_val_data, val_lengths)

        # Convert the output to predictions (assuming binary classification)
        predictions = torch.argmax(val_output, dim=-1).numpy().flatten()

        # Ensure the lengths match
        min_size = min(len(true_labels), len(predictions))
        true_labels = true_labels[:min_size]
        predictions = predictions[:min_size]

        # Calculate metrics
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

        # Inference on a new sentence
        while True:
            user_input = input("Enter a sentence for named entity recognition (or 'exit' to stop): ")

            if user_input.lower() == 'exit':
                break

            predicted_entities = inference(model, user_input, vocab)
            print("Predicted Entities:", predicted_entities)