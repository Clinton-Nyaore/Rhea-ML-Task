import spacy

def preprocess_data(data):
    tokenized_sentences = []
    entities = []

    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    for entry in data:
        sentence = entry["sentence"]
        doc = nlp(sentence)
        tokens = [token.text for token in doc]
        tokenized_sentences.append(tokens)

        entity_labels = [0] * len(tokens)  # Initialize with zeros
        for start, end, label in entry["entities"]:
            entity_labels[start:end] = [1] * (end - start)
        entities.append(entity_labels)

    return tokenized_sentences, entities
