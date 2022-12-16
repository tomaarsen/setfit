import numpy as np
from sentence_transformers import InputExample


def sentence_pairs_generation(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    num_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in num_classes]

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        label = labels[first_idx]
        second_idx = np.random.choice(idx[np.where(num_classes == label)[0][0]])
        positive_sentence = sentences[second_idx]
        # Prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

        negative_idx = np.where(labels != label)[0]
        negative_sentence = sentences[np.random.choice(negative_idx)]
        # Prepare a negative pair of sentences and update our lists
        pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our sentence pairs and labels
    return pairs


def sentence_pairs_generation_multilabel(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative
    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        sample_labels = np.where(labels[first_idx, :] == 1)[0]
        if len(np.where(labels.dot(labels[first_idx, :].T) == 0)[0]) == 0:
            continue
        else:

            for _label in sample_labels:
                second_idx = np.random.choice(np.where(labels[:, _label] == 1)[0])
                positive_sentence = sentences[second_idx]
                # Prepare a positive pair and update the sentences and labels
                # lists, respectively
                pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

            # Search for sample that don't have a label in common with current
            # sentence
            negative_idx = np.where(labels.dot(labels[first_idx, :].T) == 0)[0]
            negative_sentence = sentences[np.random.choice(negative_idx)]
            # Prepare a negative pair of sentences and update our lists
            pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our sentence pairs and labels
    return pairs


def sentence_pairs_generation_cos_sim(sentences, pairs, cos_sim_matrix):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    idx = list(range(len(sentences)))

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        second_idx = int(np.random.choice([x for x in idx if x != first_idx]))

        cos_sim = float(cos_sim_matrix[first_idx][second_idx])
        paired_sentence = sentences[second_idx]
        pairs.append(InputExample(texts=[current_sentence, paired_sentence], label=cos_sim))

        third_idx = np.random.choice([x for x in idx if x != first_idx])
        cos_sim = float(cos_sim_matrix[first_idx][third_idx])
        paired_sentence = sentences[third_idx]
        pairs.append(InputExample(texts=[current_sentence, paired_sentence], label=cos_sim))

    return pairs
