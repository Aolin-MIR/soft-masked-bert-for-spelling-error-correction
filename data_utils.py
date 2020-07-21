import numpy as np
import re

regex = re.compile(r'[\s\u3000]')


def pad(sequence, max_length):
    if len(sequence)>=max_length:
        return sequence[:max_length]
    
    pad_len = max_length - len(sequence)
    padding = np.zeros(pad_len)
    return np.concatenate((sequence, padding))


def pad_sequences(sequences, max_length):
    return [pad(x, max_length) for x in sequences]


def pad_2(sequence, max_length):
    pad_len = max_length - len(sequence)
    padding = np.full(pad_len, -1)
    return np.concatenate((sequence, padding))


def pad_sequences_2(sequences, max_length):
    return [pad_2(x, max_length) for x in sequences]


def pad_elmo(sequence, max_length):
    pad_len = max_length - len(sequence)
    padding = np.zeros((pad_len, 1024))
    return np.concatenate((sequence, padding), axis=0)


def pad_sequences_elmo(sequences, max_length):
    return [pad_elmo(x, max_length) for x in sequences]


def minibatches(data, minibatch_size):
    (word_ids, label_ids) = data
    data_len = len(word_ids)
    num_batch = int((data_len - 1) / minibatch_size) + 1
    for i in range(num_batch):
        start_id = i * minibatch_size
        end_id = min((i + 1) * minibatch_size, data_len)
        batch_x = np.array(word_ids[start_id:end_id])
        batch_y = np.array(label_ids[start_id:end_id])
        yield batch_x, batch_y


def minibatches_bert(data, minibatch_size):
    (sentences, label_ids) = data
    data_len = len(sentences)
    num_batch = int((data_len - 1) / minibatch_size) + 1
    for i in range(num_batch):
        start_id = i * minibatch_size
        end_id = min((i + 1) * minibatch_size, data_len)
        batch_x = sentences[start_id:end_id]
        batch_y = np.array(label_ids[start_id:end_id])
        yield batch_x, batch_y, start_id, end_id


def process_bert_embeddings(embeddings, sequence_lengths, max_length):
    embeddings_p = []
    # print(66, embeddings.shape,max_length)
    for i in range(len(embeddings)):
        sequence_length = sequence_lengths[i]
        # print(i, sequence_length)
        # Remove special tokens, i.e. [CLS], [SEP]
        embedding = np.delete(embeddings[i], [0, sequence_length + 1], 0)

        # print(72,embedding.shape)
        # Remove extra padding tokens
        embedding = np.delete(embedding, np.s_[max_length - 1:-1], 0)

        embeddings_p.append(embedding.tolist())
    return embeddings_p


def minibatches_elmo(data, minibatch_size):
    (words, label_ids) = data
    data_len = len(words)
    num_batch = int((data_len - 1) / minibatch_size) + 1
    for i in range(num_batch):
        start_id = i * minibatch_size
        end_id = min((i + 1) * minibatch_size, data_len)
        batch_x = words[start_id:end_id]
        batch_y = np.array(label_ids[start_id:end_id])
        yield batch_x, batch_y, start_id, end_id


def minibatches_pred(tuples, minibatch_size):
    data_len = len(tuples)
    num_batch = int((data_len - 1) / minibatch_size) + 1
    for i in range(num_batch):
        start_id = i * minibatch_size
        end_id = min((i + 1) * minibatch_size, data_len)
        batch = tuples[start_id:end_id]
        yield batch


def remove_spaces(string, indices):
    flag = 0  # 0 indicates normal case, 1 indicates target, -1 indicates space
    str_length = len(string)
    temp = [flag] * str_length
    flags = []
    for i in range(str_length):
        char = string[i]
        if i in indices:
            temp[i] = 1
        elif char == " " or char == "　":
            temp[i] = -1
    for _flag in temp:
        if _flag != -1:
            flags.append(_flag)
    new_indices = [index for index in range(len(flags)) if flags[index] == 1]
    new_string = regex.sub("", string)
    return new_string, new_indices
