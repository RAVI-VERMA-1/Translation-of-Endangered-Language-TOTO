import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def load_dataset(file_path, num_rows=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        rows = f.read().split('\n')[:num_rows]

    input_texts, target_texts = [], []
    input_characters, target_characters = set(), set()

    for row in rows:
        input_text, target_text = row.split('\t')
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text.lower())
        target_texts.append(target_text.lower())
        input_characters.update(list(input_text.lower()))
        target_characters.update(list(target_text.lower()))

    return input_texts, target_texts, sorted(list(input_characters)), sorted(list(target_characters))


def preprocess_data(input_texts, target_texts, input_characters, target_characters):
    max_input_length = max(len(i) for i in input_texts)
    max_target_length = max(len(i) for i in target_texts)

    pad_en = [1] + [0] * (len(input_characters) - 1)
    pad_dec = [0] * len(target_characters)
    pad_dec[2] = 1

    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')

    en_in_data, dec_in_data, dec_tr_data = [], [], []

    for input_t, target_t in zip(input_texts, target_texts):
        cv_inp = cv.fit(input_characters)
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

        cv_tar = cv.fit(target_characters)
        dec_in_data.append(cv_tar.transform(list(target_t)).toarray().tolist())
        dec_tr_data.append(cv_tar.transform(list(target_t)[1:]).toarray().tolist())

        for data, pad, max_length in zip([en_in_data, dec_in_data, dec_tr_data], [pad_en, pad_dec, pad_dec],
                                         [max_input_length, max_target_length, max_target_length]):
            if len(data[-1]) < max_length:
                data[-1].extend([pad] * (max_length - len(data[-1])))

    return np.array(en_in_data, dtype="float32"), np.array(dec_in_data, dtype="float32"), np.array(dec_tr_data,
                                                                                                   dtype="float32")


def build_model(num_en_chars, num_dec_chars):
    en_inputs = Input(shape=(None, num_en_chars))
    encoder = LSTM(256, return_state=True)
    en_outputs, state_h, state_c = encoder(en_inputs)
    en_states = [state_h, state_c]

    dec_inputs = Input(shape=(None, num_dec_chars))
    dec_lstm = LSTM(256, return_sequences=True, return_state=True)
    dec_outputs, _, _ = dec_lstm(dec_inputs, initial_state=en_states)

    dec_dense = Dense(num_dec_chars, activation="softmax")
    dec_outputs = dec_dense(dec_outputs)

    model = Model([en_inputs, dec_inputs], dec_outputs)
    return model


def train_model(model, en_in_data, dec_in_data, dec_tr_data):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([en_in_data, dec_in_data], dec_tr_data, batch_size=64, epochs=200, validation_split=0.2)
    return model


def save_model_and_data(model, data, model_filename="s2s", data_filename="training_data.pkl"):
    model.save(model_filename)
    pickle.dump(data, open(data_filename, "wb"))

def evaluate_model(model, en_in_data, dec_in_data, dec_tr_data):
    scores = model.evaluate([en_in_data, dec_in_data], dec_tr_data, verbose=0)
    print("Model Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(scores[0], scores[1]))



def main():
    input_texts, target_texts, input_characters, target_characters = load_dataset('eng-toto.txt')
    en_in_data, dec_in_data, dec_tr_data = preprocess_data(input_texts, target_texts, input_characters,
                                                           target_characters)
    model = build_model(len(input_characters), len(target_characters))
    trained_model = train_model(model, en_in_data, dec_in_data, dec_tr_data)

    model_data = {
        'input_characters': input_characters,
        'target_characters': target_characters,
        'max_input_length': max(len(i) for i in input_texts),
        'max_target_length': max(len(i) for i in target_texts),
        'num_en_chars': len(input_characters),
        'num_dec_chars': len(target_characters)
    }

    save_model_and_data(trained_model, model_data)
    evaluate_model(trained_model, en_in_data, dec_in_data, dec_tr_data)


if __name__ == "__main__":
    main()
