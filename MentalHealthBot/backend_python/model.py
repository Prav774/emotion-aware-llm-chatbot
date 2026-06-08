from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Attention
from tensorflow.keras.models import Model

def build_model():
    input_layer = Input(shape=(50,))

    embedding = Embedding(input_dim=10000, output_dim=128)(input_layer)

    lstm_out = LSTM(128, return_sequences=True)(embedding)

    attention = Attention()([lstm_out, lstm_out])

    lstm_out2 = LSTM(64)(attention)

    dropout = Dropout(0.5)(lstm_out2)

    output = Dense(2, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model