#building the model keras---an attempt to build a seq2seq model in keras
from data_clean import *

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def define_model(n_input,n_output,n_units,vocab_to_int,batch_size):
     embeddings = word_embedding_matrix
     enc_embed_input=tf.nn.embedding_lookup(embeddings,n_input)
     dec_input=process_encoding_input(n_output,vocab_to_int,batch_size)
     dec_embed_input=tf.nn.embedding_lookup(embeddings,dec_input)
     encoder_inputs=Input(shape=(None,enc_embed_input))

     encoder=LSTM(n_units,return_state=True)
     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
     encoder_states = [state_h, state_c]

     #define training decoder
     decoder_inputs = Input(shape=(None, dec_embed_input))

     decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
     decoder_dense = Dense(n_output, activation='softmax')
     decoder_outputs = decoder_dense(decoder_outputs)
     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

     #define inference encoder
     encoder_model = Model(encoder_inputs, encoder_states)

     #define inference decoder
     decoder_state_input_h = Input(shape=(n_units,))
     decoder_state_input_c = Input(shape=(n_units,))
     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
     decoder_states = [state_h, state_c]
     decoder_outputs = decoder_dense(decoder_outputs)
     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

     return model, encoder_model, decoder_model
