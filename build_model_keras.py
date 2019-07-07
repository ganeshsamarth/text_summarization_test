#building the model keras---an attempt to build a seq2seq model in keras
from data_clean import *



def define_model(max_text_length,max_summary_length,n_units):
     dim_rep=300


     encoder_inputs=Input(shape=(None,max_text_length,dim_rep))

     encoder=LSTM(n_units,return_state=True)
     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
     encoder_states = [state_h, state_c]

     #define training decoder
     decoder_inputs = Input(shape=(None, max_summary_length,dim_rep))

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
