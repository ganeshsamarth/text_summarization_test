#training the model in keras
from build_model_keras import *
start = 200000
end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))

embeddings=word_embedding_matrix
enc_embed_input=tf.nn.embedding_lookup(embeddings,padded_text)
dec_embed_input=tf.nn.embedding_lookup(embeddings,padded_summaries)
n_units=100
model,encoder_model,decoder_model=define_model(max_text_length,max_summary_length,n_units)


model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
