#training the model in keras
from build_model_keras import *
start = 200000
end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))



model,encoder_model,decoder_model=define_model()


model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
