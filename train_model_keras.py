from build_model_keras import *

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
checkpointer = ModelCheckpoint(filepath='seq2seq'+"_keras_sp.h5", verbose=1, save_best_only=True)

history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,
                  callbacks=[checkpointer],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()