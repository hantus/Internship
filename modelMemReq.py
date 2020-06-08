import tensorflow as tf
import numpy as np
from keras import backend as K  




model = tf.keras.models.load_model('data/models/rnn/100per')
model.summary()

shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))

memory = shapes_count * 4

print(memory)


from keras import backend as K

trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print(f"trainable_count - {trainable_count}, non_trainable_count - {non_trainable_count}")