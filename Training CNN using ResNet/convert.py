import tensorflow as tf

new_model = tf.keras.models.load_model('./final.keras')

# Check its architecture
new_model.summary()

#Save the model into h5 format
new_model.save('final.h5')