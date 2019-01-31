import tensorflow as tf
import IPMD_loadsave


'''
Exported code from Himan Shurawlani 
through his Github:https://github.com/himanshurawlani/keras-and-tensorflow-serving
'''
# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
#model = tf.keras.models.load_model('./inception.h5')
model = IPMD_loadsave.load_model("./ipmd_model.h5py")
export_path = '../my_image_classifier/1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})


# TENSORFLOW DECOMPOSING FOR SERVING
'''
	If we want to serve it using tensorflow, we will need to convert it from keras to tensorflow

	ASSIGNMENT #1 - convert the given model in IPMD_loadsave.py from keras to tensorflow

	THINGS WE KNOW ABOUT THE MODEL
		It uses a Sequential model
'''

'''
ATTEMPT TO SAVE THE IPMD MODEL USING 
model.save('ipmd_model'.h5) that way it is saved like the inception model used in the example
'''

