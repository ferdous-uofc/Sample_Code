import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU

class Discriminator(object):
    def __init__(self, width=28, height=28, channels=1):
    	self.width = width
    	self.height = height
    	self.channels = channels
    	self.optimizers = Adam(lr=0.0002, decay=8e-9)
    	self.max_size = self.width*self.height*self.channels
    	self.shape = (self.width, self.height, self.channels)

    	self.discriminator = self.model()
    	self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizers, metrics=['accuracy'])

    def model(self):
    	model = Sequential()
    	model.add(Flatten(input_shape=(self.shape)))
    	model.add(Dense(self.max_size, input_shape=self.shape))
    	model.add(LeakyReLU(alpha=0.2))   
    	model.add(Dense(self.max_size // 2))
    	model.add(LeakyReLU(alpha=0.2))
    	model.add(Dense(1, activation='sigmoid'))
    	return model

class Generator(object):
	"""docstring for Generator"""
	def __init__(self, width=28, height=28, channels=1, latent_space=128):
		self.width=width
		self.height=height
		self.channels=channels
		self.max_size=self.width * self.height * self.channels
		self.latent_space=latent_space
		self.optimizers = Adam(lr=0.0002, decay=8e-9)
		self.latent_space_rand = np.random.normal(0, 1, (self.latent_space,))

		self.generator = self.model()
		self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizers, metrics=['accuracy'])

	def model(self, block_start=128, num_blocks=4):
		model = Sequential()

		block_size = block_start
		model.add(Dense(block_size, input_shape=(self.latent_space,)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))

		for i in range(num_blocks-1):
			block_size *= 2
			model.add(Dense(block_size))
			model.add(LeakyReLU(alpha=0.2))
			model.add(BatchNormalization(momentum=0.8))

		model.add(Dense(self.max_size, activation='tanh'))
		model.add(Reshape((self.width, self.height, self.channels)))

		return model

class Gan(object):
	"""docstring for Gan"""
	def __init__(self, discriminator, generator):
		self.generator = generator
		self.discriminator = discriminator
		self.optimizers = Adam(lr=0.0002, decay=8e-9)
		self.discriminator.trainable=False
		self.gan = self.model()
		self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizers, metrics=['accuracy'])

	def model(self):
		model=Sequential()
		model.add(self.generator)
		model.add(self.discriminator)
		return model

generator_obj = Generator()
discriminator_obj = Discriminator()
gan_obj = Gan(discriminator_obj.discriminator, generator_obj.generator)

from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for x in y_test:
    if x not in list(range(10)):
        print(x)

x_train = (x_train - 127.5) / 127.5


nreal_images_ = 32
nepochs_ = 10000
ncheckpoints = 5

for i in range(nepochs_):
    im_start = np.random.randint(0, len(x_train) - nreal_images_)
    
#     discriminator_acc = 0
#     while discriminator_acc < 0.95:
    x_real = np.expand_dims(x_train[im_start:(im_start+nreal_images_)], axis=3)
    y_real = np.ones([nreal_images_,1])

    latent_samples = np.random.normal(0, 1, (nreal_images_, 128))
    x_generated = gan_obj.generator.predict(latent_samples)
    y_generated = np.zeros([nreal_images_,1])

    x_batch = np.concatenate([x_real, x_generated])
    y_batch = np.concatenate([y_real, y_generated])

    gan_obj.discriminator.trainable=True
    discriminator_acc = gan_obj.discriminator.train_on_batch(x_batch, y_batch)[1]

    y_generated = np.ones([nreal_images_ * 2, 1])
    latent_samples = np.random.normal(0, 1, (nreal_images_ * 2, 128))
    gan_obj.discriminator.trainable=False
    generator_acc = gan_obj.gan.train_on_batch(latent_samples,y_generated)[1]

    print('Epoch:', i)
    print('Discriminator Accuracy:', discriminator_acc)
    print('Generator Accuracy:', generator_acc)

np.random.seed(20)
latent_samples = np.random.normal(0, 1, (nreal_images_, 128))
# generator_obj.generator.set_weights(gan_obj.generator.get_weights())
# x_generated = gan_obj.generator.predict(latent_samples)
x_generated = generator_obj.generator.predict(latent_samples)
plt.imshow(np.reshape((x_generated[12] * 127.5) + 127.5, (28, 28)), cmap='gray')
# plt.title(str(y_generated[8]))
plt.show()



