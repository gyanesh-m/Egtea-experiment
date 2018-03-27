import keras
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import *
from keras.models import Sequential
class Classify:
	def __init__(self,batch,epoch,naction,base_dir="/output/",nverb=None,n_noun=None,t_size=(256,256)):
		self.dir=base_dir
		self.bch=batch
		self.epc=epoch
		self.naction=naction
		self.nverb=nverb
		self.n_noun=n_noun
		self.target_size=t_size
		self.mdl=self.load_model()
		self.mc=ModelCheckpoint(self.dir+"weights.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		self.es=EarlyStopping(monitor='loss', min_delta=1, patience=10, verbose=1, mode='auto')
		self.tb=TensorBoard(log_dir='./logs',  batch_size=self.bch, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0)
		self.lr=ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
	def load_model(self):
		model = Sequential()
		model.add(Conv2D(64, (2,2), input_shape=(self.target_size[0],self.target_size[1],3),strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros' ))
		model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(Conv2D(128, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(Conv2D(128, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(Conv2D(256, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(Conv2D(256, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(Conv2D(256, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))
		model.add(Flatten())
		model.add(Dense(1024,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.naction,activation='softmax'))
		optimizer = Adam(lr=1e-5, decay=1e-6)
		model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		return model

	def train(self):
		h,w=self.target_size
		train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

		test_datagen = ImageDataGenerator(rescale=1./255)

		train_generator = train_datagen.flow_from_directory(
	    '/egtea/dataset/train/',
	    target_size=(h, w),
	    batch_size=self.bch,
	    class_mode='categorical')

		validation_generator = test_datagen.flow_from_directory(
	    '/egtea/dataset/test/',
	    target_size=(h, w),
	    batch_size=self.bch,
	    class_mode='categorical')

		self.mdl.fit_generator(
	    train_generator,
	    steps_per_epoch=int(train_generator.samples/float(self.bch)),
	    epochs=self.epc,
	    callbacks=[self.mc,self.tb,self.es],
	    validation_data=validation_generator,
		validation_steps=200)

model=Classify(batch=64,epoch=30,naction=34,t_size=(128,128))
model.train()
