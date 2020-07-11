from keras.layers import BatchNormalization, Dense, Dropout , GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import time

from sklearn.metrics import classification_report


class TransferLearningModel():
    
    def __init__(self):
        """
        Initialize model parameters and hiperparameters.
        """
        self.HEIGHT = 150
        self.WIDTH = 150
        self.DEPTH = 3
        self.LOSS_FUNC = 'binary_crossentropy'
        self.BS = 64
        self.EPOCHS = 10
        self.OPT_FUNC = 'adam'
        self.METRICS = ['accuracy']
        self.LR_REDUCE = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=3, verbose=1) # reduce learning rate timely
        self.CHECKPOINT = ModelCheckpoint("output/weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min') # checkpoints the best result
        self.HIPERPARAMS = [self.LR_REDUCE, self.CHECKPOINT]
        
        
    def initialize_model(self):
        """
        Load inception model and its weights.
        """
        self.base_model = InceptionV3(weights=None, include_top=False , input_shape=(self.HEIGHT, self.WIDTH, self.DEPTH))
        self.base_model.load_weights("input/inception_v3_weights.h5")
    
    def create_model(self):
        """
        Create the base pre-trained model.
        """ 
        x = self.base_model.output
        x = Dropout(0.5)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        predictions = Dense(2, activation='sigmoid')(x)
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        return

    
    def compile_model(self):
        """
        Compile the model by using the defined hiperparameters.
        """
        self.model.compile(loss=self.LOSS_FUNC,
                          optimizer=self.OPT_FUNC,
                          metrics=self.METRICS)
        return self.model
    
    
    def fit_model(self, X_train, y_train, test_set):
        """
        Fit the model with the validation set.
        """
        start_time = time.time()
        result = self.model.fit(X_train, y_train, batch_size = self.BS, validation_data = test_set, callbacks=self.HIPERPARAMS, epochs=self.EPOCHS)
        end_time = time.time()
        self.train_time = end_time - start_time
        print( "The model took %0.3f seconds to train.\n"%self.train_time )
        return result
    
    
    def evaluate_model(self, X_test, y_test):
        """
        Predict the labels in the test set. 
        """
        pred = self.model.predict(X_test)
        pred = np.argmax(pred,axis = 1) 
        y_true = np.argmax(y_test,axis = 1)
        
#         f1_score(y_test, pred, average='weighted')
        print ('\n*Classification Report:\n', classification_report(y_test, pred))
#         confusion_matrix_graph = confusion_matrix(y_test, predictions)
        
        return pred, y_true
    
    
    def train_model(self, X_train, y_train, validation_set):
        """
        Initialize, create, compile and train the model.
        """
        model = self.initialize_model()
        model = self.create_model()
        model = self.compile_model()
        return self.fit_model(X_train, y_train, validation_set)
        