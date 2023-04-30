import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import tqdm
#allow gpu growth
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
'''gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)'''
#import train data
train_data = pd.read_json('genre_train.json')
test_data = pd.read_json('genre_test.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizing the data using bert tokenizer
def process_data(dataX, max_length = 256):
    
    inputs = np.zeros((len(dataX), max_length))
    attention = np.zeros((len(dataX),max_length))
    
    for i, data in enumerate(dataX['X']):
        data_example = tokenizer.encode_plus(data, max_length=max_length, padding='max_length', add_special_tokens=True, return_token_type_ids=True,
                                             truncation=True, return_attention_mask=True, return_tensors='tf')
        inputs[i,:] = data_example['input_ids']
        attention[i,:] = data_example['attention_mask']
    return inputs, attention

input_ids, attention_mask = process_data(train_data)
#test_input_ids, test_attention_mask = process_data(test_data)

#one hot encode Y
def oneHotEncode(Y):
    Y_train = np.zeros((len(train_data),Y.max()+1))
    Y_train[np.arange(len(train_data)), Y] = 1
    return Y_train

Y_train = oneHotEncode(train_data['Y'].values)
print(Y_train[0:5])

#create dataset
dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, Y_train))
#test_ds = tf.data.Dataset.from_tensor_slices((test_input_ids, test_attention_mask))
def mapper(ids, mask, labels):
    return {'input_ids':ids, 'attention_mask':mask}, labels

dataset = dataset.map(mapper)
dataset = dataset.shuffle(4000).batch(6, drop_remainder=True)
#split into training and validation datasets
size = int(input_ids.shape[0]/6*0.9)
train_dataset = dataset.take(size)
val_dataset = dataset.skip(size)
#initialize bert model
model = TFBertModel.from_pretrained('bert-base-uncased')
#processing of test data
def encode_test_X(text):
  tokens = tokenizer.encode_plus(text, max_length=256,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
  return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
#classifier model
class classifier_model():
    def __init__(self, model):
        self.bert = model
        self.input_ids = tf.keras.layers.Input(shape=(256, ), name='input_ids', dtype='int32')
        self.mask = tf.keras.layers.Input(shape=(256, ), name='attention_mask', dtype='int32')
        self.embeddings = self.bert.bert(self.input_ids, attention_mask=self.mask)[1]
        self.output = tf.keras.layers.Dense(64, activation='relu')(self.embeddings)
        self.y = tf.keras.layers.Dense(4, activation='softmax', name='outputs')(self.output)
        self.class_model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=self.y)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        self.class_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.acc])
        
    def forward_fit(self, train_ds, val_ds):
        history = self.class_model.fit(train_ds, validation_data= val_ds, epochs=5, batch_size=6)
        
    def predict(self, test_ds):
        y_pred = []
        for i,x in test_ds.iterrows():
            y = self.class_model.predict(encode_test_X(x['X']))
            y = np.argmax(y,axis=1)
            y_pred.append(y[0])
            
            
        return y_pred
    def saveModel(self):
        self.class_model.save('Bert_transformer_model1')
        
        
    def loadModel(self):
        self.class_model = tf.keras.models.load_model('Bert_transformer_model1')
#initialize classifier model
classifier = classifier_model(model)
#train model
classifier.forward_fit(train_dataset, val_dataset)
#test model
Y_test_pred = classifier.predict(test_data)
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

classifier.saveModel()