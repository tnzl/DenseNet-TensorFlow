# Imports
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, Dropout, Concatenate, Flatten, Input, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model

def get_model(growth_rate, layers_per_block, img_size=(32, 32, 3), compression_ratio=1.0, bottleneck_ratio=1.0, dropout=0.2, classes=100):
  
  dense_blocks = len(layers_per_block)

  print(f'Number of dense block = {dense_blocks}')
  inputs = Input((32,32,3))

  x = zero_layer(inputs, growth_rate*2, dropout)

  d=1
  for layers in layers_per_block[:-1]:
    print(f'Dense block : {d}')
    d += 1
    x = dense_block(x, layers=layers, growth_rate=growth_rate, bottleneck_ratio=bottleneck_ratio, dropout=dropout)
    x = transition_layer(x, compression_ratio=compression_ratio, dropout=dropout)
  
  print(f'Dense block : {d}')
  x = dense_block(x, layers=layers_per_block[-1], growth_rate=growth_rate, bottleneck_ratio=bottleneck_ratio, dropout=dropout)

  outputs = classification_layer(x, classes=classes)

  model = Model(inputs=inputs, outputs=outputs)

  return model

# Model blocks
def composite_function(x, filters, kernel_size):
  x1 = BatchNormalization()(x)
  x1 = ReLU()(x1)
  x1 = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x1)
  return x1

def zero_layer(x, filters, dropout):
  print('Adding zero layer')
  return Conv2D(filters, 7, padding='same', use_bias=False)(x)

def dense_layer(x, growth_rate, bottleneck_ratio, dropout):
  if bottleneck_ratio != -1:
    bottleneck_channels = bottleneck_ratio * growth_rate
    x1 = composite_function(x, bottleneck_channels, 1)
  else:
    x1 = x
  # Convolution 
  x1 = composite_function(x1, growth_rate, 3)
  x1 = Dropout(dropout)(x1)
  return x1

def dense_block(x, layers, growth_rate, bottleneck_ratio, dropout):
  print('Dense block---')
  print(f'    Adding {layers} dense layer with {bottleneck_ratio*growth_rate} bottleneck channels each')
  x1 = x
  for l in range(int(layers)):
    x2 = dense_layer(x1, growth_rate=growth_rate, bottleneck_ratio=bottleneck_ratio, dropout=dropout)
    x1 = Concatenate(axis=3)([x1, x2])
  print('-----------------------------------------------------------------')
  return x1

def transition_layer(x, compression_ratio, dropout):

  print('Adding a transition layer')
  
  compressed_channels = compression_ratio * x.shape[-1]
  
  print(f'      compressed_channels = : {compressed_channels}')
  x1 = composite_function(x, compressed_channels, 1)
  x2 = AveragePooling2D((2,2), strides=2)(x1)
  x2 = Dropout(dropout)(x2)
  print('-----------------------------------------------------------------')

  return x2

def classification_layer(x, classes):
  x1 = BatchNormalization(epsilon=1e-5)(x)
  x1 = ReLU()(x1)
  x1 = GlobalAveragePooling2D()(x1)
  x1 = Flatten()(x1)
  x1 = Dense(classes, activation=softmax)(x1)
  return x1 

