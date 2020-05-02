#!/usr/bin/env python
# coding: utf-8

# In[11]:


import keras
import tensorflow
from tensorflow.keras.layers import Conv2D, Activation, LeakyReLU, BatchNormalization , Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2DTranspose as Deconvolution2D
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


# In[13]:


get_ipython().system('pip install git+https://www.github.com/keras-team/keras-contrib.git')
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


# In[2]:


def residual(x, chan  = 64):
  input_x = x
  x = Conv2D( filters = chan, kernel_size = (3 ,3) , padding='SAME', strides = (1,1))(input_x)  ##name ='conv0',
      # .Dropout('drop', 0.5)
  x= Conv2D(filters = int(chan/2),  kernel_size = (3 ,3) ,padding='SAME', strides = (1,1) )(x)  ##name ='conv1', 
  x= Conv2D( filters = chan, kernel_size = (3 ,3) , padding='SAME', strides = (1,1) )(x)   ##name = 'conv2'
      # .Dropout('drop', 0.5)
      # .InstanceNorm('inorm')
  output_x = Add()([x , input_x])
  return output_x


# In[3]:


## Residual encoder 
def res_enc(x , chan ,name = 'none'):
  x = Conv2D(filters= chan ,kernel_size=(3 ,3),  strides = (2,2) , padding = 'SAME' )(x)   ##conv_i
  x = residual(x , chan)
  output_x= Conv2D(filters= chan ,kernel_size=(3 ,3),  strides = (1,1) , padding = 'SAME')(x)   ##conv_o
  return output_x


# In[4]:


## Residual decoder

def res_dec(x , chan , name = 'none'): 
  x = Deconvolution2D(filters= chan ,kernel_size=(3 ,3),  strides = (1,1) , padding = 'SAME' )(x)  ##conv_i
  x = residual(x , chan)
  output_x= Deconvolution2D(filters= chan ,kernel_size=(3 ,3),  strides = (2,2) , padding= 'SAME')(x)  ##conv_o
  return output_x


# In[5]:


##ARC generator 
chan = 64
def arc_generator(): 
  input_image = tensorflow.keras.Input(shape = (256 , 256,3 ))
  e0 = res_enc( input_image, chan , name = 'enc0')
  e1 = res_enc( e0 , chan*2 ,name = 'enc1' )
  e2 = res_enc( e1 , chan*4 , name = 'enc2')
  e3 = res_enc( e2, chan*8, name = 'enc3')

  d3 = res_dec (e3, chan*4 , name = 'dec3')
  Id1 = Add()([d3 , e2])
  d2 = res_dec ( Id1, chan*2 , name = 'dec2')
  Id2 = Add()([d2, e1])
  d1 = res_dec (Id2 , chan*1 , name = 'dec1')
  Id3 = Add()([d1, e0])
  d0 = res_dec (Id3 , chan*1 , name = 'dec0')
  dd = Conv2D(kernel_size= (3 , 3) , filters= 3, strides= (1,1) )(d0)
  dd = BatchNormalization()(dd)
  out_dd = LeakyReLU()(dd)
  model = tensorflow.keras.Model(inputs = input_image , outputs = out_dd )
  return model


# In[6]:


gan = arc_generator()


# In[8]:


def descriminator_model():
  input_image = keras.Input(shape = (256, 256, 3))
  x = Conv2D(filters= chan , kernel_size=(4, 4) , strides= (2,2), padding= 'SAME')(input_image)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  e0 = res_enc(x , chan )
  e1 = res_enc(e0 , chan*2)
  e2 = res_enc(e1 , chan*4)
  e3 = res_enc(e2 , chan*8)
  out_x = Conv2D(filters=1 , kernel_size= (4, 4), padding= 'SAME',strides= (1,1))(e3)
  model = tensorflow.keras.Model(input_image , out_x )
  return model 


# In[9]:


desc_model = descriminator_model()


# In[ ]:




