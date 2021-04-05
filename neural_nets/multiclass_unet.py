# converts into already normalized images (can accept float32 or float64) / 255.0 -- normalized range [0,1]
def unet_model(IMGH, IMGW, CHANNELS, CLASSES):

  # begin with 1st layer input = 128x128x3 aka. 128x128 RGB image (3 channels)
  inputs = Input((IMGH, IMGW, CHANNELS))
  s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
  s = inputs

  c1 = Conv2D(filters= 16, kernel_size=3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(s)
  c1 = Dropout(rate= 0.1)(c1)
  c1 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c1)

  # move into 2nd layer : convolution 1
  m1 = MaxPooling2D(pool_size= 2)(c1)
  # m1 becomes input to c2
  c2 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m1)
  c2 = Dropout(rate= 0.1)(c2)
  c2 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c2)

  # move into 3rd layer : convolution 2
  m2 = MaxPooling2D(pool_size= 2)(c2)
  c3 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m2)
  c3 = Dropout(rate= 0.2)(c3)
  c3 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c3)

  # move into 4th layer : convolution 3
  m3 = MaxPooling2D(pool_size= 2)(c3)
  c4 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m3)
  c4 = Dropout(rate= 0.2)(c4)
  c4 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c4)

  # move into 5th layer : convolution 4 (last convolution step)
  m4 = MaxPooling2D(pool_size= 2)(c4)
  # deepest layer, shape = 8x8x256
  c5 = Conv2D(filters= 256, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(m4)
  c5 = Dropout(rate= 0.3)(c5)
  c5 = Conv2D(filters= 256, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c5)

  # move back into 4th layer : upconvolution 1
  u1 = Conv2DTranspose(filters= 128, kernel_size= 2, strides= 2, padding= 'same')(c5)
  u1 = concatenate([u1, c4])
  c6 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u1)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(filters= 128, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c6)

  # move back into 3rd layer : upconvolution 2
  u2 = Conv2DTranspose(filters= 64, kernel_size= 2, strides= 2, padding= 'same')(c6)
  u2 = concatenate([u2, c3])
  c7 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u2)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(filters= 64, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c7)

  # move back into 2nd layer : upconvolution 3
  u3 = Conv2DTranspose(filters= 32, kernel_size= 2, strides= 2, padding= 'same')(c7)
  u3 = concatenate([u3, c2])
  c8 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u3)
  c8 = Dropout(0.1)(c8)
  c8 = Conv2D(filters= 32, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c8)

  # move back into 1st layer : upconvolution 4
  u4 = Conv2DTranspose(filters= 16, kernel_size= 2, strides= 2, padding= 'same')(c8)
  u4 = concatenate([u4, c1], axis= 3)
  c9 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(u4)
  c9 = Dropout(0.1)(c9)
  c9 = Conv2D(filters= 16, kernel_size= 3, activation= 'relu', kernel_initializer= 'he_normal', padding= 'same')(c9)

  # return a classification model with sigmoid activation
  outputs = Conv2D(CLASSES, (1,1), activation= 'softmax')(c9)

  # define the beginning and starting points of the model aka. what you feed in and what you expect it to return
  model = Model(inputs = [inputs], outputs= [outputs])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  return model
