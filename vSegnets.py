from tensorflow.keras import backend as K
#from keras.engine import Input, Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Add, Conv3DTranspose, Lambda,Dropout
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import softmax


from models.blocks.blocks_3d import DoubleConvBlock, AttnUp3DBlock, Up3DBlock, R2ConvBlock, SEAttnBlock, BasicConvBlock, ResConvBlock
from models import settings

####################################
############## U-net ###############
####################################

def convBlock(inputLayer, filters, ks=3, padding='same', strides=(1,1,1), bn=True):
    x = Conv3D(filters, (ks,ks,ks), padding=padding, strides=strides)(inputLayer)
    if bn: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def upsample(filters, poolSize=(2,2,2), ks=2, strides=(2,2,2), deconvolution=True):
    if deconvolution:
        return Conv3DTranspose(filters=filters, kernel_size=(ks,ks,ks), strides=strides)
    else:
        return UpSampling3D(size=poolSize)



def _residual_block(input, F):
    x = Conv3D(F, (3, 3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(F, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    output = Add()([x, input])
    output = Activation('relu')(output)
    return output

def unet(nLabels, activation, depth=4, baseFilters=32):
  inputs = Input((None, None, None,1))
  x = inputs
  levels = list()
  for d in range(depth):
      l1 = convBlock(x, baseFilters*(2**d))
      l2 = convBlock(l1, baseFilters*(2**d)*2)
      if d < depth-1:
          x = MaxPooling3D((2,2,2))(l2)
          levels.append([l1, l2, x])
      else:
          x = l2
          levels.append([l1, l2])
  for d in range(depth-2, -1, -1):
      upsampled = upsample(x.shape[-1])(x)
      concat = concatenate([upsampled, levels[d][1]])
      x = convBlock(concat, levels[d][1].shape[-1])
      x = convBlock(x, levels[d][1].shape[-1]) 
  outputs = Conv3D(nLabels, (1, 1, 1), activation=activation)(x)
  model = Model(inputs=inputs, outputs=outputs)
  return model



# def unet(nLabels, activation, depth=4, baseFilters=32):
#   inputs = Input((None, None, None,1))
#   x = inputs
#   levels = list()
#   for d in range(depth):
#       l1 = convBlock(x, baseFilters*(2**d))
#       r1 = _residual_block(l1, baseFilters*(2**d))
#       # r2 = _residual_block(r1, baseFilters*(2**d))
#       l2 = convBlock(r1, baseFilters*(2**d)*2)
#       r3 = _residual_block(l2, baseFilters*(2**d)*2)
#       # r4 = _residual_block(r3, baseFilters*(2**d)*2)
#       if d < depth-1:
#           x = MaxPooling3D((2,2,2))(r3)
#           levels.append([r1, r3, x])
#       else:
#           x = l2
#           levels.append([r1, r3])
#   for d in range(depth-2, -1, -1):
#       upsampled = upsample(x._keras_shape[-1])(x)
#       concat = concatenate([upsampled, levels[d][1]])
#       x = convBlock(concat, levels[d][1]._keras_shape[-1])
#       r1 = _residual_block(x, levels[d][1]._keras_shape[-1])
#       # r2 = _residual_block(r1, levels[d][1]._keras_shape[-1])
#       x = convBlock(r1, levels[d][1]._keras_shape[-1]) 
#       r3 = _residual_block(x, levels[d][1]._keras_shape[-1])
#       # r4 = _residual_block(r3, levels[d][1]._keras_shape[-1])
#   outputs = Conv3D(nLabels, (1, 1, 1), activation=activation)(r3)
#   model = Model(inputs=inputs, outputs=outputs)
#   return model



class UNetPlusPlus:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0, deep_supervision=False):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.deep_supervision = deep_supervision

        self.n_levels, self.n_stages = len(n_filters), len(n_filters)

    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1)) #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)
        
        features = []

        encoder_features = []

        for n in range(self.n_levels - 1):
            feature = DoubleConvBlock(self.n_filters[n], self.dropout_rate)(x)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = DoubleConvBlock(self.n_filters[-1], self.dropout_rate)(x)
        encoder_features.append(x)

        features.append(encoder_features)

        for m in range(1, self.n_stages):
            decoder_features = []
            for n in range(self.n_levels - m):
                skip = [features[i][n] for i in range(m)]
                x = Up3DBlock(settings.UP_TYPE)(features[m - 1][n + 1], skip)
                x = DoubleConvBlock(self.n_filters[n], self.dropout_rate)(x)
                decoder_features.append(x)
            features.append(decoder_features)

        if self.deep_supervision:
            outputs = []
            for i in range(1, self.n_stages):
                output = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(features[i][0])
                outputs.append(output)
        else:
            outputs = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(features[-1][0])            
        
        model = Model(inputs=img_input, outputs=outputs)
        return model

class R2UNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        self.n_levels = len(n_filters)

    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1)) #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)

        encoder_features = []

        for i in range(self.n_levels - 1):
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            feature = R2ConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = BasicConvBlock(self.n_filters[-1], self.dropout_rate)(x)
        x = R2ConvBlock(self.n_filters[-1], self.dropout_rate)(x)

        for i in reversed(range(self.n_levels - 1)):
            x = Up3DBlock(settings.UP_TYPE)(x, [encoder_features[i]])
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = R2ConvBlock(self.n_filters[i], self.dropout_rate)(x)

        unet_output = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(x)

        model = Model(inputs=img_input, outputs=unet_output)
        return model

class SEUNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.0, se_reduction=16):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.se_reduction = se_reduction

        self.n_levels = len(n_filters)

    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1)) #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)

        encoder_features = []

        for i in range(self.n_levels - 1):
            feature = DoubleConvBlock(self.n_filters[i], self.dropout_rate)(x)
            feature = SEAttnBlock(self.se_reduction)(feature)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = DoubleConvBlock(self.n_filters[-1], self.dropout_rate)(x)

        for i in reversed(range(self.n_levels - 1)):
            x = Up3DBlock(settings.UP_TYPE)(x, [encoder_features[i]])
            x = DoubleConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = SEAttnBlock(self.se_reduction)(x)

        unet_output = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(x)

        model = Model(inputs=img_input, outputs=unet_output)
        return model


class R2AttentionUNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        self.n_levels = len(n_filters)

    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1)) #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)

        encoder_features = []

        for i in range(self.n_levels - 1):
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            feature = R2ConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = BasicConvBlock(self.n_filters[-1], self.dropout_rate)(x)
        x = R2ConvBlock(self.n_filters[-1], self.dropout_rate)(x)

        for i in reversed(range(1, self.n_levels - 1)):
            x = AttnUp3DBlock(self.n_filters[i])(x, encoder_features[i])
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = R2ConvBlock(self.n_filters[i], self.dropout_rate)(x)

        # no attention in the last level to save memory
        x = Up3DBlock(settings.UP_TYPE)(x, [encoder_features[0]])
        x = BasicConvBlock(self.n_filters[0], self.dropout_rate)(x)
        x = R2ConvBlock(self.n_filters[0], self.dropout_rate)(x)

        unet_output = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(x)

        model = Model(inputs=img_input, outputs=unet_output)
        return model

class AttentionUNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.0):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.n_levels = len(n_filters)
        
    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1))  #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)

        encoder_features = []

        for i in range(self.n_levels - 1):
            feature = DoubleConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = DoubleConvBlock(self.n_filters[-1], self.dropout_rate)(x)

        for i in reversed(range(1, self.n_levels - 1)):
            x = AttnUp3DBlock(self.n_filters[i])(x, encoder_features[i])
            x = DoubleConvBlock(self.n_filters[i], self.dropout_rate)(x)

        # no attention in the last level to save memory
        x = Up3DBlock(settings.UP_TYPE)(x, [encoder_features[0]])
        x = DoubleConvBlock(self.n_filters[0], self.dropout_rate)(x)

        unet_output = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(x)

        model = Model(inputs=img_input, outputs=unet_output)
        return model


class ResUNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        self.n_levels = len(n_filters)

    def compile_model(self, optimizer=None, loss_function=None, metrics=None, checkpoint=None):
        model = self.build_model()
        if checkpoint is None:
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        else:
            model.load_weights(checkpoint)
        return model

    def build_model(self):
        img_input = Input((None, None, None,1))  #Input(shape=self.input_shape)

        x = Lambda(lambda x: x)(img_input)

        encoder_features = []

        for i in range(self.n_levels - 1):
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            feature = ResConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = MaxPooling3D(2, strides=2)(feature)
            encoder_features.append(feature)

        x = BasicConvBlock(self.n_filters[-1], self.dropout_rate)(x)
        x = ResConvBlock(self.n_filters[-1], self.dropout_rate)(x)

        for i in reversed(range(self.n_levels - 1)):
            x = Up3DBlock(settings.UP_TYPE)(x, [encoder_features[i]])
            x = BasicConvBlock(self.n_filters[i], self.dropout_rate)(x)
            x = ResConvBlock(self.n_filters[i], self.dropout_rate)(x)

        unet_output = Conv3D(self.n_classes, kernel_size=1, activation='softmax', kernel_initializer=settings.KERNEL_INIT, padding='same')(x)

        model = Model(inputs=img_input, outputs=unet_output)
        return model

