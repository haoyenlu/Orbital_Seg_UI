from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Lambda, MaxPooling3D

from models import settings
from models.blocks.blocks_3d import DoubleConvBlock, SEAttnBlock, Up3DBlock

# Squeeze-Excitation UNet
# https://arxiv.org/pdf/1808.08127.pdf
class SEUNet:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.5, se_reduction=16):
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
        img_input = Input(shape=self.input_shape)

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