from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Lambda, MaxPooling3D

from models import settings
from models.blocks.blocks_3d import DoubleConvBlock, Up3DBlock

class UNetPlusPlus:
    def __init__(self, input_shape, n_classes, n_filters=[32,64,128,256,512], dropout_rate=0.5, deep_supervision=False):
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
        img_input = Input(shape=self.input_shape)

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