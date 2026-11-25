

from tensorflow.keras.layers import Conv3D, Dropout, Activation, BatchNormalization, Add, GlobalAveragePooling3D
from tensorflow.keras.layers import Reshape, multiply, Conv3DTranspose, UpSampling3D, concatenate, Dense, add
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from models import settings 

def BasicConvBlock(n_filters, dropout_rate, kernel_size=3):
    seq = Sequential([
        Conv3D(n_filters, kernel_size, kernel_initializer=settings.KERNEL_INIT, padding='same'),
        BatchNormalization(),
        Activation(settings.ACT),
        Dropout(dropout_rate)
    ])
    return seq


def DoubleConvBlock(n_filters, dropout_rate, kernel_size=3):
    def layer(x):
        x = BasicConvBlock(n_filters, dropout_rate, kernel_size)(x)
        x = BasicConvBlock(n_filters, dropout_rate, kernel_size)(x)
        return x
    return layer


# Recurrent Conv Block
# https://github.com/TsukamotoShuchi/RCNN/blob/master/rcnn.py
def RecConvBlock(n_filters, dropout_rate, kernel_size=3):
    def layer(x):
        # feed-forward conv
        fw_conv = Conv3D(n_filters, kernel_size, kernel_initializer=settings.KERNEL_INIT, padding='same')
        # recurrent conv
        rc_conv = Conv3D(n_filters, kernel_size, kernel_initializer=settings.KERNEL_INIT, padding='same')
        
        x0 = fw_conv(x)
        x0 = BatchNormalization()(x0)
        x0 = Activation(settings.ACT)(x0)

        x1 = Add()([fw_conv(x0), rc_conv(x)]) 
        x1 = BatchNormalization()(x1)
        x1 = Activation(settings.ACT)(x1)

        x2 = Add()([fw_conv(x1), rc_conv(x)])    
        x2 = BatchNormalization()(x2)
        x2 = Activation(settings.ACT)(x2)

        x2 = Dropout(dropout_rate)(x2)
        return x2
    return layer



# Residual Recurrent Conv Block
def R2ConvBlock(n_filters, dropout_rate, kernel_size=3):
    def layer(x):
        output = RecConvBlock(n_filters, dropout_rate, kernel_size)(x)
        output = RecConvBlock(n_filters, dropout_rate, kernel_size)(output)
        output = Add()([x, output])
        return output
    return layer


# Residual Conv Block
def ResConvBlock(n_filters, dropout_rate, kernel_size=3):
    def layer(x):
        output = BasicConvBlock(n_filters, dropout_rate, kernel_size)(x)
        output = BasicConvBlock(n_filters, dropout_rate, kernel_size)(output)
        output = Add()([x, output])
        return output
    return layer


# Grid Attention Conv Block
# Channel Squeeze and Spatial Excitation
def GridAttnBlock(n_filters):
    def layer(x, g):
        theta_x = Conv3D(filters=n_filters, kernel_size=2, strides=2, kernel_initializer=settings.KERNEL_INIT, padding='same')(x)
        theta_x = BatchNormalization()(theta_x)

        phi_g = Conv3D(filters=n_filters, kernel_size=1, strides=1, kernel_initializer=settings.KERNEL_INIT, padding='same')(g)
        phi_g = BatchNormalization()(phi_g)

        f = Add()([theta_x, phi_g])
        f = Activation("relu")(f)

        psi_f = Conv3D(filters=1, kernel_size=1, strides=1, kernel_initializer=settings.KERNEL_INIT, padding='same')(f)
        psi_f = BatchNormalization()(psi_f)
        attn_map = Activation("sigmoid")(psi_f)
        return attn_map
    return layer

# Spatial Squeeze and Channel Excitation Attention Conv Block
def SEAttnBlock(reduction=16):
    def layer(x):
        dim = K.int_shape(x)[-1]

        sq = GlobalAveragePooling3D()(x)

        ex = Dense(dim // reduction)(sq)
        ex = Activation("relu")(ex)
        ex = Dense(dim)(ex)
        ex = Activation("sigmoid")(ex)

        attn_map = Reshape((1, 1, dim))(ex)

        x = multiply([x, attn_map])
        return x
    return layer


def Up3DBlock(up_type):
    if up_type == "transpose":
        return TransConv3DBlock()
    elif up_type == "upsample":
        return UpSample3DBlock()
    else:
        return Lambda(lambda x: x) # Identity


def TransConv3DBlock():
    def layer(x, skip):
        dim = K.int_shape(x)[-1]
        x = Conv3DTranspose(dim, kernel_size=2, strides=2, kernel_initializer=settings.KERNEL_INIT, padding='same')(x)
        skip.append(x)
        x = concatenate(skip)
        return x
    return layer

def UpSample3DBlock():
    def layer(x, skip):
        x = UpSampling3D(2)(x)
        skip.append(x)
        x = concatenate(skip)
        return x
    return layer


def AttnUp3DBlock(n_filters):
    def layer(x, skip):
        attn_map = GridAttnBlock(n_filters)(skip, x)
        attn_map = UpSampling3D(2)(attn_map)
        out = multiply([skip, attn_map])
        return out
    return layer



