import mxnet as mx


def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
         name=None, with_relu=True, dilate=(1, 1), num_group=1):

    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name=name,dilate=dilate,num_group=num_group)
    if with_relu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='relu_%s_relu' % (name))

    return conv


def fullconnect(data,num_filter,dropout,name):
    data = mx.sym.FullyConnected(data=data, num_hidden=num_filter,name=name)
    data = mx.sym.Activation(data=data, act_type='relu', name='relu_%s_relu' % (name))
    if 0!= dropout:
        data = mx.sym.Dropout(data=data, p=dropout)
    return data


def get_conv(data, suffix=""):
    data1 = Conv(data, 96, kernel=(11, 11), stride=(4, 4), name="conv1"+suffix)
    data1 = mx.sym.Pooling(data=data1, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool1"+suffix)
    data1 = mx.sym.LRN(data1,nsize=5,alpha=0.0001,beta=0.75,name="norm1"+suffix,knorm=1)

    data1 = Conv(data1, 256, kernel=(5, 5), name="conv2"+suffix,num_group=2,pad=(2,2))
    data1 = mx.sym.Pooling(data=data1, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool2"+suffix)
    data1 = mx.sym.LRN(data1,nsize=5,alpha=0.0001,beta=0.75,name="norm2"+suffix,knorm=1)

    data1 = Conv(data1, 384, kernel=(3, 3), name="conv3"+suffix,pad=(1,1))

    data1 = Conv(data1, 384, kernel=(3, 3), name="conv4"+suffix,num_group=2,pad=(1,1))

    data1 = Conv(data1, 256, kernel=(3, 3), name="conv5"+suffix,num_group=2,pad=(1,1))
    data1 = mx.sym.Pooling(data=data1, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool5"+suffix)
    data = mx.sym.Flatten(data=data1)
    return data


def get_train_symbol():
    data = mx.symbol.Variable(name="target")
    data = get_conv(data,"")
    image = mx.symbol.Variable(name="image")
    data1 = get_conv(image,"_p")
    data = mx.sym.Concat(data,data1)
    data = mx.sym.Flatten(data=data)

    data = fullconnect(data, 4096, 0.5, "fc6_new")
    data = fullconnect(data,4096,0.5,"fc7_new")
    data = fullconnect(data,4096,0.5,"fc7_newb")
    data = fullconnect(data, 4, 0, "fc8_shapes")

    label = mx.symbol.Variable(name="label")
    data = mx.symbol.smooth_l1(name="loc_loss_", \
                                    data=(data - label), scalar=1.0)
    data = mx.symbol.MakeLoss(data, grad_scale=1., \
                                  normalization='valid', name="loc_loss")
    return data


def get_symbol():
    sym  = get_train_symbol()
    internel = sym.get_internals()
    return internel["fc8_shapes_output"]


if '__main__' == __name__:
    from mxplot import plot_symbol
    plot_symbol(get_symbol(),shape={"image":(1,3,227,227),"target":(1,3,227,227)})
