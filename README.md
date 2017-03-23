# CreateCaffeProtFromPython
A library for creating caffe prototxt models using Python
This repo will hold a Python library that will allow the creation of prototxt programmatistically.
Currently the library is under preparation, but for anyone who is interested into using Caffe Python bindings here is a simple example to do this. 
P.S. In the future I will add the corresponding code to achieve this will.

```python
# General Example
import caffe
from caffe import layers as cl
from caffe import params as cp

###############################################
#   parameters
batch_size =  10
lmdbpath= "addyourpath"
backend = cp.Data.LMDB
# Create dictionary with transform params
transformParams=dict(crop_size=224,scale=1./255, mirror=1,mean_file="Your_mean_binaryProto" ) # mirror=0 to turn off
# number of inputs
ntop=2
# Random initialization type
weight_filler= dict(type='gaussian');
###############################################


###############################################
# Create a basic conv net- CaffeNet
n = caffe.NetSpec()
n.data, n.label = cl.Data(batch_size=batch_size, backend=backend, source=lmdbpath, transform_param=transformParams, ntop=ntop)
n.conv1 = cl.Convolution(n.data, kernel_size=11, num_output=96, stride=4, weight_filler=weight_filler)
n.relu1 = cl.ReLU(n.conv1, in_place=True)
n.pool1 = cl.Pooling(n.relu1, kernel_size=2, stride=2, pool=cp.Pooling.MAX)
n.norm1 = cl.LRN(n.pool1)
n.conv2 = cl.Convolution(n.norm1, kernel_size=5, num_output=256, group=2, pad=2, weight_filler=weight_filler)
n.relu2 = cl.ReLU(n.conv2, in_place=True)
n.pool2 = cl.Pooling(n.relu2, kernel_size=3, stride=2, pool=cp.Pooling.MAX)
n.norm2 = cl.LRN(n.pool2)
n.conv3 = cl.Convolution(n.norm2, kernel_size=3, num_output=384, pad=1, weight_filler=weight_filler)
n.relu3 = cl.ReLU(n.conv3, in_place=True)
n.conv4 = cl.Convolution(n.relu3, kernel_size=3, num_output=384, group=2, pad=1, weight_filler=weight_filler)
n.relu4 = cl.ReLU(n.conv4, in_place=True)
n.conv5 = cl.Convolution(n.relu4, kernel_size=3, num_output=384, group=2, pad=1, weight_filler=weight_filler)
n.relu5 = cl.ReLU(n.conv5, in_place=True)
n.pool5= cl.Pooling(n.relu5, kernel_size=3, stride=2, pool=cp.Pooling.MAX)
n.fc6= cl.InnerProduct(n.pool5,  num_output=4096,weight_filler=weight_filler)
n.relu6 = cl.ReLU(n.fc6, in_place=True)
n.drop6 = cl.Dropout(n.relu6, dropout_ratio=0.5)
n.fc7= cl.InnerProduct(n.drop6,  num_output=4096,weight_filler=weight_filler)
n.relu7 = cl.ReLU(n.fc7, in_place=True)
n.drop7 = cl.Dropout(n.relu7, dropout_ratio=0.5)
n.fc8 = cl.InnerProduct(n.drop7,  num_output=4096,weight_filler=weight_filler)
n.loss= cl.SoftmaxWithLoss(n.fc8,n.label)
# Continue
###############################################
proto = n.to_proto()

with open('genCaffeNet.prototxt', 'w') as f:
    f.write("# ==========================\n");
    f.write("#          Generated Network\n");
    f.write("# ==========================\n");
    f.write(str(proto))

# Enjoy
```
