#import torch
import onnx
#import cv2
import os
import platform
import tvm 
import tvm.relay as relay

from PIL import Image
import numpy as np
from torchvision import transforms

NEWLIBDIR="./tvm_output_lib/"

sysstr = platform.system()
if(sysstr =="Windows"):
    libext=".dll"
else:
    libext=".so"

libpath=NEWLIBDIR+"mobildv2"+libext
if not os.path.exists(NEWLIBDIR):
    os.makedirs(NEWLIBDIR)
# https://zhuanlan.zhihu.com/p/60981432
onnx_model = onnx.load('mobilenetv2.onnx')
img = Image.open('cat.jpg').resize((244, 244))

# 对图片进行正则化处理，参数来自于官方网站

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = preprocess(img)
x = img[np.newaxis, :]

# 这里首先在PC的CPU上进行测试 所以使用LLVM进行导出
target = tvm.target.Target('llvm')

input_name = 'input'  # 这儿的名字可以用netron 查看网络结构，找到input节点的name 
shape_dict = {input_name: x.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print(sym)
# 这里利用TVM构建出优化后模型的信息
with relay.build_config(opt_level=2):
    lib = relay.build(sym, target, params=params)

dtype = 'float32'

from tvm.contrib import graph_runtime

# 下面的函数导出我们需要的动态链接库 地址可以自己定义
print("Output model files")

lib.export_library(libpath)

# -------------至此导出模型阶段已经结束--------

# 接下来我们加载导出的模型去测试导出的模型是否可以正常工作

loaded_lib = tvm.runtime.load_module(libpath)
# 这里执行的平台为CPU
ctx = tvm.cpu()

module = graph_runtime.GraphModule(lib["default"](ctx))
module.set_input("input", x)  #此处也是输入节点名字
module.run()
out_deploy = module.get_output(0).asnumpy()

def soft_max(z):
    t=np.exp(z)
    a=np.exp(z)/np.sum(t,axis=1)
    return a
print(np.argmax(soft_max(out_deploy)))