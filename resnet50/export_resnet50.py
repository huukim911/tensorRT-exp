import torchvision.models as models
import torch
import torch.onnx
import os

# load the pretrained model
resnet50 = models.resnet50(pretrained=True, progress=False)
output_onnx="resnet50_pytorch.onnx"
if not os.path.exists(output_onnx):
    # set up a dummy input tensor and export the model to ONNX
    BATCH_SIZE = 32
    dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
    torch.onnx.export(resnet50, dummy_input, output_onnx, verbose=False)

# import os

# os._exit(0) # Shut down all kernels so TRT doesn't fight with PyTorch for GPU memory

# BATCH_SIZE = 32

# import numpy as np

# USE_FP16 = True

# target_dtype = np.float16 if USE_FP16 else np.float32
# dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = np.float32)

# create engine

# step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
# if USE_FP16:
#     trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch --fp16
# else:
#     trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch