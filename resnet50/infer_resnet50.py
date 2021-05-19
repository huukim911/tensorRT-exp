# Runtime
import numpy as np
BATCH_SIZE = 32

USE_FP16 = True

target_dtype = np.float16 if USE_FP16 else np.float32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = np.float32)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

f = open("resnet_engine_pytorch.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * dummy_input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output

print("Warming up...")

predict(dummy_input_batch)

print("Done warming up!")


pred = predict(dummy_input_batch)
print(pred)