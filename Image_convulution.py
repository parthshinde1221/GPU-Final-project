from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time
import json

@cuda.jit
def conv_per_img(img, kernel, out):

    sImage = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype='float32')
    sKernel = cuda.shared.array(shape=(MASK_SIZE, MASK_SIZE), dtype='float32')
    
    # Output vector size
    outSize_x, outSize_y = out.shape[0], out.shape[1]
    
    # Input vector size
    inSize_x, inSize_y = img.shape[0], img.shape[1]

    # Location of thread in the grid, used to get input and output index
    x, y = cuda.grid(2)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    
    # If location outsie image bounds
    if x > inSize_x or y > inSize_y:
        return

    # Move image data to shared memory location. Memory coalescent friendly
    sImage[tx, ty] = img[x, y]
    
    # Move kernel vector to shared memory
    if tx < MASK_SIZE and ty < MASK_SIZE:
        sKernel[tx, ty] = kernel[tx, ty]
    
    cuda.syncthreads()


    # One thread per input element
    # Move the mask and atomic add to corresponding output element
    for i in range(MASK_SIZE):
        # Move mask
        for j in range(MASK_SIZE):
            outIdx_x, outIdx_y = x - i, y - j
            if outIdx_x < 0 or outIdx_x >= outSize_x or outIdx_y < 0 or outIdx_y >= outSize_y:
                continue
            cuda.atomic.add(out, (outIdx_x, outIdx_y), sImage[tx, ty]*sKernel[i, j])


def convolve2d_cpu(input_array, kernel):
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_array.shape

    # Calculate the shape of the output array
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Initialize the output array
    output_array = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Element-wise multiplication and sum
            output_array[i, j] = np.sum(input_array[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output_array

TILE_SIZE = 16
CPU_TIMES = {}
GPU_TIMES = {}

MASK_SIZE = 3


## Testing if code is correct
IMAGE_SIZE_x = 500
IMAGE_SIZE_y = 500

#Initializing input vectors
I = np.random.rand(IMAGE_SIZE_x, IMAGE_SIZE_y) * 255.
K = np.random.randn(MASK_SIZE,MASK_SIZE)

# Output vector, size = Input size - kernel size + 1
outsize = [I.shape[0] - MASK_SIZE + 1, I.shape[1] - MASK_SIZE + 1]
O = np.zeros((outsize[0],outsize[1]))

# Number of threads based on input size
dimBlock = (TILE_SIZE, TILE_SIZE, 1)
dimGrid = ((I.shape[0] + TILE_SIZE - 1) // TILE_SIZE, (I.shape[1] + TILE_SIZE - 1) // TILE_SIZE)

d_I = cuda.to_device(I)
d_O = cuda.to_device(O)
d_K = cuda.to_device(K)

conv_per_img[dimGrid, dimBlock](d_I, d_K, d_O)

h_O = d_O.copy_to_host()

# Benchmark output matrix to test CUDA implementation
bench_O = convolve2d_cpu(I, K)

if np.isclose(h_O, bench_O, 1e-2).all():
    print('test passed')
else:
    print("test failed")

## Creating a test output
img = Image.open("UCRBellTower1.jpg").convert('L')
imgArr = np.asarray(img) / 255.
sobel_kernel_x = np.array([[1,0,-1],[2, 0, -2],[1, 0, -1]])
sobel_kernel_y = sobel_kernel_x.transpose()

outsize = [imgArr.shape[0] - MASK_SIZE + 1, imgArr.shape[1] - MASK_SIZE + 1]
O = np.zeros((outsize[0],outsize[1]))

dimBlock = (TILE_SIZE, TILE_SIZE, 1)
dimGrid = ((I.shape[0] + TILE_SIZE - 1) // TILE_SIZE, (I.shape[1] + TILE_SIZE - 1) // TILE_SIZE)

d_I = cuda.to_device(imgArr)
d_O = cuda.to_device(O)
d_K = cuda.to_device(sobel_kernel_x)

conv_per_img[dimGrid, dimBlock](d_I, d_K, d_O)

h_O = d_O.copy_to_host()
h_O = h_O * 255.
plt.imsave('UCR_IMAGE_SOBEL.jpg', h_O)

## Calculating speedup for various problem sizes

for IMAGE_SIZE_x in [100, 1000, 10000]:
    for IMAGE_SIZE_y in [100, 1000, 10000]:
        for MASK_SIZE in [3, 5, 10]:

            I = np.random.rand(IMAGE_SIZE_x, IMAGE_SIZE_y) * 255.
            K = np.random.randn(MASK_SIZE,MASK_SIZE)


            outsize = [I.shape[0] - MASK_SIZE + 1, I.shape[1] - MASK_SIZE + 1]
            O = np.zeros((outsize[0],outsize[1]))

            dimBlock = (TILE_SIZE, TILE_SIZE, 1)
            dimGrid = ((I.shape[0] + TILE_SIZE - 1) // TILE_SIZE, (I.shape[1] + TILE_SIZE - 1) // TILE_SIZE)

            d_I = cuda.to_device(I)
            d_O = cuda.to_device(O)
            d_K = cuda.to_device(K)

            start_time = time.time()
            conv_per_img[dimGrid, dimBlock](d_I, d_K, d_O)
            gpu_time = time.time() - start_time

            h_O = d_O.copy_to_host()

            start_time = time.time()
            bench_O = convolve2d_cpu(I, K)
            cpu_time = time.time() - start_time

            print(str(IMAGE_SIZE_x) + 'x' + str(IMAGE_SIZE_y) + ' filter ' + str(MASK_SIZE))
            
            GPU_TIMES[str(IMAGE_SIZE_x) + 'x' + str(IMAGE_SIZE_y) + '_' + str(MASK_SIZE)] = gpu_time
            CPU_TIMES[str(IMAGE_SIZE_x) + 'x' + str(IMAGE_SIZE_y) + '_' + str(MASK_SIZE)] = cpu_time

with open("CPU_TIMES.json", "w") as outfile: 
    json.dump(CPU_TIMES, outfile)

with open("GPU_TIMES.json", "w") as outfile: 
    json.dump(GPU_TIMES, outfile)
