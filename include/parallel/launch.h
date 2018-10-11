/*
 * Parallelising JVM Compiler
 *
 * Copyright 2010 Peter Calvert, University of Cambridge
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PARALLEL_LAUNCH
#define PARALLEL_LAUNCH

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <jni.h>

#define MIN(A,B)         (  ((A) < (B)) ? (A) : (B)  )
#define DIV_UP(A,B)      (  (((A) - 1) / (B)) + 1    )
#define ROUND_UP(A,B)    (  DIV_UP(A,B) * (B)        )
#define ROUND_DOWN(A,B)  (  ((A) / (B)) * (B)        )

/**
 *    Filename: launch.h
 * Description: Provides functions for optimizing launch dimensions.
 */

/**
 * Properties of the device being used.
 */
cudaDeviceProp deviceProp;

bool convertDoubles;

/**
 * Initialises CUDA device on library load.
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  // FIXME: Use something better than the default device (cudaChooseDevice).
  cudaSetDevice(0);
  
  // Get device properties
  cudaGetDeviceProperties(&deviceProp, 0);
  
  // Determine whether to alter doubles.
  double original, convert;
  
  original = convert = 3.1415926535897932384626433832795;
  
  cudaSetDoubleForDevice(&convert);
  
  convertDoubles = (original != convert);
  
  return JNI_VERSION_1_4;
}

/**
 * Determines the best launch grid/block dimensions for a given requirement and
 * register count for the kernel. It returns the number of iterations that
 * should be performed in each dimension on each invocation.
 */
dim3 calculateDimensions(void *func, dim3 *gridDim, dim3 *blockDim, dim3 required) {
  // Get function attributes.
  cudaFuncAttributes funcAttr;
  cudaFuncGetAttributes(&funcAttr, (const char *) func);
  
  // Check that computation is required.
  if((required.x == 0) || (required.y == 0) || (required.z == 0)) {
    return dim3(0, 0, 0);
  }
  
  // Maximum threads per block for this particular kernel.
  int maxThreads = ROUND_DOWN(MIN(
    deviceProp.maxThreadsPerBlock,
    funcAttr.maxThreadsPerBlock
  ), deviceProp.warpSize);
  
  // Stuck in X direction.
  if(required.x > maxThreads) {
    int iteration = ((required.x - 1) / maxThreads) + 1;
    
    blockDim->x = ROUND_UP(DIV_UP(required.x, iteration), deviceProp.warpSize);
    blockDim->y = 1;
    blockDim->z = 1;
  // Can have 2D block.
  } else {
    blockDim->x = ROUND_UP(required.x, deviceProp.warpSize);
    blockDim->y = MIN(maxThreads / blockDim->x, required.y);
    blockDim->z = 1;
  }
  
  // Grid Dimensions
  gridDim->x = MIN(DIV_UP(required.x, blockDim->x), deviceProp.maxGridSize[0]);
  gridDim->y = MIN(DIV_UP(required.y, blockDim->y), deviceProp.maxGridSize[1]);
  gridDim->z = MIN(DIV_UP(required.z, blockDim->z), deviceProp.maxGridSize[2]);
  
  // Result (how much we can perform in an iteration).
  return dim3(
    gridDim->x * blockDim->x,
    gridDim->y * blockDim->y,
    gridDim->z * blockDim->z
  );
}

#endif
