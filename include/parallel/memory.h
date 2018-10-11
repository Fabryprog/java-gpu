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
 
#ifndef PARALLEL_MEMORY
#define PARALLEL_MEMORY

#include <map>
#include <cuda.h>

#define SINGLE_ALLOC

#ifdef SINGLE_ALLOC
__constant__ char *devOffset;
char *HdevOffset;

template<typename T> __device__ T* DEVPTR(T* ptr) {
  return (T*) (((size_t) ptr) + devOffset);
}

template<typename T> T* HDEVPTR(T* ptr) {
  return (T*) (((size_t) ptr) + HdevOffset);
}
#else
template<typename T> __device__ T* DEVPTR(T* ptr) {
  return ptr;
}

template<typename T> T* HDEVPTR(T* ptr) {
  return ptr;
}
#endif

/**
 *    Filename: memory.h
 * Description: Provides joint memory allocator for device and host memory.
 */

/**
 * Counts of memory allocated.
 */
size_t totalHost = 0, totalDevice = 0, countDevice = 0;

class Region {
  size_t  size;
  int     count;
  int     available;
  void   *host;
  void   *device;
  bool    copy, deviceOnly;
  
public:
  Region *next;
  
  Region(size_t size, int count, Region*& list) {
    this->size       = size;
    this->count      = available = count;
    this->deviceOnly = false;
    this->host       = malloc(size * count);
        
    // FIXME: Proper error handling
    if(this->host == NULL) {
      fprintf(stderr, "ERROR: Could not allocate %ld bytes on host (allocated %ld).\n", size * count, totalHost);
    }
    
#ifndef SINGLE_ALLOC
    if(cudaMalloc(&this->device, size * count) != cudaSuccess) {
      fprintf(stderr, "ERROR: Could not allocate %ld bytes on device (allocated %ld in %ld).\n", size * count, totalDevice, countDevice);
      unsigned int free, total;
      cuMemGetInfo(&free, &total);
      fprintf(stderr, "       Should be %d bytes free (out of %d).\n", free, total);
    }
#else
    this->device = (void *) totalDevice;
#endif
    
    totalHost   += size * count;
    totalDevice += size * count;
    countDevice++;
    
    this->next = list;
    list       = this;
  }
  
  Region(size_t size, int count, void *host, Region*& list) {
    this->size       = size;
    this->count      = available = count;
    this->deviceOnly = true;
    this->host       = host;
    
#ifndef SINGLE_ALLOC
    // FIXME: Proper error handling
    if(cudaMalloc(&this->device, size * count) != cudaSuccess) {
      fprintf(stderr, "ERROR: Could not allocate %ld bytes on device (allocated %ld in %ld).\n", size * count, totalDevice, countDevice);
      unsigned int free, total;
      cuMemGetInfo(&free, &total);
      fprintf(stderr, "       Should be %d bytes free (out of %d).\n", free, total);
    }
#else
    this->device = (void *) totalDevice;
#endif
    
    totalDevice += size * count;
    countDevice++;
    
    this->next = list;
    list       = this;
  }
  
  template <typename T>
  bool allocate(Object<T> &object, bool copy) {
    // Wrong size or no space available.
    if((sizeof(T) != size) || (available == 0)) {
      return false;
    }
    
    // Allocate
    available--;
    this->copy |= copy;
    
    object.host   = (T*) host   + available;
    object.device = (T*) device + available;
    
    return true;
  }
  
  template <typename T>
  bool allocate(Array<T> &object, bool copy) {
    // Wrong size or no space available.
    if((sizeof(T) != size) || (available < object.length)) {
      return false;
    }
    
    // Allocate
    available -= object.length;
    this->copy |= copy;
    
    object.host   = (T*) host   + available;
    object.device = (T*) device + available;
    
    return true;
  }
  
  void copyToDevice() {
    cudaMemcpy(HDEVPTR(device), host, size * count, cudaMemcpyHostToDevice);
  }
  
  void copyToHost() {
    if(copy) {
      cudaMemcpy(host, HDEVPTR(device), size * count, cudaMemcpyDeviceToHost);
    }
  }
  
  ~Region() {
    if(!deviceOnly) {
      free(host);
      totalHost -= size * count;
    }

#ifndef SINGLE_ALLOC
    cudaFree(device);
#endif

    totalDevice -= size * count;
    countDevice--;
  }
};

#define MAX_SIZE  100

class Allocator {
  static Region *regions[MAX_SIZE];
  
  template <typename T>
  static void prepare(Array<T>& object) {
    // Nothing
  }
  
  template <typename T>
  static void prepare(Array<Object<T> >& object) {
    new Region(sizeof(T), object.length, regions[sizeof(T)]);
  }
  
  template <typename T>
  static void prepare(Array<Array<T> >& object) {
    // TODO: Anything clever?
  }

public:
  template <typename T>
  static void allocateDual(Object<T>& object, bool copy) {
    Region *current = regions[sizeof(T)];
    
    // Try existing regions.
    while(current != NULL) {
      if(current->allocate(object, copy)) {
        return;
      }
      
      current = current->next;
    }
    
    // Create new `singleton' region.
    current = new Region(sizeof(T), 1, regions[sizeof(T)]);
    current->allocate(object, copy);
  }
  
  template <typename T>
  static void allocateDevice(Object<T>& object, bool copy) {
    // Create new `singleton' region.
    Region *current = new Region(sizeof(T), 1, object.host, regions[sizeof(T)]);
    current->allocate(object, copy);
  }
  
  template <typename T>
  static void allocateDual(Array<T>& object, bool copy) {
    Region *current = regions[sizeof(T)];
    
    // Try existing regions.
    while(current != NULL) {
      if(current->allocate(object, copy)) {
        return;
      }
      
      current = current->next;
    }
    
    // Create new `singleton' region.
    current = new Region(sizeof(T), object.length, regions[sizeof(T)]);
    current->allocate(object, copy);
    
    // Prepare for elements
    prepare(object);
  }
  
  template <typename T>
  static void allocateDevice(Array<T>& object, bool copy) {
    // Create new `singleton' region.
    Region *current = new Region(
      sizeof(T),
      object.length,
      object.host,
      regions[sizeof(T)]
    );
    
    current->allocate(object, copy);
    
    // Prepare for elements
    prepare(object);
  }
  
  static void copyToDevice(JNIEnv *env) {
#ifdef SINGLE_ALLOC
    cudaError_t err = cudaMalloc(&HdevOffset, totalDevice);
    
    if(err != cudaSuccess) {
      env->FatalError(cudaGetErrorString(err));
    }
    
    cudaMemcpyToSymbol("devOffset", &HdevOffset, sizeof(char *));
#endif

    for(int i = 0; i < MAX_SIZE; i++) {
      Region *region = regions[i];
      
      while(region != NULL) {
        region->copyToDevice();
        
        region = region->next;
      }
    }
  }
  
  static void copyToHost() {
    for(int i = 0; i < MAX_SIZE; i++) {
      Region *region = regions[i];
      
      while(region != NULL) {
        region->copyToHost();
        
        region = region->next;
      }
    }
  }
  
  static void freeAll() {
    for(int i = 0; i < MAX_SIZE; i++) {
      Region *region = regions[i];
      
      regions[i] = NULL;
      
      while(region != NULL) {
        Region *del = region;
        
        region = region->next;
        
        delete del;
      }
    }
#ifdef SINGLE_ALLOC
    cudaFree(HdevOffset);
#endif
  }
};

Region* Allocator::regions[MAX_SIZE];

#endif
