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

#ifndef PARALLEL_TRANSFER
#define PARALLEL_TRANSFER

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <jni.h>
#include <map>

/**
 *    Filename: transfer.h
 * Description: Provides functions for copying data to and from a CUDA device
 *              from JNI.
 */

/**
 * Declaration of import/export template with implementations for primitive
 * types (i.e. straight copy).
 */
template <typename T> class Transfer {
public:
  static void fromJava(JNIEnv *env, T obj, T& cuda, bool copy) {
    cuda = obj;
  }
  
  static T toJava(JNIEnv *env, const T& cuda, bool copy) {
    return cuda;
  }
};

/**
 * Cached object, implements visitor pattern so that we can iterate through the
 * cache, exporting objects.
 */
class Cached {
public:
  Cached *next;
  jobject object;
  
  virtual void toJava(JNIEnv *env) { }
  virtual void* get() { }
  
  void insert(Cached *&list) {
    this->next   = list;
    list         = this;
  }
  
  // WARNING: Only apply this where T is known to be of the same type as the
  //          cached value.
  template <typename T> void set(T& lhs) {
    memcpy(&lhs, get(), sizeof(T));
  }
};

template <typename T>
class CachedImpl : public Cached {
  T    cached;
  bool copy;
  
public:
  CachedImpl(T& value, bool copy) {
    this->cached = value;
    this->copy   = copy;
    this->object = value.object;
  }
  
  virtual void toJava(JNIEnv *env) {
    Transfer<T>::toJava(env, cached, copy);
  }
  
  virtual void* get() {
    return &cached;
  }
};

static Cached *cache[10007];

Cached *find(jobject object) {
  int bin = (size_t) object % 10007;
  
  Cached *current = cache[bin];
  
  while(current != NULL) {
    if(current->object == object) {
      return current;
    }
    
    current = current->next;
  }
  
  return NULL;
}

void insert(Cached *cached) {
  int bin = (size_t) cached->object % 10007;
  
  cached->insert(cache[bin]);
}

void exportAll(JNIEnv *env) {
  for(int i = 0; i < 10007; i++) {
    Cached *current = cache[i], *previous;
    
    while(current != NULL) {
      current->toJava(env);
      
      previous = current;
      
      current = current->next;
      
      delete previous;
    }
    
    cache[i] = NULL;
  }
}

/**
 * Modified import/export behaviour for doubles.
 */
template <> class Transfer<jdouble> {
public:
  static void fromJava(JNIEnv *env, jdouble obj, jdouble& cuda, bool copy) {
    cudaSetDoubleForDevice(&obj);
    
    cuda = obj;
  }
  
  static double toJava(JNIEnv *env, jdouble& cuda, bool copy) {
    cudaSetDoubleForHost(&cuda);
    
    return cuda;
  }
};

/**
 * Import/export implementations for objects.
 */
template <typename T> class Transfer<Object<T> > {
public:
  static void fromJava(JNIEnv *env, jobject obj, Object<T>& cuda, bool copy) {
    cuda.object = obj;
    
    Cached *cached = find(obj);
    
    // Object already imported.
    if(cached != NULL) {
      cached->set(cuda);
    // Fresh object.
    } else if(sizeof(T) > 0) {
      Allocator::allocateDual(cuda, copy);
      cached = new CachedImpl<Object<T> >(cuda, copy);
      insert(cached);
      
      Transfer<T>::fromJava(env, obj, *cuda.host, copy);
    }
  }
  
  static void toJava(JNIEnv *env, const Object<T>& cuda, bool copy) {
    if((sizeof(T) > 0) && copy) {
      Transfer<T>::toJava(env, cuda.object, *cuda.host);
    }
  }
};

/**
 * Import/export implementations for array types.
 */
template <typename T, bool primitive> class ArrayImp {};

template <typename T> class ArrayImp<T, true> {
public:
  static void fromJava(JNIEnv *env, jobject arr, Array<T>& cuda, bool copy) {
    cuda.object = (jarray) arr;
    
    Cached *cached = find(arr);
    
    // Object already imported.
    if(cached != NULL) {
      cached->set(cuda);
    // Fresh array.
    } else {
      cuda.length = env->GetArrayLength(cuda.object);
      cuda.host   = (T *) env->GetPrimitiveArrayCritical(cuda.object, NULL);
      
      Allocator::allocateDevice(cuda, copy);
      cached = new CachedImpl<Array<T> >(cuda, copy);
      insert(cached);
    }
  }
  
  static void toJava(JNIEnv *env, const Array<T>& cuda, bool copy) {
    env->ReleasePrimitiveArrayCritical(
      cuda.object,
      cuda.host,
      copy ? 0 : JNI_ABORT
    );
  }
};

template <typename T> class ArrayImp<T, false> {
public:
  static void fromJava(JNIEnv *env, jobject arr, Array<T>& cuda, bool copy) {
    cuda.object = (jarray) arr;
    
    Cached *cached = find(arr);
    
    // Object already imported.
    if(cached != NULL) {
      cached->set(cuda);
    // Fresh array.
    } else {
      cuda.length = env->GetArrayLength(cuda.object);
      
      Allocator::allocateDual(cuda, copy);
      cached = new CachedImpl<Array<T> >(cuda, copy);
      insert(cached);
    
      for(int i = 0; i < cuda.length; i++) {
        Transfer<T>::fromJava(
          env,
          env->GetObjectArrayElement((jobjectArray) cuda.object, i),
          cuda.host[i],
          copy
        );
      }
    }
  }
  
  static void toJava(JNIEnv *env, const Array<T>& cuda, bool copy) {
    // Update
    if(copy) {
      for(int i = 0; i < cuda.length; i++) {
        env->SetObjectArrayElement(
          (jobjectArray) cuda.object,
          i,
          cuda.host[i].object
        );
      }
    }
  }
};

template <typename T> class Transfer<Array<T> > {
public:
  static void fromJava(JNIEnv *env, jobject arr, Array<T>& cuda, bool copy) {
    ArrayImp<T, is_primitive<T>::value>::fromJava(env, arr, cuda, copy);
  }
  
  static void toJava(JNIEnv *env, const Array<T>& cuda, bool copy) {
    ArrayImp<T, is_primitive<T>::value>::toJava(env, cuda, copy);
  }
};

/**
 * Modified routines for dealing with doubles (since some CUDA devices do not
 * support these natively).
 */
template <> class Transfer<Array<jdouble> > {
public:
  static void fromJava(JNIEnv *env, jobject arr, Array<jdouble>& cuda, bool copy) {
    cuda.object = (jarray) arr;
    
    Cached *cached = find(arr);
    
    // Object already imported.
    if(cached != NULL) {
      cached->set(cuda);
    // Fresh array.
    } else {
      cuda.length = env->GetArrayLength(cuda.object);
      cuda.host   = (jdouble *) env->GetPrimitiveArrayCritical(cuda.object, NULL);
      
      Allocator::allocateDevice(cuda, copy);
      cached = new CachedImpl<Array<jdouble> >(cuda, copy);
      insert(cached);
    
      // Convert Doubles
      if(convertDoubles) {
        for(int i = 0; i < cuda.length; i++) {
          cudaSetDoubleForDevice(&cuda.host[i]);
        }
      }
    }
  }
  
  static void toJava(JNIEnv *env, const Array<jdouble>& cuda, bool copy) {
    if(copy && convertDoubles) {
      // Convert Doubles
      for(int i = 0; i < cuda.length; i++) {
        cudaSetDoubleForHost(&cuda.host[i]);
      }
    }
    
    env->ReleasePrimitiveArrayCritical(
      cuda.object,
      cuda.host,
      copy ? 0 : JNI_ABORT
    );
  }
};

#endif

