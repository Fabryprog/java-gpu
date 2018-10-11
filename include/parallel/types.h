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

#ifndef PARALLEL_TYPES
#define PARALLEL_TYPES

#include <jni.h>

/**
 *    Filename: types.h
 * Description: Defines types for use in GPU execution, as well as 
 */

/**
 * All Object Types.
 */
template <typename T> struct Object {
  jobject  object;
  T       *host;
  T       *device;
};

/**
 * Array Types
 */
template <typename T> struct Array {
  jarray   object;
  T       *host;
  T       *device;
  jsize    length;
};

/**
 * Determines whether a type is primitive or not.
 */
template <typename T> class is_primitive { public: enum { value = 0 }; };
template <> class is_primitive<jint>     { public: enum { value = 1 }; };
template <> class is_primitive<jboolean> { public: enum { value = 1 }; };
template <> class is_primitive<jbyte>    { public: enum { value = 1 }; };
template <> class is_primitive<jchar>    { public: enum { value = 1 }; };
template <> class is_primitive<jshort>   { public: enum { value = 1 }; };
template <> class is_primitive<jlong>    { public: enum { value = 1 }; };
template <> class is_primitive<jfloat>   { public: enum { value = 1 }; };
template <> class is_primitive<jdouble>  { public: enum { value = 1 }; };

#endif
