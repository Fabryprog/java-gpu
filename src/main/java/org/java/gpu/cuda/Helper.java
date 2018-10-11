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

package org.java.gpu.cuda;

import org.java.gpu.graph.ClassNode;
import org.java.gpu.graph.Kernel;
import org.java.gpu.graph.Kernel.Parameter;
import org.java.gpu.graph.Method;
import org.java.gpu.graph.Modifier;
import org.java.gpu.graph.Type;
import org.java.gpu.graph.state.Field;
import org.java.gpu.graph.state.State;
import org.java.gpu.graph.state.Variable;

import org.java.gpu.util.TransformIterable;
import org.java.gpu.util.Utils;

import java.io.PrintStream;

import java.util.Map;

import org.apache.log4j.Logger;

/**
 * Helper methods for exporting methods to C code (of the CUDA/JNI variety).
 */
class Helper {
  /**
   * Transforms a name according to the JNI specification for managling method
   * names (see Table 2-1 of the JNI specification). However, this is used for
   * more than just method names.
   *
   * @param name   Name.
   * @return       Mangled name.
   */
  public static String jniMangle(String name) {
    return name.replace("_", "_1").replace("$", "_00024");
  }

  /**
   * Returns a function name for the given method. This follows the JNI scheme
   * but excludes the <code>Java_</code> prefix (since it is also used for
   * CUDA <code>__device__</code> functions).
   *
   * @param method Method.
   * @return       C function name for the method.
   */
  public static String getName(Method method) {
    return jniMangle(method.getOwner().getName()).replace('/', '_') + "_"
         + jniMangle(method.getName());
  }

  /**
   * Returns a variable name for the given <code>State</code> object.
   *
   * @param  state State.
   * @return       C variable name.
   */
  public static String getName(State state) {
    // Variables
    if(state instanceof Variable) {
      Variable v = (Variable) state;

      if(v.getType().getSort() == Type.Sort.REF) {
        if(v.getType().hashCode() < 0) {
          return "v" + v.getIndex() + "_M" + (-v.getType().hashCode());
        } else {
          return "v" + v.getIndex() + "_" + v.getType().hashCode();
        }
      } else {
        return "v" + v.getIndex() + "_" + v.getType().getSort();
      }
    // Static Fields
    } else if(state instanceof Field) {
      Field f = (Field) state;

      return "Static_" + jniMangle(f.getOwner().getName()).replace('/', '_')
           + "_" + jniMangle(f.getName());
    }

    throw new RuntimeException(
      "States that aren't variables or statics must be handled case-by-case."
    );
  }

  /**
   * Returns the JNI type for a given type representation. This handles
   * primitive types and reference types.
   *
   * @param type   Type.
   * @return       JNI C type.
   */
  public static String getType(Type type) {
    // Void
    if(type.getSort() == null) {
      return "void";
    // Primitive Types
    } else if(type.getSort() != Type.Sort.REF) {
      switch(type.getSort()) {
        case BOOL:   return "jboolean";
        case BYTE:   return "jbyte";
        case CHAR:   return "jchar";
        case SHORT:  return "jshort";
        case INT:    return "jint";
        case FLOAT:  return "jfloat";
        case LONG:   return "jlong";
        case DOUBLE: return "jdouble";
        default:     throw new UnsupportedOperationException("ADDRESS TYPE");
      }
    // Array Types
    } else if(type.getElementType() != null) {
      return "Array<" + getType(type.getElementType()) + " >";
    // Object Types
    } else {
      return "Object<Data_" + jniMangle(type.getInternalName()).replace('/', '_') +">";
    }
  }

  /**
   * Returns the name of the sort of a given type. This name corresponds to the
   * infix &lt;Type%gt; parameter in many JNI function names.
   *
   * @param  type  Type whose sort should be considered.
   * @return       Name of sort.
   */
  public static String getSort(Type type) {
    // Primitive Types
    if(type.getSort() != Type.Sort.REF) {
      switch(type.getSort()) {
        case BOOL:   return "Boolean";
        case BYTE:   return "Byte";
        case CHAR:   return "Char";
        case SHORT:  return "Short";
        case INT:    return "Int";
        case FLOAT:  return "Float";
        case LONG:   return "Long";
        case DOUBLE: return "Double";
        default:     throw new UnsupportedOperationException("ADDRESS TYPE");
      }
    // Reference Types
    } else {
      return "Object";
    }
  }

  /**
   * Defines the data structure and transfer methods for a given Java class.
   *
   * @param clazz  Class to export.
   * @param out    Output printstream.
   */
  public static void defineClass(ClassNode clazz, PrintStream out) {
    String type = "Data_" + jniMangle(clazz.getName()).replace('/', '_');

    // Structure
    out.println("struct " + type + " {");
    
    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        out.println(getType(f.getType()) + " " + f.getName() + ";");
      }
    }

    out.println("};");
    
    // Transfer Functions.
    out.println("template <> class Transfer<" + type + "> {");
    out.println("static bool _init;");

    // JNI field ID variables.
    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        out.println("static jfieldID " + jniMangle(f.getName()) + ";");
      }
    }

    // Field ID initialiser.
    out.println("static void init(JNIEnv *env) {");
    out.println("jclass clazz = env->FindClass(\"" + clazz.getName() + "\");");

    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        out.print(jniMangle(f.getName()) + " = env->GetFieldID(clazz, \"");
        out.println(f.getName() + "\", \"" + f.getType().getDescriptor() + "\");");
      }
    }

    out.println("_init = true;");
    out.println("}");

    // Import function.
    out.println("public:");
    out.println("static void fromJava(JNIEnv *env, jobject obj, " + type + "& s, bool copy) {");
    out.println("if(!_init) init(env);");

    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        out.print("Transfer<" + getType(f.getType()) + " >::fromJava(env, ");
        out.print("env->Get" + getSort(f.getType()) + "Field(obj, ");
        out.println(jniMangle(f.getName()) + "), s." + f.getName() + ", copy);");
      }
    }

    out.println("}");

    // Export function.
    out.println("static void toJava(JNIEnv *env, jobject obj, " + type + "& s) {");
    out.println("if(!_init) init(env);");

    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        // Reference Types
        if(f.getType().getSort() == Type.Sort.REF) {
          out.print("env->Set" + getSort(f.getType()) + "Field(obj, ");
          out.println(jniMangle(f.getName()) + ", s." + f.getName() + ".object);");
        // Primitive Types
        } else {
          out.print("env->Set" + getSort(f.getType()) + "Field(obj, ");
          out.print(jniMangle(f.getName()) + ", Transfer<" + getType(f.getType()));
          out.println(" >::toJava(env, s." + f.getName() + ", true));");
        }
      }
    }

    out.println("}");
    out.println("};");

    out.println("bool Transfer<" + type +">::_init = false;");
    
    for(Field f : clazz.getFields()) {
      if(!f.getModifiers().contains(Modifier.STATIC)) {
        out.println("jfieldID Transfer<" + type +">::" + jniMangle(f.getName()) + " = NULL;");
      }
    }
  }

  /**
   * Outputs the code for a given state to be imported, with or without
   * specifying copy out when the corresponding export occurs.
   *
   * @param state  State to import.
   * @param copy   <code>true</code> if the state should be copied back to Java,
   *               <code>false</code> otherwise.
   * @param out    Output printstream.
   */
  public static void importState(State state, boolean copy, PrintStream out) {
    out.println(getType(state.getType()) + " " + getName(state) + ";");
    out.print("Transfer<" + getType(state.getType()) + " >::fromJava(e, ");
    out.println("java_" + getName(state) + ", " + getName(state) + ", " + copy + ");");
  }

  /**
   * Outputs the code for the start of a kernel implementation. This calculates
   * the values of any variables that depend on the loop indices, and also
   * returns if a thread is above the limit.
   *
   * @param kernel Kernel to export.
   * @param out    Output printstream.
   */
  public static void kernelStart(Kernel kernel, PrintStream out) {
    out.print("__global__ void " + kernel.getName() + "(");

    // Dimension limits. Only need to pass these for X and Y, since CUDA will
    // never need to execute 'extra' threads in Z, and does not support higher
    // dimensions.
    for(int d = 0; (d < 2) && (d < kernel.getDimensions()); d++) {
      if(d != 0) {
        out.print(", ");
      }

      out.print("int limit" + d);
    }

    // Variable kernel parameters (statics are passed via __constant__ memory).
    for(Parameter p : kernel.getRealParameters()) {
      if(p.getState() instanceof Variable) {
        out.print(", " + getType(p.getState().getType()) + " " + getName(p.getState()));
      }
    }

    out.println(") {");

    // Calculate dimensions.
    switch(kernel.getDimensions()) {
      default:
      case 3: out.println("jint dim2 = blockIdx.z * blockDim.z;");
      case 2: out.println("jint dim1 = blockIdx.y * blockDim.y + threadIdx.y;");
      case 1: out.println("jint dim0 = blockIdx.x * blockDim.x + threadIdx.x;");
    }

    // TODO: Allow variables that are incremented not just on a single dimension
    //       - or put in code elsewhere that puts it in the increment sets for
    //       each relevant dimension.

    // Perform increments.
    for(int d = 0; (d < 2) && (d < kernel.getDimensions()); d++) {
      for(Map.Entry<Variable, Integer> inc : kernel.getIncrements(d).entrySet()) {
        out.println(getName(inc.getKey()) + " += " + inc.getValue().intValue() + " * dim" + d + ";");
      }
    }

    // Check limits (again only for 3 dimensions).
    for(int d = 0; (d < 2) && (d < kernel.getDimensions()); d++) {
      Variable index = kernel.getIndex(d);

      out.print("if(" + getName(index));

      if(kernel.getIncrements(d).get(index).intValue() > 0) {
        out.print(" >= ");
      } else {
        out.print(" <= ");
      }

      out.println("limit" + d + ") return;");
    }
  }

  public static void kernelEnd(Kernel kernel, PrintStream out) {
    out.println("}");
  }

  public static void launcher(Kernel kernel, PrintStream out) {
    // Required since JNI cannot handle C++ name mangling.
    out.println("extern \"C\" {");

    // Creates prototype for launcher, according to JNI name mangling.
    out.print("void Java_" + getName(kernel) + "(JNIEnv *e, jclass clazz");

    // Limits for each of the requried dimensions.
    for(int d = 0; d < kernel.getDimensions(); d++) {
      out.print(", jint limit" + d);
    }

    // Kernel Parameters
    for(Parameter p : kernel.getRealParameters()) {
      if(p.getState().getType().getSort() == Type.Sort.REF) {
        out.print(", jobject java_" + getName(p.getState()));
      } else {
        out.print(", " + getType(p.getState().getType()) + " java_" + getName(p.getState()));
      }
    }

    out.println(") {");

    // Add timing if in debug mode.
    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEvent_t eventA, eventB, eventC, eventD, eventE, eventF;");
      out.println("cudaEventCreate(&eventA);");
      out.println("cudaEventCreate(&eventB);");
      out.println("cudaEventCreate(&eventC);");
      out.println("cudaEventCreate(&eventD);");
      out.println("cudaEventCreate(&eventE);");
      out.println("cudaEventCreate(&eventF);");
      out.println("cudaEventRecord(eventA, 0);");
    }

    // Import 'copy in' state.
    for(Parameter p : kernel.getRealParameters()) {
      importState(p.getState(), p.getCopyOut(), out);

      // Statics reside in __constant__ memory.
      if(p.getState() instanceof Field) {
        out.println("cudaMemcpyToSymbol(\"" + getName(p.getState()) + "\", &" + getName(p.getState()) + ", sizeof(" + getName(p.getState()) + "));");
      }
    }

    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEventRecord(eventB, 0);");
    }

    out.println("Allocator::copyToDevice(e);");

    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEventRecord(eventC, 0);");
    }

    // Calculate required iterations
    for(int d = 0; d < kernel.getDimensions(); d++) {
      Variable index = kernel.getIndex(d);
      out.print("int required" + d + " = (limit" + d + " - " + getName(index));
      out.println(") / " + kernel.getIncrements(d).get(index) + ";");
    }
    
    // Represent required for first 3 dimensions in CUDA structure.
    switch(kernel.getDimensions()) {
      case 1: out.println("dim3 required(required0);");                      break;
      case 2: out.println("dim3 required(required0, required1);");           break;
      default:out.println("dim3 required(required0, required1, required2);");break;
    }

    // Calculate how much can be done in a single invocation.
    out.println("dim3 gridSize;");
    out.println("dim3 blockSize;");
    out.print("dim3 inc = calculateDimensions((void *) &" + kernel.getName());
    out.println(", &gridSize, &blockSize, required);");

    // Debug execution size.
    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.print("printf(\"DEBUG: Required %dx%dx%d, Per-Invocation: %dx%dx%d ");
      out.print("(%dx%dx%d)\\n\", required.x, required.y, required.z, inc.x, ");
      out.println("inc.y, inc.z, blockSize.x, blockSize.y, blockSize.z);");
    }

    // Iterate over outer dimensions (i.e. >= 3).
    for(int d = kernel.getDimensions(); d >= 3; d--) {
      out.println("for(int d" + d + "; d < required" + d + "; d++) {");
    }

    // Iterate over inner 3 dimensions (as 'inc' maybe smaller than 'required').
    switch(kernel.getDimensions()) {
      default:
      case 3: out.println("for(int d2 = 0; d2 < required2; d2 += inc.z) {");
      case 2: out.println("for(int d1 = 0; d1 < required1; d1 += inc.y) {");
      case 1: out.println("for(int d0 = 0; d0 < required0; d0 += inc.x) {");
    }

    out.print(kernel.getName() + "<<<gridSize, blockSize>>>(");

    // Dimension limits for kernel checks (the grid will normally invoke extra
    // kernels around the edge).
    for(int d = 0; d < kernel.getDimensions(); d++) {
      if(d != 0) {
        out.print(", ");
      }

      out.print("limit" + d);
    }

    // Kernel parameters.
    for(Parameter p : kernel.getRealParameters()) {
      // Only variables must be passed - statics are in constant memory.
      if(p.getState() instanceof Variable) {
        Variable var = (Variable) p.getState();

        out.print(", " + getName(var));

        for(int d = 0; d < kernel.getDimensions(); d++) {
          if(kernel.getIncrements(d).containsKey(var)) {
            out.print(" + (d" + d + " * " + kernel.getIncrements(d).get(var) + ")");
          }
        }
      }
    }

    out.println(");");

    for(int d = 0; d < kernel.getDimensions(); d++) {
      out.println("}");
    }

    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEventRecord(eventD, 0);");
    }

    // Export 'copy out' state.
    out.println("Allocator::copyToHost();");

    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEventRecord(eventE, 0);");
    }

    out.println("exportAll(e);");
    out.println("Allocator::freeAll();");

    // Final timing, plus output.
    if(Logger.getLogger("cuda").isDebugEnabled()) {
      out.println("cudaEventRecord(eventF, 0);");
      out.println("cudaEventSynchronize(eventF);");
      out.println("float timeImport, timeCopyIn, timeExecute, timeCopyOut, timeExport;");
      out.println("cudaEventElapsedTime(&timeImport, eventA, eventB);");
      out.println("cudaEventElapsedTime(&timeCopyIn, eventB, eventC);");
      out.println("cudaEventElapsedTime(&timeExecute, eventC, eventD);");
      out.println("cudaEventElapsedTime(&timeCopyOut, eventD, eventE);");
      out.println("cudaEventElapsedTime(&timeExport, eventE, eventF);");
      out.println("printf(\"DEBUG: Import %fms, Copy In %fms, Execution %fms, Copy Out %fms, Export %fms\\n\", timeImport, timeCopyIn, timeExecute, timeCopyOut, timeExport);");
      out.println("cudaEventDestroy(eventA);");
      out.println("cudaEventDestroy(eventB);");
      out.println("cudaEventDestroy(eventC);");
      out.println("cudaEventDestroy(eventD);");
      out.println("cudaEventDestroy(eventE);");
      out.println("cudaEventDestroy(eventF);");
    }

    // Output function, and C export block ends.
    out.println("}");
    out.println("}");
  }

  public static void methodStart(Method method, PrintStream out) {
    // Creates prototype for function, according to JNI name mangling.
    out.print("__device__ " + getType(method.getReturnType()) + " ");
    out.print(getName(method) + "(");

    // Arguments.
    out.print(Utils.join(
      new TransformIterable<Variable, String>(method.getParameterVariables()) {
        @Override
        protected String transform(Variable variable) {
          return getType(variable.getType()) + " " + getName(variable);
        }
      },
      ", "
    ));

    out.println(") {");
  }

  public static void methodEnd(Method method, PrintStream out) {
    out.println("}");
  }
}
