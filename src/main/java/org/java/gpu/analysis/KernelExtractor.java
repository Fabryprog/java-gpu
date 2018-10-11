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

package org.java.gpu.analysis;

import org.java.gpu.analysis.dataflow.AliasUsed;
import org.java.gpu.analysis.dataflow.ReachingConstants;
import org.java.gpu.analysis.dataflow.SimpleUsed;
import org.java.gpu.analysis.dataflow.LiveVariable;

import org.java.gpu.analysis.dependency.DependencyCheck;

import org.java.gpu.cuda.CUDAExporter;

import org.java.gpu.graph.BasicBlock;
import org.java.gpu.graph.Block;
import org.java.gpu.graph.ClassNode;
import org.java.gpu.graph.Kernel;
import org.java.gpu.graph.Modifier;
import org.java.gpu.graph.TrivialLoop;

import org.java.gpu.graph.instructions.Call;
import org.java.gpu.graph.instructions.Constant;
import org.java.gpu.graph.instructions.Producer;
import org.java.gpu.graph.instructions.Read;

import org.java.gpu.exceptions.UnsupportedInstruction;

import org.java.gpu.graph.instructions.Write;
import org.java.gpu.graph.state.State;
import org.java.gpu.graph.state.Variable;

import org.java.gpu.util.Tree;
import org.java.gpu.util.Utils;

import java.util.Collections;
import java.util.Deque;
import java.util.EnumSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

/**
 *
 */
public class KernelExtractor {
  /**
   * Attempts to extract all kernels from the given nested loop tree, according
   * to judgements made by the given dependency checker.
   *
   * @param clazz  Class concerned.
   * @param loops  Root level set of loop nestings.
   * @param check  Dependency checker.
   * @return       Number of kernels successfully extracted.
   */
  public static int extract(ClassNode clazz, Set<Tree<TrivialLoop>> loops, DependencyCheck check) {
    Deque<Tree<TrivialLoop>> consider = new LinkedList<Tree<TrivialLoop>>();

    int count = 0;

    consider.addAll(loops);

    while(!consider.isEmpty()) {
      Tree<TrivialLoop> level = consider.remove();

      if(extract(clazz, level, check)) {
        count++;
      } else {
        consider.addAll(level.getChildren());
      }
    }

    return count;
  }

  /**
   * Attempts to extract the given loop level as a kernel, including any nested
   * loops in the kernel if possible (i.e. multiple dimensions).
   * 
   * @param  clazz Class concerned.
   * @param  level Outer loop to first consider.
   * @param  check Dependency checker.
   * @return       <code>true</code> if the extraction succeeded,
   *               <code>false</code> otherwise.
   */
  public static boolean extract(ClassNode clazz, Tree<TrivialLoop> level, DependencyCheck check) {
    Deque<TrivialLoop> loops = new LinkedList<TrivialLoop>();

    // Various lists for creating kernel.
    List<Variable>               indices    = new LinkedList<Variable>();
    List<Producer>               limits     = new LinkedList<Producer>();
    List<Map<Variable, Constant>>constants  = new LinkedList<Map<Variable, Constant>>();
    List<Map<Variable, Integer>> increments = new LinkedList<Map<Variable, Integer>>();
    List<Kernel.Parameter>       parameters = new LinkedList<Kernel.Parameter>();

    // First check that one level is possible.
    if(!check.check(level.getValue())) {
      return false;
    }
    
    // Add first level.
    constants.add(Collections.EMPTY_MAP);
    indices.add(level.getValue().getIndex());
    limits.add(level.getValue().getLimit());
    increments.add(level.getValue().getIncrements());

    loops.add(level.getValue());
    level = Utils.getSingleElement(level.getChildren());

    // Determine the number of dimensions.
    while(level != null) {
      // Check inner loop (includes checking limit doesn't depend on loop).
      if(!check.check(level.getValue())) {
        break;
      }

      // Pre-inner loop region.
      Set<Block> preRegion = BlockCollector.collect(
        loops.peekLast().getStart(),
        level.getValue(),
        true,
        false
      );

      SimpleUsed pre = new SimpleUsed(preRegion);

      // Post-inner loop region.
      SimpleUsed post = new SimpleUsed(
        BlockCollector.collect(level.getValue().getNext())
      );

      // Ensure that pre-inner loop region defines exactly the increment set.
      Map<Variable, Constant> loopConstants = new ReachingConstants(preRegion)
                                            .getResultAtStart(level.getValue());

      if(!loopConstants.keySet().equals(level.getValue().getIncrements().keySet())) {
        break;
      }

      // Neither pre or post region should not have any reference writes.
      if(pre.containsReferenceWrites() || post.containsReferenceWrites()) {
        break;
      }

      // Post region should just have direct writes to increment variables.
      if(!loops.peekLast().getIncrements().keySet().containsAll(post.getDirectWrites())) {
        break;
      }

      // Collect constants, indices, increment and limit lists.
      constants.add(loopConstants);
      indices.add(level.getValue().getIndex());
      limits.add(level.getValue().getLimit());
      increments.add(level.getValue().getIncrements());

      // Get next level.
      loops.add(level.getValue());
      level = Utils.getSingleElement(level.getChildren());
    }

    // Reverse Lists (dimension 0 should be inner most).
    Collections.reverse(constants);
    Collections.reverse(indices);
    Collections.reverse(limits);
    Collections.reverse(increments);
    Collections.reverse((LinkedList) loops);

    // Kernel body.
    Block body = loops.peekFirst().getStart();

    // Line Number String
    String line = (body.getLineNumber() == null)
                                  ? "" : " (line " + body.getLineNumber() + ")";

    // Kernel Parameters: Copy-In Variables
    SimpleUsed used = new SimpleUsed(body);
    Set<State> copyIn = new LiveVariable(body).getLive(body);

    copyIn.addAll(used.getStatics());

    // Determine 'copy-out' state.
    AliasUsed alias = new AliasUsed(body, copyIn);

    // Note: used.getWrites() should be disjoint with LIVE after loop - unless
    //       dependency checks have gone wrong.

    // Kernel Parameters
    for(State s : copyIn) {
      parameters.add(
        new Kernel.Parameter(
          s,
          alias.getBaseWrites().contains(s) || !alias.isAccurate()
        )
      );
    }

    // Create kernel.
    Kernel kernel = new Kernel(
      "kernel_" + (loops.hashCode() > 0 ? loops.hashCode() : "M" + -loops.hashCode()),
      indices,
      increments,
      parameters
    );

    kernel.setImplementation(body);
    kernel.getModifiers().addAll(EnumSet.of(Modifier.STATIC, Modifier.PRIVATE));

    clazz.addMethod(kernel);

    // Try exporting kernel
    kernel.getModifiers().add(Modifier.NATIVE);

    try {
      CUDAExporter.export(kernel);
    } catch(UnsupportedInstruction e) {
      Logger.getLogger("extract").warn(
        "Kernel" + line + " could not be compiled for CUDA (" + e + ")."
      );

      clazz.removeMethod(kernel);
      return false;
    }

    BasicBlock invoke = new BasicBlock();

    // Constant initialisation.
    for(Map<Variable, Constant> map : constants) {
      for(Map.Entry<Variable, Constant> pair : map.entrySet()) {
        invoke.getStateful().add(new Write(pair.getKey(), pair.getValue()));
      }
    }

    // Actual Call
    Producer[] arguments = new Producer[kernel.getParameterCount()];

    int i = 0;

    for(Producer limit : limits) {
      arguments[i++] = limit;
    }
    
    for(Kernel.Parameter p : kernel.getRealParameters()) {
      arguments[i++] = new Read(p.getState());
    }

    invoke.getStateful().add(
      new Call(arguments, kernel, Call.Sort.STATIC)
    );

    // Replace outer loop with invocation.
    loops.peekLast().replace(invoke);

    // User Feedback
    Logger.getLogger("extract").info(
      "Kernel of " + indices.size() + " dimensions extracted" + line + "."
    );

    Logger.getLogger("extract").info("   Copy In: " + copyIn);
    Logger.getLogger("extract").info("   Copy Out: " + alias.getBaseWrites());
    
    return true;
  }
}
