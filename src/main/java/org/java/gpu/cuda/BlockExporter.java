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

import org.java.gpu.graph.BasicBlock;
import org.java.gpu.graph.Block;
import org.java.gpu.graph.BlockVisitor;
import org.java.gpu.graph.Loop;
import org.java.gpu.graph.TrivialLoop;
import org.java.gpu.graph.Type;

import org.java.gpu.graph.instructions.Instruction;
import org.java.gpu.graph.instructions.Producer;
import org.java.gpu.graph.instructions.RestoreStack;

import org.java.gpu.graph.state.Variable;

import java.io.PrintStream;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 *
 */
class BlockExporter extends BlockVisitor<Void> {
  /**
   * Mapping from graph blocks to labels.
   */
  static private Map<Block, String> labels = new HashMap<Block, String>();

  /**
   * Blocks that have already been exported.
   */
  private Set<Block> visited = new HashSet<Block>();

  /**
   * Print stream for block export
   */
  private PrintStream ps;

  /**
   * Default destination (in case of null jumps).
   */
  private Block defaultDestination = null;

  /**
   * Constructs an exporter for the given export destination.
   *
   * @param ps     Printstream to which the method should be exported.
   */
  BlockExporter(PrintStream ps) {
    this.ps = ps;
  }

  /**
   * Constructs an exporter for the given export destination, along with a
   * default destination for null jumps. This is useful for loop exports where
   * the end of the loop
   *
   * @param ps     Printstream to which the method should be exported.
   */
  BlockExporter(PrintStream ps, Block dflt) {
    this.ps = ps;
    this.defaultDestination = dflt;
  }

  /**
   * Returns a label for a block, creating a new one if one hasn't yet been
   * allocated.
   *
   * @param b      Block to get label for.
   * @return       C label.
   */
  static public String getLabel(Block b) {
    if(labels.containsKey(b)) return labels.get(b);

    // Create Label
    String l = new String("l" + (labels.size() - (labels.containsKey(null) ? 1 : 0)));
    labels.put(b, l);

    return l;
  }

  /**
   * Exports the given basic block to the export destination. Where possible a
   * cached flattening of the block code is used. Otherwise a depth first
   * traversal of the code is used (blindly!).
   *
   * @param bb     Basic block to export.
   * @return       <code>null</code>
   */
  @Override
  public Void visit(BasicBlock bb) {
    // Ensure we haven't already exported this block.
    if(visited.contains(bb)) return null;
    visited.add(bb);

    // Label name
    ps.println(getLabel(bb) + ": { // " + bb);

    if(defaultDestination != null) labels.put(null, getLabel(defaultDestination));

    // Generate code by traversal.
    CppGenerator cg = new CppGenerator(ps);

    // First stack restorations.
    int index = 0;
    
    for(Type t : bb.getTypesIn()) {
      (new RestoreStack(index++, t)).accept(cg);
    }

    // Stateful timeline.
    for(Instruction i : bb.getStateful()) {
      i.accept(cg);
    }

    // Stack saving.
    index = 0;

    for(Producer value : bb.getValuesOut()) {
      ps.println("s" + Helper.getName(new Variable(index++, value.getType())) + " = " + value.accept(cg) + ";");
    }

    // Branch.
    if(bb.getBranch() != null) {
      bb.getBranch().accept(cg);
    }

    ps.println("}");

    labels.remove(null);

    // Generate jump to next label.
    if(bb.getNext() != null) {
      if(visited.contains(bb.getNext())) {
        ps.println("goto " + getLabel(bb.getNext()) + ";");
      } else {
        bb.getNext().accept(this);
      }
    } else if(defaultDestination != null) {
      ps.println("goto " + getLabel(defaultDestination) + ";");
    } else {
      // TODO: Assert branch is RETURN, SWITCH, THROW, ...
    }

    // Ensure successors are exported.
    visit(bb.getSuccessors());

    return null;
  }

  /**
   * Exports the given loop to the export destination. This simply exports the
   * loop body, looping its end back to the beginning of the loop.
   *
   * @param l      Loop to export
   * @return       <code>null</code>
   */
  @Override
  public Void visit(Loop l) {
    // Ensure we haven't already exported this block.
    if(visited.contains(l)) return null;
    visited.add(l);

    // Label name
    ps.println(getLabel(l) + ": // " + l);

    // Export loop body, looping back to here at the end (default destination).
    BlockExporter ke = new BlockExporter(ps, l);
    ke.visited.addAll(l.getSuccessors());
    l.getStart().accept(ke);

    // Ensure successors are exported.
    visit(l.getSuccessors());

    return null;
  }

  /**
   * Exports the given trivial loop to the export destination. This simply
   * exports the loop body within a <code>while</code> loop for the required
   * condition.
   *
   * @param l      Trivial loop to export
   * @return       <code>null</code>
   */
  @Override
  public Void visit(TrivialLoop l) {
    // Ensure we haven't already exported this block.
    if(visited.contains(l)) return null;
    visited.add(l);

    // Label name
    ps.println(getLabel(l) + ": { // " + l);

    // Calculate loop bound.
    String limit = l.getLimit().accept(new CppGenerator(ps));

    // Produce loop condition.
    if(l.getIncrements().get(l.getIndex()) > 0) {
      ps.println("while(" + Helper.getName(l.getIndex()) + " < " + limit + ") {");
    } else {
      ps.println("while(" + Helper.getName(l.getIndex()) + " > " + limit + ") {");
    }

    // Export loop body.
    BlockExporter ke = new BlockExporter(ps);
    l.getStart().accept(ke);

    ps.println("}");
    ps.println("}");

    // Ensure the next block is exported.
    if(l.getNext() != null) {
      if(visited.contains(l.getNext())) {
        ps.println("goto " + getLabel(l.getNext()) + ";");
      } else {
        l.getNext().accept(this);
      }
    } else if(defaultDestination != null) {
      ps.println("goto " + getLabel(defaultDestination) + ";");
    }

    return null;
  }
}
