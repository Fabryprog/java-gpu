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

package org.java.gpu.debug;

import org.java.gpu.graph.BlockVisitor;
import org.java.gpu.graph.Block;
import org.java.gpu.graph.Loop;
import org.java.gpu.graph.Method;

import org.java.gpu.graph.TrivialLoop;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import java.util.HashSet;
import java.util.Set;

import org.apache.log4j.Logger;

/**
 * Outputs a Graphviz format control flow graph.
 */
public class ControlFlowOutput extends BlockVisitor<Void> {
  /**
   * Set of blocks that have been outputed.
   */
  private Set<Block> visited = new HashSet<Block>();

  /**
   * Output stream to write to.
   */
  private PrintStream ps;

  /**
   * Convenience method for outputting graph for a method to a given file.
   *
   * @param method  Method to output.
   * @param file    File for output.
   */
  public static void outputMethod(Method method, File file) {
    if(method.getImplementation() != null) {
      try {
        PrintStream outPS = new PrintStream(file);
        ControlFlowOutput cfo = new ControlFlowOutput(outPS);

        outPS.println("d2toptions =\"-fpgf --figonly --figpreamble=\\tiny\";");
        outPS.println("graph [ranksep=0.12,minlen=0.15,nodesep=0.15];");
        outPS.println("node [shape=circle, margin=0, width=\"0.175\", fontsize=4pt];");

        method.getImplementation().accept(cfo);
        cfo.end();

        Logger.getLogger("diagrams").info("Diagram for " + method + " outputed to " + file + ".");
      } catch(IOException io) {
        Logger.getLogger("diagrams").warn("Could not output diagram to " + file + ".");
      }
    }
  }

  /**
   * Constructs an 'outputter' onto the given print stream.
   */
  public ControlFlowOutput(PrintStream ps) {
    this.ps = ps;
    this.ps.println("digraph g {");
  }

  /**
   * Outputs a block, along with relevant edges.
   *
   * @param b      Block to output.
   * @return       <code>null</code>
   */
  @Override
  public Void visit(Block b) {
    // Never visit a block twice.
    if(visited.contains(b)) return null;
    visited.add(b);

    // Output a node for the block.
    ps.println("\"" + b.getID() + "\" [style=\"ball color=green!50\"];");

    // Output edges from block to its successors.
    for(Block p : b.getPredecessors()) {
      ps.println("\"" + p.getID() + "\" -> \"" + b.getID() + "\";");
    }

    // Output any successors.
    visit(b.getSuccessors());

    return null;
  }

  /**
   * Outputs a loop, along with relevant edges. The loop node is shaded.
   *
   * @param b      Loop to output.
   * @return       <code>null</code>
   */
  @Override
  public Void visit(Loop b) {
    // Never visit a loop twice.
    if(visited.contains(b)) return null;
    visited.add(b);

    // Output a node for the loop, shaded.
    ps.println("\"" + b.getID() + "\" [style=\"ball color=red!50\"];");

    // Output edges from block to its successors.
    for(Block p : b.getPredecessors()) {
      ps.println("\"" + p.getID() + "\" -> \"" + b.getID() + "\";");
    }

    // Output the loop body, connected by a dotted line.
    ps.println("\"" + b.getID() + "\" -> \"" + b.getStart().getID() + "\" [style=\"dotted\"];");
    b.getStart().accept(this);

    // Output any successors.
    visit(b.getSuccessors());

    return null;
  }

  /**
   * Outputs a trivial loop, along with relevant edges. The loop node is shaded.
   *
   * @param b      Loop to output.
   * @return       <code>null</code>
   */
  @Override
  public Void visit(TrivialLoop b) {
    // Never visit a loop twice.
    if(visited.contains(b)) return null;
    visited.add(b);

    // Output a node for the loop, shaded.
    ps.println("\"" + b.getID() + "\" [style=\"ball color=blue!50\"];");

    // Output edges from block to its successors.
    for(Block p : b.getPredecessors()) {
      ps.println("\"" + p.getID() + "\" -> \"" + b.getID() + "\";");
    }

    // Output the loop body, connected by a dotted line.
    ps.println("\"" + b.getID() + "\" -> \"" + b.getStart().getID() + "\" [style=\"dotted\"];");
    b.getStart().accept(this);

    // Output any successors.
    visit(b.getSuccessors());

    return null;
  }

  /**
   * Ends the output.
   */
  public void end() {
    ps.println("}");
  }
}
