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

import org.java.gpu.util.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * Attempts to make outputted curly-brace style code slightly easier to read by
 * inserting some indentation and spacing between methods.
 */
public class Beautifier extends PrintStream {
  /**
   * Number of spaces per `tab' or indent.
   */
  public static final int TAB = 2;

  /**
   * Current indentation (measured in tabs).
   */
  private int indent = 0;

  /**
   * Wraps a file so that code written to it is correctly indented.
   *
   * @param out    Output stream to wrap.
   */
  public Beautifier(File out) throws FileNotFoundException {
    super(out);
  }

  /**
   * Wraps an existing output stream so that code written to it is correctly
   * indented.
   *
   * @param out    Output stream to wrap.
   */
  public Beautifier(OutputStream out) {
    super(out);
  }

  /**
   * When outputting a string, this first alters the indentation accordingly
   * before acting as a standard <code>PrintStream</code>.
   *
   * @param string String of code.
   */
  @Override
  public void print(String string) {
    indent += Utils.count("{", string);

    if(indent > 0) {
      indent -= Utils.count("}", string);
    }

    super.print(string);
  }

  /**
   * Acts identically to <code>print(string); println();</code>.
   *
   * @param string String of code.
   */
  @Override
  public void println(String string) {
    print(string);
    println();
  }

  /**
   * For each new line this inserts the current indent, and spacing if at the
   * unindented `root' level.
   */
  @Override
  public void println() {
    super.println();

    // Gaps between methods/structs etc.
    if(indent == 0) {
      super.println();
    } else {
      for(int i = 0; i < indent * TAB; i++) {
        super.print(" ");
      }
    }
  }
}
