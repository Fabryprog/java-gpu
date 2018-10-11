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

package org.java.gpu.tools;

import java.io.File;
import java.io.PrintStream;
import java.lang.reflect.Method;

/**
 * Class for running all benchmarks. Provides a main method that runs the
 * benchmark over the given size range producing a Gnuplot data file.
 */
public class Benchmark {
  public static void main(String[] args) {
    // Usage information.
    if(args.length != 4) {
      System.err.println("Usage: java tools.Benchmark class step limit output");
    // Run benchmark
    } else {
      try {
        final int  step  = Integer.parseInt(args[1]);
        final long limit = Long.parseLong(args[2]);

        final Class       clazz = Class.forName(args[0]);
        final Method      run   = clazz.getMethod("run", Integer.TYPE);
        final PrintStream out   = new PrintStream(new File(args[3]));

        out.println("# Parallelising JVM Compiler");
        out.println("#");
        out.println("# Copyright 2010 Peter Calvert, University of Cambridge");
        out.println("#");
        out.println("# Benchmarking Results for " + clazz.getCanonicalName());
        out.println("# Limit: " + limit + "ms (Step: " + step + ")");
        out.println("#");
        out.println("# Columns: Size, Time (ms)");

        long time = 0;

        // Throw away result for startup times.
        run.invoke(null, new Integer(step));

        // Actual benchmarks.
        for(int size = step; time < limit; size += step) {
          time = (Long) run.invoke(null, new Integer(size));

          out.println(size + "\t" + time);
        }
      } catch(Exception e) {
        e.printStackTrace();
      }
    }
  }
}
