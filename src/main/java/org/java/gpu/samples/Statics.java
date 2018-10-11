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

package org.java.gpu.samples;

import org.java.gpu.tools.Parallel;

/**
 * Tests support for static fields and allows the performance of these to be
 * compared to other cases.
 */
public class Statics {
  static double[] nums;
  
  private static double compute(double x) {
    return (Math.sin(x) * Math.sin(x)) + (Math.cos(x) * Math.cos(x));
  }

  @Parallel(loops = "i")
  public static long run(int size) {
    nums = new double[size];

    // Parallelisable, but CUDA can't compile due to 'random'
    for(int i = 0; i < nums.length; i++) {
      nums[i] = Math.random() * 4;
    }

    long time = System.currentTimeMillis();

    // KERNEL-------------------------------
    for(int i = 0; i < nums.length; i++) {
      nums[i] = compute(nums[i]);
    }
    // -------------------------------------

    time = System.currentTimeMillis() - time;

    int errors = 0;

    for(int j = 0; j < nums.length; j++) {
      if(Math.abs(nums[j] - 1.0) > 0.005) {
        errors++;
      }
    }

    if(errors > 0) {
      throw new RuntimeException("Failed (size = " + size + "): " + errors + " errors");
    }

    return time;
  }
}
