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
import org.java.gpu.tools.Restrict;

@Restrict
public class ReverseArray {
  @Parallel(loops = "j")
  public static void main(String[] args) {
    float[][] data = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
    float[][] new_data = new float[3][];

    for(int i = 0; i < 3; i++) {
      System.out.println(data[i][0] + " " + data[i][1] + " " + data[i][2]);
    }

    for(int j = 0; j < 3; j++) {
      new_data[j] = data[2 - j];
    }

    for(int i = 0; i < 3; i++) {
      System.out.println(new_data[i][0] + " " + new_data[i][1] + " " + new_data[i][2]);
    }
  }
}
