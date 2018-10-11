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

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import org.java.gpu.tools.Parallel;
import org.java.gpu.tools.Restrict;

/**
 * Mandelbrot Benchmark. Adapted from the Computer Language Benchmarks Game
 * (http://shootout.alioth.debian.org/) so that all cells are calculated before
 * outputing the complete image (since sequentially outputing pixels restricts
 * parallelisation).
 *
 * Note that <code>float</code>s are used since not all GPUs support double
 * precision, and also that exceptions are just thrown out of <code>main</code>
 * to avoid <code>try { ... } catch { ... }</code> blocks.
 *
 * @author Stefan Krause
 * @author Peter Calvert
 */
@Restrict
public class Mandelbrot {
  final static float LIMIT = 4.0f;

  private short data[][];
  private int   width, height;
  private int   iterations;
  private float spacing;

  public Mandelbrot(int w, int h) {
    this.width   = w;
    this.height  = h;
    this.spacing = 2.0f / this.width;
    this.iterations = 250; // 50: Shootout, 250: Leung '09

    data = new short[this.height][this.width];
  }

  public Mandelbrot(int i) {
    this.width   = 8000;
    this.height  = 8000;
    this.spacing = 2.0f / this.width;
    this.iterations = i;

    data = new short[this.height][this.width];
  }

  @Parallel(loops = {"y", "x"})
  public void compute() {
    for(int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++) {
        float Zr = 0.0f;
        float Zi = 0.0f;
        float Cr = (x * spacing - 1.5f);
        float Ci = (y * spacing - 1.0f);

        float ZrN = 0;
        float ZiN = 0;
        int i;

        for(i = 0; (i < iterations) && (ZiN + ZrN <= LIMIT); i++) {
          Zi = 2.0f * Zr * Zi + Ci;
          Zr = ZrN - ZiN + Cr;
          ZiN = Zi * Zi;
          ZrN = Zr * Zr;
        }

        data[y][x] = (short)((i * 255) / iterations);
      }
    }
  }

  public void output(File out) throws IOException {
    BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
    WritableRaster r = img.getRaster();

    for(int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++) {
        r.setSample(x, y, 0, data[y][x]);
      }
    }

    ImageIO.write(img, "png", out);
  }

  public static void main(String[] args) throws Exception {
    Mandelbrot set   = new Mandelbrot(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
    long       start = System.currentTimeMillis();

    set.compute();

    System.out.println(
      set.width + "x" + set.height + " points considered in " +
      (System.currentTimeMillis() - start) + "ms"
    );

    if(args.length > 2) {
      set.output(new File(args[2]));
    }
  }

  public static long run(int size) {
    Mandelbrot set   = new Mandelbrot(size, size);
    long       start = System.currentTimeMillis();

    set.compute();

    return System.currentTimeMillis() - start;
  }
}
