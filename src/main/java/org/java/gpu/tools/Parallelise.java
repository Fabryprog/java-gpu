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

import org.java.gpu.analysis.KernelExtractor;

import org.java.gpu.analysis.dependency.AnnotationCheck;
import org.java.gpu.analysis.dependency.BasicCheck;
import org.java.gpu.analysis.dependency.CombinedCheck;
import org.java.gpu.analysis.dependency.DependencyCheck;

import org.java.gpu.analysis.loops.LoopDetector;
import org.java.gpu.analysis.loops.LoopNester;
import org.java.gpu.analysis.loops.LoopTrivialiser;

import org.java.gpu.bytecode.ClassExporter;
import org.java.gpu.bytecode.ClassFinder;
import org.java.gpu.bytecode.ClassImporter;

import org.java.gpu.cuda.CUDAExporter;

import org.java.gpu.debug.ControlFlowOutput;

import org.java.gpu.debug.GraphOutput;
import org.java.gpu.graph.ClassNode;
import org.java.gpu.graph.Loop;
import org.java.gpu.graph.Method;
import org.java.gpu.graph.Modifier;
import org.java.gpu.graph.TrivialLoop;

import org.java.gpu.util.Tree;

import java.io.File;
import java.io.IOException;

import java.net.URL;
import java.util.HashSet;
import java.util.Set;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;
import joptsimple.ValueConversionException;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;

/**
 *
 */
public class Parallelise {
  public static void main(String[] arguments) {
    // Determine default location of parallel includes.
    URL includeURL = Parallelise.class.getResource("/include");
    String defaultIncludes;

    if(includeURL == null) {
      defaultIncludes = "";
    } else {
      defaultIncludes = includeURL.getFile();
    }

    // Setup command line options.
    OptionParser       parser    = new OptionParser();
    OptionSpec<File>   cuda      = parser.accepts("cuda")
                                         .withRequiredArg()
                                         .ofType(File.class)
                                         .defaultsTo(new File(
                                           getEnvironment("CUDA_HOME","")
                                         ));
    OptionSpec<File>   jdk       = parser.accepts("jdk")
                                         .withRequiredArg()
                                         .ofType(File.class)
                                         .defaultsTo(new File(
                                           getEnvironment("JDK_HOME", "")
                                         ));
    OptionSpec<File>   includes  = parser.accepts("includes")
                                         .withRequiredArg()
                                         .ofType(File.class)
                                         .defaultsTo(new File(defaultIncludes));
    OptionSpec<String> library   = parser.accepts("library")
                                         .withRequiredArg()
                                         .ofType(String.class)
                                         .defaultsTo("parallel");
    OptionSpec<File>   classpath = parser.accepts("classpath")
                                         .withRequiredArg()
                                         .ofType(File.class)
                                         .withValuesSeparatedBy(File.pathSeparatorChar);
    OptionSpec<File>   output    = parser.accepts("output")
                                         .withRequiredArg()
                                         .ofType(File.class)
                                         .defaultsTo(new File("."));
    OptionSpec<String> logLevel  = parser.accepts("log")
                                         .withOptionalArg();
    OptionSpec<String> detect    = parser.accepts("detect")
                                         .withRequiredArg()
                                         .ofType(String.class)
                                         .defaultsTo("manual");
    parser.accepts("diagrams");
    parser.accepts("fulldiagrams");
    parser.accepts("generate");
    parser.accepts("nonportable");

    // Parse Options
    OptionSet options;

    try {
      options = parser.parse(arguments);
    } catch(ValueConversionException e) {
      printUsage();
      return;
    } catch(OptionException e) {
      printUsage();
      return;
    }

    if(options.nonOptionArguments().size() == 0) {
      printUsage();
      return;
    }

    // Setup Logging
    Logger rootLogger = Logger.getRootLogger();

    rootLogger.addAppender(new ConsoleAppender(
      new PatternLayout("%-5p [%c]: %m%n")
    ));

    rootLogger.setLevel(Level.WARN);

    if(options.hasArgument(logLevel)) {
      rootLogger.setLevel(Level.toLevel(options.valueOf(logLevel)));
    } else if(options.has(logLevel)) {
      rootLogger.setLevel(Level.DEBUG);
    }

    // Update ClassFinder with input class path.
    if(options.has(classpath)) {
      ClassFinder.setClassPath(options.valuesOf(classpath));
    }

    // Update ClassImporter as to whether to include JDK classes (not portable).
    ClassImporter.setPortable(!options.has("nonportable"));

    // Update ClassExporter with output directory.
    ClassExporter.setOutputDirectory(options.valueOf(output));

    // Update CUDAExporter with given options.
    CUDAExporter.setSystem(
      options.valueOf(cuda),
      options.valueOf(jdk),
      options.valueOf(includes)
    );

    try {
      CUDAExporter.setDestination(
        options.valueOf(output),
        options.valueOf(library),
        !options.has("generate")
      );
    } catch (IOException ex) {
      ex.printStackTrace();
    }

    // Form set of classes to transform.
    Set<String> classes = new HashSet<String>();

    for(String input : options.nonOptionArguments()) {
      classes.addAll(ClassFinder.listClasses(input.replace(".", "/")));
    }

    // Select dependency checker.
    DependencyCheck check;

    if(options.valueOf(detect).equals("auto")) {
      check = new BasicCheck();
    } else if(options.valueOf(detect).equals("both")) {
      check = new CombinedCheck(new AnnotationCheck(), new BasicCheck());
    } else {
      check = new AnnotationCheck();
    }

    // Apply Transformations
    for(String className : classes) {
      ClassNode clazz = ClassImporter.getClass(className);
      boolean altered = false;

      for(Method m : clazz.getMethods()) {
        if((m.getImplementation() != null) &&
                               !m.getModifiers().contains(Modifier.INHERITED)) {
          Logger.getLogger("core").info("Considering " + m);
         
          // Output pre-diagrams (full).
          if(options.has("fulldiagrams")) {
            GraphOutput.outputMethod(m, new File(m.getOwner().getName().replace("/", "-") + "-" + m.getName() + ".pre.dot"));
          }

          // Output pre-diagrams.
          if(options.has("diagrams")) {
            ControlFlowOutput.outputMethod(m, new File(m.getOwner().getName().replace("/", "-") + "-" + m.getName() + ".1.dot"));
          }

          // Detect Loops.
          Set<Loop> loops = LoopDetector.detect(m.getImplementation());

          // Output mid-diagrams.
          if(options.has("diagrams")) {
            ControlFlowOutput.outputMethod(m, new File(m.getOwner().getName().replace("/", "-") + "-" + m.getName() + ".2.dot"));
          }

          // Trivialise Loops.
          Set<TrivialLoop> tloops = LoopTrivialiser.convert(loops);

          // Output post-diagrams.
          if(options.has("diagrams")) {
            ControlFlowOutput.outputMethod(m, new File(m.getOwner().getName().replace("/", "-") + "-" + m.getName() + ".3.dot"));
          }

          // Calculate Nesting.
          Set<Tree<TrivialLoop>> nesting = LoopNester.nest(tloops);

          // Set context (MUST be after loop conversions).
          check.setContext(m);

          // Extract Kernels
          if(KernelExtractor.extract(clazz, nesting, check) > 0) {
            altered = true;
          }

          // Output post-diagrams (full).
          if(options.has("fulldiagrams")) {
            GraphOutput.outputMethod(m, new File(m.getOwner().getName().replace("/", "-") + "-" + m.getName() + ".post.dot"));
          }
        }
      }

      if(altered) {
        CUDAExporter.addLoad(clazz);
        ClassExporter.export(className);
      }
    }

    // Compile CUDA Code
    try {
      CUDAExporter.compile();
    } catch(IOException ex) {
      ex.printStackTrace();
    }
  }

  public static void printUsage() {
    System.out.println();
    System.out.println("Parallelising JVM Compiler");
    System.out.println("Part II Project, Computer Science Tripos");
    System.out.println();
    System.out.println("Copyright (c) 2009, 2010 - Peter Calvert, University of Cambridge");
    System.out.println();
    System.out.println("Usage: java org.java.gpu.tools.Parallelise [options] input");
    System.out.println("       java -jar Parallel.jar [options] input");
    System.out.println();
    System.out.println("Options:");
    System.out.println(" --generate           Only generate CUDA code, do not compile.");
    System.out.println(" --log [LEVEL=DEBUG]  Set logging level (default = WARN).");
    System.out.println(" --cuda DIR           CUDA Home Directory (default: $CUDA_HOME).");
    System.out.println(" --jdk DIR            JDK Home Directory (default: $JDK_HOME).");
    System.out.println(" --includes DIR       Parallel includes (default: internal).");
    System.out.println(" --library NAME       Name of the native library to output.");
    System.out.println(" --classpath ...      Class path for input (default: java.class.path).");
    System.out.println(" --output DIR         Directory for placing output.");
    System.out.println(" --detect MODE        Detection mode (default: manual, other options: auto both.");
    System.out.println(" --diagrams           Outputs control flow diagrams at various stages of analysis for each method.");
    System.out.println(" --fulldiagrams       Outputs more detailed diagrams for each method.");
    System.out.println(" --nonportable        Allows analysis to produce CUDA code for JDK classes (untested).");
    System.out.println();
  }

  public static String getEnvironment(String key, String def) {
    String result = System.getenv(key);

    if(result == null) {
      return def;
    } else {
      return result;
    }
  }
}
