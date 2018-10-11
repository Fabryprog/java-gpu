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

import org.java.gpu.exceptions.UnsupportedInstruction;

import org.java.gpu.graph.Block;
import org.java.gpu.graph.CodeVisitor;
import org.java.gpu.graph.Method;
import org.java.gpu.graph.Type;

import org.java.gpu.graph.instructions.*;
import org.java.gpu.graph.state.*;

import org.java.gpu.util.TransformIterable;
import org.java.gpu.util.Utils;

import java.io.PrintStream;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 *
 */
public class CppGenerator extends CodeVisitor<String> {
  private int nextTemporary = 0;
  private PrintStream output;

  public CppGenerator(PrintStream output) {
    this.output = output;
  }

  private static final Set<String> mathSupported = new HashSet<String>() {{
    add("sin"); add("cos"); add("tan"); add("pow");
  }};

  public String assignTemporary(Type type, String value) {
    final String temporary = "t" + (nextTemporary++);

    output.print("const " + Helper.getType(type) + " " + temporary + " = ");
    output.println(value + ";");

    return temporary;
  }

  @Override
  public String visit(Instruction instruction) {
    throw new UnsupportedInstruction(instruction, "CUDA");
  }

  @Override
  public String visit(RestoreStack instruction) {
    return assignTemporary(
      instruction.getType(),
      "s" + Helper.getName(
        new Variable(instruction.getIndex(), instruction.getType())
      )
    );
  }

  @Override
  public String visit(Condition instruction) {
    String operator;

    // Choose operator symbol
    switch(instruction.getOperator()) {
      case EQ: operator = "==";  break;
      case NE: operator = "!=";  break;
      case LT: operator = "<";  break;
      case GE: operator = ">=";  break;
      case GT: operator = ">";  break;
      case LE: operator = "<=";  break;
      default: operator = "";
    }

    output.println(
      "if(" + instruction.getOperandA().accept(this) + operator +
      instruction.getOperandB().accept(this) + ") goto " +
      BlockExporter.getLabel(instruction.getDestination()) + ";"
    );

    return null;
  }

  @Override
  public String visit(Switch instruction) {
    output.println("switch(" + instruction.getOperand().accept(this) + ") {");

    for(Map.Entry<Integer, Block> mapping : instruction.getMapping().entrySet()) {
      output.print("  case " + mapping.getKey() + ": goto ");
      output.println(BlockExporter.getLabel(mapping.getValue()) + ";");
    }

    if(instruction.getDefault() != null) {
      output.println("  default: goto " + BlockExporter.getLabel(instruction.getDefault()) + ";");
    }

    output.println("}");

    return null;
  }

  @Override
  public String visit(Compare instruction) {
    String a = instruction.getOperandA().accept(this);
    String b = instruction.getOperandB().accept(this);

    // TODO: Correct treatment of NaNs (but GPUs don't support NaN properly!)
    return assignTemporary(
      instruction.getType(),
      "(" + a + "<" + b + ") ? -1 : (" + a + ">" + b + ") ? 1 : 0"
    );
  }

  @Override
  public String visit(Arithmetic instruction) {
    String operator;

    // Choose operator symbol
    switch(instruction.getOperator()) {
      case ADD: operator = "+";  break;
      case SUB: operator = "-";  break;
      case MUL: operator = "*";  break;
      case DIV: operator = "/";  break;
      case REM: operator = "%";  break;
      case AND: operator = "&";  break;
      case OR:  operator = "|";  break;
      case XOR: operator = "^";  break;
      case SHL: operator = "<<"; break;
      case SHR: operator = ">>"; break;
      case USHR:operator = ">>>";break;
      default:  operator = "";
    }

    return assignTemporary(
      instruction.getType(),
      instruction.getOperandA().accept(this) + operator +
      instruction.getOperandB().accept(this)
    );
  }

  @Override
  public String visit(Constant instruction) {
    // Object constants
    if(instruction.getType().getSort() == Type.Sort.REF) {
      throw new RuntimeException("Object/String constants not supported");
    // Primitive Types
    } else {
      switch(instruction.getType().getSort()) {
        // Longs (L suffix)
        case LONG:   return assignTemporary(
                       instruction.getType(),
                       instruction.getConstant().toString() + "L"
                     );
        // Floats (f suffix)
        case FLOAT: return assignTemporary(
                       instruction.getType(),
                       instruction.getConstant().toString() + "f"
                     );
        // Others
        default:     return assignTemporary(
                       instruction.getType(),
                       instruction.getConstant().toString()
                     );
      }
    }
  }

  @Override
  public String visit(Negate instruction) {
    return assignTemporary(
      instruction.getType(),
      "-" + instruction.getOperand().accept(this)
    );
  }

  @Override
  public String visit(Convert instruction) {
    return assignTemporary(
      instruction.getType(),
      "(" + Helper.getType(instruction.getType()) + ") " +
      instruction.getOperand().accept(this)
    );
  }

  @Override
  public String visit(Increment instruction) {
    output.println(
      Helper.getName(instruction.getState()) + " += " +
      instruction.getIncrement() + ";"
    );

    return null;
  }

  @Override
  public String visit(Read instruction) {
    State  state = instruction.getState();
    String value;

    // Object Fields
    if(state instanceof InstanceField) {
      InstanceField f = (InstanceField) state;

      value = "DEVPTR(" + f.getObject().accept(this) + ".device)->" + f.getField().getName();
    // Array Elements
    } else if(state instanceof ArrayElement) {
      ArrayElement e = (ArrayElement) state;

      value = "DEVPTR(" + e.getArray().accept(this) + ".device)[" + e.getIndex().accept(this) + "]";
    // Variables and Statics
    } else {
      // TODO: Math.E and Math.PI
      value = Helper.getName(state);
    }

    return assignTemporary(instruction.getType(), value);
  }

  @Override
  public String visit(Write instruction) {
    String line = "";
    State  state = instruction.getState();

    // Object Fields
    if(state instanceof InstanceField) {
      InstanceField f = (InstanceField) state;

      line += "DEVPTR(" + f.getObject().accept(this) + ".device)->" + f.getField().getName();
    // Array Elements
    } else if(state instanceof ArrayElement) {
      ArrayElement e = (ArrayElement) state;

      line += "DEVPTR(" + e.getArray().accept(this) + ".device)[" + e.getIndex().accept(this) + "]";
    // Variables and Statics
    } else {
      line += Helper.getName(state);
    }

    output.println(line + " = " + instruction.getValue().accept(this) + ";");

    return null;
  }

  @Override
  public String visit(Call instruction) {
    Method method = instruction.getMethod();

    // Math Functions (use built-in CUDA versions).
    if(method.getOwner().getName().equals("java/lang/Math")) {
      if(mathSupported.contains(method.getName())) {
        return assignTemporary(
          instruction.getType(),
          method.getName() + "(" +
          Utils.join(new TransformIterable<Producer,String>(Arrays.asList(instruction.getOperands())) {
            @Override
            protected String transform(Producer obj) {
              return obj.accept(CppGenerator.this);
            }
          }, ", ") + ")"
        );
      } else {
        throw new UnsupportedInstruction(
          instruction,
          "CUDA",
          "Math." + method.getName() + " is not supported yet."
        );
      }
    // Standard Method
    } else {
      if(method.getImplementation() != null) {
        String arguments = Utils.join(
          new TransformIterable<Producer, String>(Arrays.asList(instruction.getOperands())) {
            @Override
            protected String transform(Producer arg) {
              return arg.accept(CppGenerator.this);
            }
          },
          ", "
        );

        CUDAExporter.export(method);

        // Void return (no value).
        if(method.getReturnType() == Type.VOID) {
          output.println(Helper.getName(method) + "(" + arguments + ");");

          return null;
        // Value return.
        } else {
          return assignTemporary(
            instruction.getType(),
            Helper.getName(method) + "(" + arguments + ")"
          );
        }
      } else {
        throw new UnsupportedInstruction(
          instruction,
          "CUDA",
          method + " outside scope of transform."
        );
      }
    }
  }

  @Override
  public String visit(Return instruction) {
    output.println("return;");
    
    return null;
  }

  @Override
  public String visit(ValueReturn instruction) {
    output.println("return " + instruction.getOperand().accept(this) + ";");
    
    return null;
  }

  @Override
  public String visit(ArrayLength instruction) {
    return assignTemporary(
      instruction.getType(),
      instruction.getOperand().accept(this) + ".length"
    );
  }

  @Override
  public String visit(NewArray instruction) {
    return super.visit(instruction);
  }

  @Override
  public String visit(NewMultiArray instruction) {
    return super.visit(instruction);
  }

  @Override
  public String visit(NewObject instruction) {
    return super.visit(instruction);
  }
}
