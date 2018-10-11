/*
 * Parallelising JVM Compiler
 * Part II Project, Computer Science Tripos
 *
 * Copyright (c) 2009, 2010 - Peter Calvert, University of Cambridge
 */

package org.java.gpu.exceptions;

import org.java.gpu.graph.instructions.*;

/**
 * Exception that can be thrown by exporters if the instruction can not be
 * performed in the given output. This can also be used to indicate that the
 * instruction export has not been implemented.
 */
public class UnsupportedInstruction extends RuntimeException {
  /**
   * Instruction concerned.
   */
  private Instruction instruction;

  /**
   * Export target.
   */
  private String target;

  /**
   * Reason for failure (optional).
   */
  private String reason;

  /**
   * Constructs the exception in the case where all details are given.
   *
   * @param ins    Unsupported instruction.
   * @param target Target concerned.
   * @param reason Full reason that export failed.
   */
  public UnsupportedInstruction(Instruction ins, String target, String reason) {
    this.instruction = ins;
    this.target      = target;
    this.reason      = reason;
  }

  /**
   * Constructs the exception in the case without a full reason.
   *
   * @param ins    Unsupported instruction.
   * @param target Target concerned.
   */
  public UnsupportedInstruction(Instruction ins, String target) {
    this(ins, target, "");
  }

  /**
   * Returns the actual instruction node that caused the exception.
   *
   * @return       Instruction.
   */
  public Instruction getSpecificInstruction() {
    return instruction;
  }

  /**
   * Returns the name of the instruction that caused the exception.
   *
   * @return       Instruction name.
   */
  public String getInstruction() {
    return instruction.getClass().getSimpleName();
  }

  /**
   * Returns the target that threw the exception.
   *
   * @return       Export target.
   */
  public String getTarget() {
    return target;
  }

  /**
   * Returns the full reason for the failure.
   *
   * @return       Full reason.
   */
  public String getReason() {
    return reason;
  }

  /**
   * Full description of exception.
   *
   * @return       Description.
   */
  @Override
  public String toString() {
    return "Could not export '" + getInstruction() + "' to '" + getTarget()
         + "': " + getReason();
  }
}
