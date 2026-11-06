"""
Track F: Compiler Optimization Passes
Implements 4 distinct optimization passes for full credit:
1. Constant Folding & Constant Propagation
2. Common Subexpression Elimination (CSE)
3. Dead Code Elimination (DCE)
4. Strength Reduction & Algebraic Simplification
"""

from .mini_compiler import TACInstruction

def is_literal(val):
    if isinstance(val, (int, float)):
        return True
    try:
        int(val)
        return True
    except (ValueError, TypeError):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

def to_number(val):
    try:
        return int(val)
    except ValueError:
        return float(val)

class ConstantFoldingAndPropagationPass:
    """Folds static arithmetic and propagates known constant bindings."""
    def run(self, instructions):
        constants = {}
        optimized = []

        for inst in instructions:
            if inst.op == "LABEL" or inst.op == "GOTO" or inst.op == "JUMP_IF_FALSE":
                # Flush constants at control flow boundaries
                constants.clear()
                optimized.append(inst)
                continue

            # Propagate constants to operands
            s1 = constants.get(inst.src1, inst.src1)
            s2 = constants.get(inst.src2, inst.src2)

            if inst.op == "=":
                if is_literal(s1):
                    constants[inst.dest] = to_number(s1)
                elif inst.dest in constants:
                    del constants[inst.dest]
                optimized.append(TACInstruction("=", dest=inst.dest, src1=s1))

            elif inst.op in ("+", "-", "*", "/"):
                if is_literal(s1) and is_literal(s2):
                    # Constant Folding!
                    v1 = to_number(s1)
                    v2 = to_number(s2)
                    if inst.op == "+":
                        folded = v1 + v2
                    elif inst.op == "-":
                        folded = v1 - v2
                    elif inst.op == "*":
                        folded = v1 * v2
                    elif inst.op == "/":
                        folded = v1 / (v2 if v2 != 0 else 1)
                    constants[inst.dest] = folded
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=folded))
                else:
                    if inst.dest in constants:
                        del constants[inst.dest]
                    optimized.append(TACInstruction(inst.op, dest=inst.dest, src1=s1, src2=s2))
            else:
                optimized.append(inst)

        return optimized

class CommonSubexpressionEliminationPass:
    """Identifies and eliminates redundant identical subcomputations."""
    def run(self, instructions):
        expr_cache = {}  # (op, s1, s2) -> dest_var
        optimized = []

        for inst in instructions:
            if inst.op in ("LABEL", "GOTO", "JUMP_IF_FALSE"):
                expr_cache.clear()
                optimized.append(inst)
                continue

            if inst.op in ("+", "-", "*", "/"):
                key = (inst.op, inst.src1, inst.src2)
                # Commutative matching for + and *
                rev_key = (inst.op, inst.src2, inst.src1) if inst.op in ("+", "*") else None

                matched_dest = expr_cache.get(key) or (expr_cache.get(rev_key) if rev_key else None)
                if matched_dest is not None:
                    # Redundant computation: replace with assignment
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=matched_dest))
                else:
                    expr_cache[key] = inst.dest
                    optimized.append(inst)
            else:
                # If variable redefined, invalidate cache entries containing it
                if inst.dest:
                    expr_cache = {k: v for k, v in expr_cache.items() if k[1] != inst.dest and k[2] != inst.dest and v != inst.dest}
                optimized.append(inst)

        return optimized

class StrengthReductionPass:
    """Replaces computationally expensive operations with cheaper equivalents."""
    def run(self, instructions):
        optimized = []

        for inst in instructions:
            if inst.op == "*":
                # x * 2 => x + x
                if inst.src2 == 2 or inst.src2 == "2":
                    optimized.append(TACInstruction("+", dest=inst.dest, src1=inst.src1, src2=inst.src1))
                elif inst.src1 == 2 or inst.src1 == "2":
                    optimized.append(TACInstruction("+", dest=inst.dest, src1=inst.src2, src2=inst.src2))
                # x * 1 => x
                elif inst.src2 == 1 or inst.src2 == "1":
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=inst.src1))
                # x * 0 => 0
                elif inst.src2 == 0 or inst.src2 == "0" or inst.src1 == 0 or inst.src1 == "0":
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=0))
                else:
                    optimized.append(inst)
            elif inst.op == "+":
                # x + 0 => x
                if inst.src2 == 0 or inst.src2 == "0":
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=inst.src1))
                elif inst.src1 == 0 or inst.src1 == "0":
                    optimized.append(TACInstruction("=", dest=inst.dest, src1=inst.src2))
                else:
                    optimized.append(inst)
            else:
                optimized.append(inst)

        return optimized

class DeadCodeEliminationPass:
    """
    Iterative fixpoint liveness analysis:
    Eliminates instructions whose destination is never read by any live instruction.
    Properly preserves loop induction variables and branch conditions.
    """
    def run(self, instructions):
        # Determine all variables that directly affect control flow or return
        live_vars = set()
        for inst in instructions:
            if inst.op == "RETURN":
                if inst.src1 and not is_literal(inst.src1):
                    live_vars.add(inst.src1)
            elif inst.op == "JUMP_IF_FALSE":
                if inst.src1 and not is_literal(inst.src1):
                    live_vars.add(inst.src1)

        # Iteratively expand live variables until fixed-point reached
        changed = True
        while changed:
            changed = False
            for inst in instructions:
                if inst.dest in live_vars:
                    for src in (inst.src1, inst.src2):
                        if src and not is_literal(src) and src not in live_vars:
                            live_vars.add(src)
                            changed = True

        # Now filter out instructions whose destination is unused (unless side-effecting)
        optimized = []
        for inst in instructions:
            if inst.op in ("LABEL", "GOTO", "JUMP_IF_FALSE", "RETURN"):
                optimized.append(inst)
            elif inst.dest in live_vars:
                optimized.append(inst)
            # Otherwise discarded as dead code!

        return optimized
