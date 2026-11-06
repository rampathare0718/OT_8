"""
Track F: Mini-Compiler Architecture & Three-Address Code (TAC) IR
Provides:
- TAC Instruction data structures (Op, Dest, Src1, Src2)
- Virtual Machine / Interpreter for TAC execution
- Correctness verification engine
"""

class TACInstruction:
    """Three-Address Code instruction: dest = src1 op src2, or JUMP, ASSIGN, PRINT."""
    def __init__(self, op, dest=None, src1=None, src2=None, label=None):
        self.op = op          # '+', '-', '*', '/', '=', 'LABEL', 'JUMP_IF_FALSE', 'GOTO', 'RETURN'
        self.dest = dest      # Destination register/variable name
        self.src1 = src1      # First operand (literal or variable)
        self.src2 = src2      # Second operand
        self.label = label    # Target label for jumps

    def __repr__(self):
        if self.op == "LABEL":
            return f"{self.label}:"
        elif self.op == "GOTO":
            return f"    goto {self.label}"
        elif self.op == "JUMP_IF_FALSE":
            return f"    if not {self.src1} goto {self.label}"
        elif self.op == "=":
            return f"    {self.dest} = {self.src1}"
        elif self.op == "RETURN":
            return f"    return {self.src1}"
        else:
            return f"    {self.dest} = {self.src1} {self.op} {self.src2}"

class TACInterpreter:
    """Executes a list of TAC instructions and records output and execution cycle count."""
    def __init__(self):
        self.environment = {}
        self.cycles = 0

    def _resolve(self, val):
        if isinstance(val, (int, float)):
            return val
        if val in self.environment:
            return self.environment[val]
        try:
            return int(val)
        except (ValueError, TypeError):
            try:
                return float(val)
            except (ValueError, TypeError):
                return self.environment.get(val, 0)

    def execute(self, instructions):
        self.environment = {}
        self.cycles = 0
        pc = 0
        n = len(instructions)

        # Build label index map
        labels = {}
        for idx, inst in enumerate(instructions):
            if inst.op == "LABEL":
                labels[inst.label] = idx

        return_val = None
        while pc < n:
            inst = instructions[pc]
            self.cycles += 1

            if inst.op == "LABEL":
                pc += 1
            elif inst.op == "=":
                self.environment[inst.dest] = self._resolve(inst.src1)
                pc += 1
            elif inst.op in ("+", "-", "*", "/", "<", "=="):
                v1 = self._resolve(inst.src1)
                v2 = self._resolve(inst.src2)
                if inst.op == "+":
                    res = v1 + v2
                elif inst.op == "-":
                    res = v1 - v2
                elif inst.op == "*":
                    res = v1 * v2
                elif inst.op == "/":
                    res = v1 / (v2 if v2 != 0 else 1e-9)
                elif inst.op == "<":
                    res = 1 if v1 < v2 else 0
                elif inst.op == "==":
                    res = 1 if v1 == v2 else 0
                self.environment[inst.dest] = res
                pc += 1
            elif inst.op == "JUMP_IF_FALSE":
                cond = self._resolve(inst.src1)
                if cond == 0:
                    pc = labels[inst.label]
                else:
                    pc += 1
            elif inst.op == "GOTO":
                pc = labels[inst.label]
            elif inst.op == "RETURN":
                return_val = self._resolve(inst.src1)
                break
            else:
                pc += 1

        return return_val, self.cycles, self.environment

