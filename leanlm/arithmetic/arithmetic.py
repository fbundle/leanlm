from typing import Literal

import numpy as np
from pydantic import BaseModel

type Operator = Literal["+", "-", "*", "/", "%"]


class Operation(BaseModel):
    op: Operator
    a: int
    b: int

    def calculate(self) -> int:
        if self.op == "+":
            return self.a + self.b
        elif self.op == "-":
            return self.a - self.b
        elif self.op == "*":
            return self.a * self.b
        elif self.op == "/":
            return self.a // self.b
        elif self.op == "%":
            return self.a % self.b
        else:
            raise RuntimeError("op")

    def marshal(self) -> str:
        return f"{self.a} {self.op} {self.b}"

def unmarshal(input_text: str) -> Operation:
    for op in ["+", "-", "*", "/", "%"]:
        parts = input_text.split(op)
        if len(parts) == 2:
            return Operation(op=op, a=int(parts[0]), b=int(parts[1]))

    raise RuntimeError("unmarshal")

def cap(x: int, m: int) -> int:
    # if x > m, give m
    return min(x, m)


def generate_input(p: float, m: int = 18) -> str:
    a = np.random.randint(10 ** cap(np.random.geometric(p), m=m))
    b = np.random.randint(10 ** cap(np.random.geometric(p), m=m))
    if b == 0:
        op = np.random.choice(["+", "-", "*"])
    else:
        op = np.random.choice(["+", "-", "*", "/", "%"])

    operation = Operation(op=op, a=a, b=b)
    return operation.marshal()


def get_expected_output(input_text: str) -> str:
    operation = unmarshal(input_text)
    return str(operation.calculate())
