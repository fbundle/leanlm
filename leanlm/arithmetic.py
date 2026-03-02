import numpy as np
import jiwer

def calculate(op_str: str, a: int, b: int) -> int:
    if op_str == "+":
        return a + b
    if op_str == "*":
        return a * b
    raise RuntimeError("op")

def cap(x: int, m: int) -> int:
    # if x > m, give m
    return min(x, m)

def generate_input(m: int = 18) -> str:
    p = 0.5
    a = np.random.randint(10**cap(np.random.geometric(p), m=m))
    b = np.random.randint(10**cap(np.random.geometric(p), m=m))
    op = np.random.choice(["+", "*"])
    input_str = "".join([str(a), op, str(b), "="])
    return input_str

def parse_input(input_str: str) -> tuple[str, int, int]:
    input_str = input_str.split("=", maxsplit=1)[0]
    parts = input_str.split("+")
    if len(parts) == 2:
        return "+", int(parts[0]), int(parts[1])
    parts = input_str.split("*")
    if len(parts) == 2:
        return "*", int(parts[0]), int(parts[1])
    raise RuntimeError("parse")

def match_output(input_str: str, output_str: str) -> float:
    op, a, b = parse_input(input_str)
    expected_str = str(calculate(op, a, b))
    cer = jiwer.cer(expected_str, output_str)
    return - cer
