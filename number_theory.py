import sympy as sp

def is_special_number(n):
    return sp.isprime(n) and sp.fibonacci(n) == n

def find_special_numbers(limit):
    special_numbers = [n for n in range(2, limit) if is_special_number(n)]
    return special_numbers
