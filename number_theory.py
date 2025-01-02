import sympy as sp

def is_special_number(n, fibonacci_set):
    return sp.isprime(n) and n in fibonacci_set

def find_special_numbers(limit):
    fibonacci_set = set()
    a, b = 1, 1
    while a < limit:
        fibonacci_set.add(a)
        a, b = b, a + b

    special_numbers = []
    for n in range(2, limit):
        if is_special_number(n, fibonacci_set):
            special_numbers.append(n)
    return special_numbers

if __name__ == "__main__":
    limit = 100
    print(find_special_numbers(limit))
