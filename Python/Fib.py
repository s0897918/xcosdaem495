class Fib:

    def __init__ (self):
        self.n = 10

    def fib(self, n):    # write Fibonacci series up to n
        a, b = 0, 1
        while a < n:
            print(a, end=' ')
            a, b = b, a+b
        print()

class NewFib(Fib):
    def func(self):
        print(self.n)
        
        
