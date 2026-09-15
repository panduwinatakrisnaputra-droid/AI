# Exercise 2.13.1

import math

print('Input a list of float numbers:')
numbers = input().split()

for x in numbers:
    x = float(x)
    y = math.sin(x)
    print('The sine of ' + str(x) + ' is ' + str(y))