import shubert
import numpy

f = lambda x: x*x
a = 0
b = 4
L = 8
e = 0.1
minimo = shubert.evaluate_min(f, a, b, L + 1, e)
print(minimo)

f = numpy.sin
L=1
a=0
b = 2*numpy.pi
minimo = shubert.evaluate_min(f, a, b, L + 1, e)
print(minimo)