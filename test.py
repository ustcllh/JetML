import numpy as np


ifs = open('./training/jewel/jewel_1')

line = ifs.readline()

line = np.fromstring(line, dtype=float, sep=',')

line = np.reshape(line, (-1, 5))

print(line)


a = [1, 2, 3]

b = [2, 3, 4]

print(a+b)

a.append(b)

print(a)
