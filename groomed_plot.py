import matplotlib.pyplot as plt

ifs = open('results_groomed.txt', 'r')
lines = ifs.readlines()

m = []
z = []
delta = []

for line in lines:
    vars = line.split()
    if not vars:
        continue
    m.append(float(vars[0]))
    z.append(float(vars[1]))
    delta.append(float(vars[2]))


def var_plot(var, name, bins=100):
    plt.clf()
    plt.hist(var, bins=bins)
    plt.savefig(name)


var_plot(m, './plot/m_groomed.png')
var_plot(z, './plot/z_groomed.png')
var_plot(delta, './plot/delta_groomed.png')
