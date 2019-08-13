import matplotlib.pyplot as plt

ifs = open('results.txt', 'r')
lines = ifs.readlines()
e = []
lstm = []
cnn = []
m = []
z = []
delta = []
jet_ptd = []
jet_mass = []
jet_width = []
for line in lines:
    vars = line.split()
    e.append(float(vars[3]))
    lstm.append(float(vars[4]))
    cnn.append(float(vars[5]))
    m.append(float(vars[6]))
    z.append(float(vars[7]))
    delta.append(float(vars[8]))
    jet_ptd.append(float(vars[9]))
    jet_mass.append(float(vars[10]))
    jet_width.append(float(vars[11]))

def var_plot(var, name, bins=100):
    plt.clf()
    plt.hist(var, bins=bins)
    plt.savefig(name)

var_plot(e, './plot/e.png')
var_plot(lstm, './plot/lstm.png')
var_plot(cnn, './plot/cnn.png')
var_plot(m, './plot/m.png')
var_plot(z, './plot/z.png')
var_plot(delta, './plot/delta.png')
var_plot(jet_ptd, './plot/ptd.png')
var_plot(jet_mass, './plot/jet_mass.png')
var_plot(jet_width, './plot/width.png')
