from ROOT import TFile
from sklearn import metrics
import matplotlib.pyplot as plt

# mult = 1700
mult = 7000

# input files
f1 = '../data/Classified/pythia_csejet_pt200_rg0p1_mult' + str(mult) + '_merged.root'
f2 = '../data/Classified/jewel_R_csejet_pt200_rg0p1_mult'+ str(mult) + '_merged.root'

# jet pt cut
jetpt_min = 200
jetpt_max = 400

nbins = 25

def calibrate(file_neg, file_pos):
    x = []
    y = []
    weights = []

    f1 = TFile(file_neg, "READ")
    tr = f1.Get('jet')
    max = tr.GetEntriesFast()
    idx=0
    weight_sum = 0
    weights_neg = []
    while idx<max:
        tr.GetEntry(idx)
        if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
            idx += 1
            continue

        weight_sum += tr.weight

        x.append(0)
        y.append(tr.lstm)
        weights_neg.append(tr.weight)
        idx+=1

    for i in range(len(weights_neg)):
        weights.append(weights_neg[i]/weight_sum)

    f2 = TFile(file_pos, "READ")
    tr = f2.Get('jet')
    max = tr.GetEntriesFast()
    idx=0
    weight_sum = 0
    weights_pos = []
    while idx<max:
        tr.GetEntry(idx)
        if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
            idx += 1
            continue

        weight_sum += tr.weight

        x.append(1)
        y.append(tr.lstm)
        weights_pos.append(tr.weight)
        idx+=1

    for i in range(len(weights_pos)):
        weights.append(weights_pos[i]/weight_sum)

    fpr, tpr, thresholds = metrics.roc_curve(x, y, sample_weight=weights, pos_label=1)

    d_max = [1 for i in range(nbins)]
    tpr_calibrated = [0 for i in range(nbins)]
    fpr_calibrated = [0 for i in range(nbins)]
    threshold_calibrated = [0 for i in range(nbins)]
    for i in range(len(thresholds)):
        for j in range(nbins):
            d = pow(1. - 1./nbins*(j+1) - fpr[i], 2)
            if d<d_max[j]:
                d_max[j] = d
                fpr_calibrated[j] = fpr[i]
                tpr_calibrated[j] = tpr[i]
                threshold_calibrated[j] = thresholds[i]

    threshold_calibrated.append(1.0)
    return threshold_calibrated


def rebin(file, thresholds):
    f = TFile(file, "READ")
    tr = f.Get('jet')
    max = tr.GetEntriesFast()
    idx=0
    weight_sum = 0
    lstm_calibrated = [0 for i in range(nbins)]
    while idx<max:
        tr.GetEntry(idx)
        if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
            idx += 1
            continue

        weight_sum += tr.weight

        i=0
        while tr.lstm > thresholds[i]:
            i+=1
        lstm_calibrated[i] += tr.weight
        idx+=1

    for i in range(nbins):
        lstm_calibrated[i] /= weight_sum*1./nbins
    return lstm_calibrated




#############################################
# LSTM after calib
#############################################


thresholds = calibrate(f1, f2)
lstm_pythia_calibrated = rebin(f1, thresholds)
lstm_jewel_calibrated = rebin(f2, thresholds)

bins =[0]
for i in range(nbins):
    bins.append((i+1)*1./nbins)

fig,ax = plt.subplots()

cblue = (0.,0.,1.,0.5)
cred = (1.,0.,0.,0.6)

x = [(0.5+i)*1./nbins for i in range(nbins)]

_, _, patches = plt.hist([x,x], bins, weights=[lstm_pythia_calibrated, lstm_jewel_calibrated], label=['Pythia8 + Most-central BG', 'Jewel + Most-central BG'], color=[cblue, cred], rwidth=0.9)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.01, 7.5])

plt.xlabel("Calibrated LSTM", fontsize=30)
plt.ylabel("Density", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


plt.legend(loc='upper left', fontsize=26, frameon=False) 
plt.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.12)

plt.gcf().set_size_inches(8, 8)
plt.savefig("lstm_calibrated.pdf")
