from ROOT import TFile,TH1F,TH2F,TCanvas,gPad,TLatex,TLegend,TPad,TLine,THStack
from ROOT import TColor
from utils import DrawFrame,SetFillStyle,SetLineStyle,SetMarkerStyle
from math import sqrt,log
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# choose background level

# mult = 1700
mult = 7000

# input files
f1 = TFile('../data/Classified/pythia_csejet_pt200_rg0p1_mult' + str(mult) + '_merged.root', "READ")
f2 = TFile('../data/Classified/jewel_R_csejet_pt200_rg0p1_mult'+ str(mult) + '_merged.root', "READ")

# jet pt cut
jetpt_min = 200
jetpt_max = 400


#############################################
# ROC
#############################################

# roc
x = []
y = []
weight = []

# pythia
tr = f1.Get('jet')
max = tr.GetEntriesFast()
idx=0
weight_sum = 0
weight_pythia = []
while idx<max:
    tr.GetEntry(idx)
    if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
        idx += 1
        continue

    weight_sum += tr.weight

    x.append(0)
    y.append(tr.lstm)
    weight_pythia.append(tr.weight)
    idx+=1

for w in weight_pythia:
    weight.append(w/weight_sum)


tr = f2.Get('jet')
max = tr.GetEntriesFast()
idx=0
weight_sum = 0
weight_jewel = []
while idx<max:
    tr.GetEntry(idx)
    if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
        idx += 1
        continue

    weight_sum += tr.weight

    x.append(1)
    y.append(tr.lstm)
    weight_jewel.append(tr.weight)
    idx+=1

for w in weight_jewel:
    weight.append(w/weight_sum)


fpr, tpr, thresholds = metrics.roc_curve(x, y, sample_weight=weight, pos_label=1)

# threshold with tpr=0.4
fpr_select = 0
tpr_select = 0
threshold = 0
d_max = 1


for i in range(len(thresholds)):
    d = pow(0.4-tpr[i], 2)
    if d<d_max:
        d_max = d
        fpr_select = fpr[i]
        tpr_select = tpr[i]
        threshold = thresholds[i]

print("TPR: %0.2f, FPR: %0.2f, Threshold: %0.4f" % (tpr_select, fpr_select, threshold))

auc = metrics.roc_auc_score(x, y,sample_weight=weight)

plt.figure()
lw = 2
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC Curve (AUC=%0.3f)" % auc)
plt.plot([fpr_select], [tpr_select], 'ro', label=r"TPR=%0.2f, FPR=%0.2f" % (tpr_select, fpr_select))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.15, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=30)
plt.ylabel("True Positive Rate (TPR)", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.title("Receiver Operating Characteristic (ROC)", fontsize=18)
plt.legend(loc="lower right", fontsize=26)

plt.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.16)
plt.gcf().set_size_inches(8, 6)
plt.savefig("roc.pdf")


#############################################
# hist
############################################# 

# hist of lstm (matplotlib)
lstm_pythia = []
weight_pythia = []

lstm_jewel = []
weight_jewel = []

# hist of zg
h_z_stack_jewel = THStack("hs_z", "")

h_z_jewel_1 = TH1F("h_z_jewel_1", "", 10, 0, 0.5)
h_z_jewel_2 = TH1F("h_z_jewel_2", "", 10, 0, 0.5)

h_z_pythia_1 = TH1F("h_z_pythia_1", "", 10, 0, 0.5)
h_z_pythia_2 = TH1F("h_z_pythia_2", "", 10, 0, 0.5)

h_z_pythia = TH1F("h_z_pythia", "", 10, 0, 0.5)
h_z_jewel = TH1F("h_z_jewel", "", 10, 0, 0.5)

# hist of Rg (delta)

h_delta_stack_jewel = THStack("hs_delta", "")

h_delta_jewel_1 = TH1F("h_delta_jewel_1", "", 16, 0, 0.4)
h_delta_jewel_2 = TH1F("h_delta_jewel_2", "", 16, 0, 0.4)

h_delta_pythia_1 = TH1F("h_delta_pythia_1", "", 16, 0, 0.4)
h_delta_pythia_2 = TH1F("h_delta_pythia_2", "", 16, 0, 0.4)

h_delta_pythia = TH1F("h_delta_pythia", "", 16, 0, 0.4)
h_delta_jewel = TH1F("h_delta_jewel", "", 16, 0, 0.4)

# hist of mg/jetpt
# mg : groomed jet mass

h_m_stack_jewel = THStack("hs_m", "")

h_m_jewel_1 = TH1F("h_m_jewel_1", "", 12, 0, 0.3)
h_m_jewel_2 = TH1F("h_m_jewel_2", "", 12, 0, 0.3)

h_m_pythia_1 = TH1F("h_m_pythia_1", "", 12, 0, 0.3)
h_m_pythia_2 = TH1F("h_m_pythia_2", "", 12, 0, 0.3)

h_m_pythia = TH1F("h_m_pythia", "", 12, 0, 0.3)
h_m_jewel = TH1F("h_m_jewel", "", 12, 0, 0.3)


# lund

h2_pythia = TH2F("h2_pythia", "h2_pythia", 200, 0, 5, 140, -7, 0)
h2_pythia_rebin = TH2F("h2_pythia_rebin", "h2_pythia_rebin", 100, 0, 5, 70, -7, 0)

h2_jewel = TH2F("h2_jewel", "h2_jewel", 200, 0, 5, 140, -7, 0)
h2_jewel_1 = TH2F("h2_jewel_1", "h2_jewel_1", 100, 0, 5, 70, -7, 0)
h2_jewel_2 = TH2F("h2_jewel_2", "h2_jewel_2", 100, 0, 5, 70, -7, 0)



# pythia jets
tr = f1.Get('jet')
max = tr.GetEntriesFast()
idx=0
weight_sum = 0
while idx<max:
    tr.GetEntry(idx)
    if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
        idx += 1
        continue

    weight_sum += tr.weight

    lstm_pythia.append(tr.lstm)
    weight_pythia.append(tr.weight)

    h_z_pythia.Fill(tr.z[0], tr.weight)
    h_delta_pythia.Fill(tr.delta[0], tr.weight)
    h_m_pythia.Fill(tr.m[0]/tr.jetpt, tr.weight)

    h2_pythia.Fill(-log(tr.delta[0]), log(tr.delta[0]*tr.z[0]), tr.weight)
    h2_pythia_rebin.Fill(-log(tr.delta[0]), log(tr.delta[0]*tr.z[0]), tr.weight)

    if tr.lstm>threshold:
        h_z_pythia_1.Fill(tr.z[0], tr.weight)
        h_delta_pythia_1.Fill(tr.delta[0], tr.weight)
        h_m_pythia_1.Fill(tr.m[0]/tr.jetpt, tr.weight)
    else:
        h_z_pythia_2.Fill(tr.z[0], tr.weight)
        h_delta_pythia_2.Fill(tr.delta[0], tr.weight)
        h_m_pythia_2.Fill(tr.m[0]/tr.jetpt, tr.weight)

    idx+=1

h_z_pythia.Sumw2()
h_z_pythia.Scale(1./h_z_pythia.Integral()/0.05)
SetMarkerStyle(h_z_pythia, 21, 1, 1, 1.3)
SetLineStyle(h_z_pythia, 1, 1, 1, 3)
SetFillStyle(h_z_pythia, 1001, 1, 0.0)

h_delta_pythia.Sumw2()
h_delta_pythia.Scale(1./h_delta_pythia.Integral()/0.025)
SetMarkerStyle(h_delta_pythia, 21, 1, 1, 1.3)
SetLineStyle(h_delta_pythia, 1, 1, 1, 3)
SetFillStyle(h_delta_pythia, 1001, 1, 0.0)

h_m_pythia.Sumw2()
h_m_pythia.Scale(1./h_m_pythia.Integral()/0.025)
SetMarkerStyle(h_m_pythia, 21, 1, 1, 1.3)
SetLineStyle(h_m_pythia, 1, 1, 1, 3)
SetFillStyle(h_m_pythia, 1001, 1, 0.0)

h2_pythia.Scale(1./weight_sum/0.00125)
h2_pythia_rebin.Scale(1./weight_sum/0.005)


# jewel jets
tr = f2.Get('jet')
max = tr.GetEntriesFast()
idx=0
weight_sum = 0
while idx<max:
    tr.GetEntry(idx)
    if tr.depth==0 or tr.jetpt<jetpt_min or tr.jetpt>jetpt_max or tr.delta[0]<0.1:
        idx += 1
        continue

    weight_sum += tr.weight

    lstm_jewel.append(tr.lstm)
    weight_jewel.append(tr.weight)

    h_z_jewel.Fill(tr.z[0], tr.weight)
    h_delta_jewel.Fill(tr.delta[0], tr.weight)
    h_m_jewel.Fill(tr.m[0]/tr.jetpt, tr.weight)

    h2_jewel.Fill(-log(tr.delta[0]), log(tr.delta[0]*tr.z[0]), tr.weight)

    if tr.lstm>threshold:
        h_z_jewel_1.Fill(tr.z[0], tr.weight)
        h_delta_jewel_1.Fill(tr.delta[0], tr.weight)
        h_m_jewel_1.Fill(tr.m[0]/tr.jetpt, tr.weight)
        h2_jewel_1.Fill(-log(tr.delta[0]), log(tr.delta[0]*tr.z[0]), tr.weight)

    else:
        h_z_jewel_2.Fill(tr.z[0], tr.weight)
        h_delta_jewel_2.Fill(tr.delta[0], tr.weight)
        h_m_jewel_2.Fill(tr.m[0]/tr.jetpt, tr.weight)
        h2_jewel_2.Fill(-log(tr.delta[0]), log(tr.delta[0]*tr.z[0]), tr.weight)

    idx+=1


h_z_jewel_1.Sumw2()
h_z_jewel_2.Sumw2()

h_z_jewel_1.Scale(1./h_z_jewel.Integral()/0.05)
h_z_jewel_2.Scale(1./h_z_jewel.Integral()/0.05)

ci = TColor.GetFreeColorIndex()
color = TColor(ci, 1.,0.4,0.4,"",1.)

SetMarkerStyle(h_z_jewel_1, 20, ci, 1, 1.3)
SetLineStyle(h_z_jewel_1, 1, ci, 1, 1)
SetFillStyle(h_z_jewel_1, 1001, ci, 1)

SetMarkerStyle(h_z_jewel_2, 20, 9, 1, 1.3)
SetLineStyle(h_z_jewel_2, 1, 9, 1, 1)
SetFillStyle(h_z_jewel_2, 1001, 9, 1.0)

h_z_jewel.Sumw2()
h_z_jewel.Scale(1./h_z_jewel.Integral()/0.05)
SetMarkerStyle(h_z_jewel, 20, 2, 1, 1.3)
SetLineStyle(h_z_jewel, 1, 2, 1, 3)
SetFillStyle(h_z_jewel, 1001, 2, 0.4)


h_delta_jewel_1.Sumw2()
h_delta_jewel_2.Sumw2()

h_delta_jewel_1.Scale(1./h_delta_jewel.Integral()/0.025)
h_delta_jewel_2.Scale(1./h_delta_jewel.Integral()/0.025)

SetMarkerStyle(h_delta_jewel_1, 20, ci, 1, 1.3)
SetLineStyle(h_delta_jewel_1, 1, ci, 1, 1)
SetFillStyle(h_delta_jewel_1, 1001, ci, 1)

SetMarkerStyle(h_delta_jewel_2, 20, 9, 1, 1.3)
SetLineStyle(h_delta_jewel_2, 1, 9, 1, 1)
SetFillStyle(h_delta_jewel_2, 1001, 9, 1.0)

h_delta_jewel.Sumw2()
h_delta_jewel.Scale(1./h_delta_jewel.Integral()/0.025)
SetMarkerStyle(h_delta_jewel, 20, 2, 1, 1.3)
SetLineStyle(h_delta_jewel, 1, 2, 1, 3)
SetFillStyle(h_delta_jewel, 1001, 2, 0.4)

h_m_jewel_1.Sumw2()
h_m_jewel_2.Sumw2()

h_m_jewel_1.Scale(1./h_m_jewel.Integral()/0.025)
h_m_jewel_2.Scale(1./h_m_jewel.Integral()/0.025)

SetMarkerStyle(h_m_jewel_1, 20, ci, 1, 1.3)
SetLineStyle(h_m_jewel_1, 1, ci, 1, 1)
SetFillStyle(h_m_jewel_1, 1001, ci, 1)

SetMarkerStyle(h_m_jewel_2, 20, 9, 1, 1.3)
SetLineStyle(h_m_jewel_2, 1, 9, 1, 1)
SetFillStyle(h_m_jewel_2, 1001, 9, 1.0)

h_m_jewel.Sumw2()
h_m_jewel.Scale(1./h_m_jewel.Integral()/0.025)
SetMarkerStyle(h_m_jewel, 20, 2, 1, 1.3)
SetLineStyle(h_m_jewel, 1, 2, 1, 3)
SetFillStyle(h_m_jewel, 1001, 2, 0.4)

h2_jewel.Scale(1./weight_sum/0.00125)
h2_jewel_1.Scale(1./weight_sum/0.005)
h2_jewel_2.Scale(1./weight_sum/0.005)

# div

div_z_1 = h_z_jewel_1.Clone()
div_z_2 = h_z_jewel_2.Clone()
divided_z = h_z_pythia.Clone()
div_z_1.Divide(divided_z)
div_z_2.Divide(divided_z)


div_delta_1 = h_delta_jewel_1.Clone()
div_delta_2 = h_delta_jewel_2.Clone()
divided_delta = h_delta_pythia.Clone()
div_delta_1.Divide(divided_delta)
div_delta_2.Divide(divided_delta)

div_m_1 = h_m_jewel_1.Clone()
div_m_2 = h_m_jewel_2.Clone()
divided_m = h_m_pythia.Clone()
div_m_1.Divide(divided_m)
div_m_2.Divide(divided_m)

#############################################
# LSTM (matplotlib)
#############################################

bins = np.linspace(0, 1., 26)

plt.figure()

cblue = (0.,0.,1.,0.5)
cred = (1.,0.,0.,0.6)

if mult == 1700:
    labels = ['Pythia8 + Mid-central BG', 'Jewel + Mid-central BG']
else:
    labels = ['Pythia8 + Most-central BG', 'Jewel + Most-central BG']
plt.hist([lstm_pythia, lstm_jewel], weights=[weight_pythia, weight_jewel], bins=bins, label=labels, color=[cblue, cred], rwidth=0.9, density=True)

plt.plot([threshold, threshold], [0, 5], color='r', lw=lw, linestyle="--", label=r"TPR=%0.2f, FPR=%0.2f" % (tpr_select, fpr_select))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.01, 7.5])
# plt.ylim([-0.01, 8.5])
plt.xlabel("Raw LSTM Output", fontsize=30)
plt.ylabel("Density", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', fontsize=26, frameon=False) 
plt.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.12)
plt.gcf().set_size_inches(8, 8)
plt.savefig("lstm.pdf")


#############################################
# stack plot stylizer
#############################################

def style_upper_frame(frame):
    frame.GetYaxis().SetLabelFont(43)
    frame.GetYaxis().SetTitleFont(43)
    frame.GetYaxis().SetLabelSize(28)
    frame.GetYaxis().SetTitleSize(28)
    frame.GetYaxis().SetTitleOffset(2.0)

    frame.GetXaxis().SetNdivisions(505)

def style_lower_frame(frame):

    frame.GetYaxis().SetLabelFont(43)
    frame.GetYaxis().SetTitleFont(43)
    frame.GetYaxis().SetLabelSize(28)
    frame.GetYaxis().SetTitleSize(28)
    frame.GetYaxis().SetTitleOffset(2.)
    frame.GetYaxis().SetNdivisions(505)

    frame.GetXaxis().SetLabelFont(43)
    frame.GetXaxis().SetTitleFont(43)
    frame.GetXaxis().SetLabelSize(28)
    frame.GetXaxis().SetTitleSize(40)
    frame.GetXaxis().SetTitleOffset(2.0)
    frame.GetXaxis().SetNdivisions(505)


#############################################
# z stack
#############################################
c1 = TCanvas("c1", "c1", 600, 800)
c1.cd()
p1 = TPad("p1", "p1", 0., 0.4, 1., 1.)
p1.SetTopMargin(0.02)
p1.SetBottomMargin(0)
p1.SetLeftMargin(0.18)
p1.SetRightMargin(0.02)
p1.Draw()
p1.cd()
# p1.SetLogy()

title = ";;#frac{1}{N} #frac{dN}{dz_{g}}"
frame1 = DrawFrame(0.08, 0.52, 0.10, 15.9, title, False)
style_upper_frame(frame1)

Tl = TLatex()
if mult == 1700:
    s1 = "Mixed with Mid-central BG"
else:
    s1 = "Mixed with Most-central BG"
s2 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s3 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"

Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.DrawLatex(0.22,0.90, s1)
Tl.DrawLatex(0.22,0.835, s2)
Tl.DrawLatex(0.22,0.755, s3)

# legend

lg = TLegend(0.50, 0.5, 0.88, 0.72)
lg.AddEntry(h_z_pythia, "Pythia8", "f")
# lg.AddEntry(h_z_jewel_1, "Jewel (LSTM > {:0.3f})".format(threshold), "f")
# lg.AddEntry(h_z_jewel_2, "Jewel (LSTM < {:0.3f})".format(threshold), "f")
lg.AddEntry(h_z_jewel_1, "Jewel (Top 40%)", "f")
lg.AddEntry(h_z_jewel_2, "Jewel (Bot 60%)", "f")

lg.SetBorderSize(0)
lg.SetTextSize(0.06)
lg.Draw("same")


h_z_stack_jewel.Add(h_z_jewel_2)
h_z_stack_jewel.Add(h_z_jewel_1)

h_z_stack_jewel.Draw("hist same")
h_z_pythia.Draw("hist same")

c1.cd()

p2 = TPad("p2", "p2", 0., 0., 1., 0.4)
p2.SetTopMargin(0)
p2.SetBottomMargin(0.25)
p2.SetLeftMargin(0.18)
p2.SetRightMargin(0.02)
p2.Draw()
p2.cd()
frame2 = DrawFrame(0.08, 0.52, -0.1, 1.9, ";z_{g};Ratio", False)
style_lower_frame(frame2)


lg2 = TLegend(0.3, 0.78, 0.85, 0.98)
# lg2.AddEntry(div_z_1, "Jewel (LSTM > {:0.3f}) / Pythia8".format(threshold), "lep")
# lg2.AddEntry(div_z_2, "Jewel (LSTM < {:0.3f}) / Pythia8".format(threshold), "lep")
lg2.AddEntry(div_z_1, "Jewel (Top 40%) / Pythia8", "lep")
lg2.AddEntry(div_z_2, "Jewel (Bot 60%) / Pythia8", "lep")
lg2.SetBorderSize(0)
lg2.SetTextSize(0.09)
lg2.Draw("same")

div_z_2.Draw("same")
div_z_1.Draw("same")


c1.SaveAs("zg_stack.pdf")


#############################################
# delta stack
#############################################

c1 = TCanvas("c1", "c1", 600, 800)

# upper panel

c1.cd()
p1 = TPad("p1", "p1", 0., 0.4, 1., 1.)
p1.SetTopMargin(0.02)
p1.SetBottomMargin(0)
p1.SetLeftMargin(0.18)
p1.SetRightMargin(0.02)
p1.Draw()
p1.cd()
# p1.SetLogy()

title = ";;#frac{1}{N} #frac{dN}{dR_{g}}"
frame1 = DrawFrame(0.08, 0.42, 0.10, 11.9, title, False)
style_upper_frame(frame1)


Tl = TLatex()
if mult == 1700:
    s1 = "Mixed with Mid-central BG"
else:
    s1 = "Mixed with Most-central BG"
s2 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s3 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"

Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.DrawLatex(0.22,0.90, s1)
Tl.DrawLatex(0.22,0.835, s2)
Tl.DrawLatex(0.22,0.755, s3)

# legend

lg = TLegend(0.50, 0.5, 0.88, 0.72)
lg.AddEntry(h_delta_pythia, "Pythia8", "f")
# lg.AddEntry(h_delta_jewel_1, "Jewel (LSTM > {:0.3f})".format(threshold), "f")
# lg.AddEntry(h_delta_jewel_2, "Jewel (LSTM < {:0.3f})".format(threshold), "f")
lg.AddEntry(h_delta_jewel_1, "Jewel (Top 40%)", "f")
lg.AddEntry(h_delta_jewel_2, "Jewel (Bot 60%)", "f")

lg.SetBorderSize(0)
lg.SetTextSize(0.06)
lg.Draw("same")

h_delta_stack_jewel.Add(h_delta_jewel_2)
h_delta_stack_jewel.Add(h_delta_jewel_1)

h_delta_stack_jewel.Draw("hist same")
h_delta_pythia.Draw("hist same")

# lower panel

c1.cd()
p2 = TPad("p2", "p2", 0., 0., 1., 0.4)
p2.SetTopMargin(0)
p2.SetBottomMargin(0.25)
p2.SetLeftMargin(0.18)
p2.SetRightMargin(0.02)
p2.Draw()
p2.cd()
frame2 = DrawFrame(0.08, 0.42, -0.1, 1.9, ";R_{g};Ratio", False)
style_lower_frame(frame2)

lg2 = TLegend(0.3, 0.78, 0.85, 0.98)
# lg2.AddEntry(div_delta_1, "Jewel (LSTM > {:0.3f}) / Pythia8".format(threshold), "lep")
# lg2.AddEntry(div_delta_2, "Jewel (LSTM < {:0.3f}) / Pythia8".format(threshold), "lep")
lg2.AddEntry(div_delta_1, "Jewel (Top 40%) / Pythia8", "lep")
lg2.AddEntry(div_delta_2, "Jewel (Bot 60%) / Pythia8", "lep")
lg2.SetBorderSize(0)
lg2.SetTextSize(0.09)
lg2.Draw("same")

div_delta_2.Draw("same")
div_delta_1.Draw("same")


c1.SaveAs("rg_stack.pdf")

#############################################
# mg stack
#############################################

c1 = TCanvas("c1", "c1", 600, 800)

# upper panel

c1.cd()
p1 = TPad("p1", "p1", 0., 0.4, 1., 1.)
p1.SetTopMargin(0.02)
p1.SetBottomMargin(0)
p1.SetLeftMargin(0.18)
p1.SetRightMargin(0.02)
p1.Draw()
p1.cd()
# p1.SetLogy()

title = ";;#frac{1}{N} #frac{dN}{dm_{g}/p_{T,jet}}"
# frame1 = DrawFrame(0., 0.26, 4e-3, 5e6, title, False);
frame1 = DrawFrame(0.01, 0.26, 4e-3, 17.9, title, False)
style_upper_frame(frame1)



Tl = TLatex()
if mult == 1700:
    s1 = "Mixed with Mid-central BG"
else:
    s1 = "Mixed with Most-central BG"
s2 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s3 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"

Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.DrawLatex(0.22,0.90, s1)
Tl.DrawLatex(0.22,0.835, s2)
Tl.DrawLatex(0.22,0.755, s3)

# legend

lg = TLegend(0.50, 0.5, 0.88, 0.72)
lg.AddEntry(h_m_pythia, "Pythia8", "f")
# lg.AddEntry(h_m_jewel_1, "Jewel (LSTM > {:0.3f})".format(threshold), "f")
# lg.AddEntry(h_m_jewel_2, "Jewel (LSTM < {:0.3f})".format(threshold), "f")
lg.AddEntry(h_m_jewel_1, "Jewel (Top 40%)", "f")
lg.AddEntry(h_m_jewel_2, "Jewel (Bot 60%)", "f")

lg.SetBorderSize(0)
lg.SetTextSize(0.06)
lg.Draw("same")

h_m_stack_jewel.Add(h_m_jewel_2)
h_m_stack_jewel.Add(h_m_jewel_1)

h_m_stack_jewel.Draw("hist same")
h_m_pythia.Draw("hist same")

# lower panel

c1.cd()
p2 = TPad("p2", "p2", 0., 0., 1., 0.4)
p2.SetTopMargin(0)
p2.SetBottomMargin(0.25)
p2.SetLeftMargin(0.18)
p2.SetRightMargin(0.02)
p2.Draw()
p2.cd()
# p2.SetLogy()

frame2 = DrawFrame(0.01, 0.26, -0.1, 2.8, ";m_{g}/p_{T,jet};Ratio", False)
style_lower_frame(frame2)
frame2.GetXaxis().SetTitleSize(39)


lg2 = TLegend(0.3, 0.78, 0.85, 0.98)
# lg2.AddEntry(div_m_1, "Jewel (LSTM > {:0.3f}) / Pythia8".format(threshold), "lep")
# lg2.AddEntry(div_m_2, "Jewel (LSTM < {:0.3f}) / Pythia8".format(threshold), "lep")
lg2.AddEntry(div_m_1, "Jewel (Top 40%) / Pythia8", "lep")
lg2.AddEntry(div_m_2, "Jewel (Bot 60%) / Pythia8", "lep")
lg2.SetBorderSize(0)
lg2.SetTextSize(0.09)
lg2.Draw('same')


div_m_1.Draw("same")
div_m_2.Draw("same")


c1.SaveAs("mg_stack.pdf")



#############################################
# lund pythia
#############################################

c1 = TCanvas("c1", "c1", 800, 800)

title = ";ln(1/R_{g});ln(z_{g}R_{g})"
gPad.SetLeftMargin(0.12)
gPad.SetRightMargin(0.12)
gPad.SetBottomMargin(0.1)
gPad.SetTopMargin(0.02)
frame = DrawFrame(0.55, 2.7, -5.3, -0.8, title, False)
frame.GetYaxis().SetLabelFont(43)
frame.GetYaxis().SetTitleFont(43)
frame.GetYaxis().SetLabelSize(25)
frame.GetYaxis().SetTitleSize(30)
frame.GetYaxis().SetTitleOffset(1.3)

frame.GetXaxis().SetLabelFont(43)
frame.GetXaxis().SetTitleFont(43)
frame.GetXaxis().SetLabelSize(25)
frame.GetXaxis().SetTitleSize(30)
frame.GetXaxis().SetTitleOffset(1.1)

h2_pythia.SetMaximum(3.2)
h2_pythia.Draw("colz same")

Tl = TLatex()
# Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.SetTextAngle(-30)
s1 = "z_{g}=0.5"
s2 = "z_{g}=0.2"
s3 = "z_{g}=0.1"

Tl.DrawLatex(2.37, -3.05, s1)
Tl.DrawLatex(2.37, -3.95, s2)
Tl.DrawLatex(2.37, -4.65, s3)


Tl2 = TLatex()
# Tl.SetNDC()
Tl2.SetTextFont(43)
Tl2.SetTextSize(28)
Tl2.SetTextAngle(90)
s1 = "R_{g}=0.4"
s2 = "R_{g}=0.2"
s3 = "R_{g}=0.1"

Tl2.DrawLatex(0.86, -5.15, s1)
Tl2.DrawLatex(1.55, -5.15, s2)
Tl2.DrawLatex(2.25, -5.15, s3)

if mult == 1700:
    s1 = "Pythia8 + Mid-central BG"
else:
    s1 = "Pythia8 + Most-central BG"
s2 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s3 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"


Tl3 = TLatex()
Tl3.SetNDC()
Tl3.SetTextFont(43)
Tl3.SetTextSize(28)
Tl3.DrawLatex(0.33,0.92, s1)
Tl3.DrawLatex(0.33,0.875, s2)
Tl3.DrawLatex(0.33,0.82, s3)

line1 = TLine(-log(0.4),log(0.4*0.5),-log(0.1),log(0.1*0.5))
line1.SetLineStyle(2)
line1.SetLineWidth(2)
line1.SetLineColor(2)
line1.Draw("same")

line2 = TLine(-log(0.4),log(0.4*0.1),-log(0.1),log(0.1*0.1))
line2.SetLineStyle(2)
line2.SetLineWidth(2)
line2.SetLineColor(2)
line2.Draw("same")

line3 = TLine(-log(0.4),log(0.4*0.2),-log(0.1),log(0.1*0.2))
line3.SetLineStyle(2)
line3.SetLineWidth(2)
line3.SetLineColor(2)
line3.Draw("same")

line4 = TLine(-log(0.4),-5.3,-log(0.4),log(0.4*0.5))
line4.SetLineStyle(2)
line4.SetLineWidth(2)
line4.SetLineColor(2)
line4.Draw("same")

line5 = TLine(-log(0.1),-5.3,-log(0.1),log(0.1*0.5))
line5.SetLineStyle(2)
line5.SetLineWidth(2)
line5.SetLineColor(2)
line5.Draw("same")

line6 = TLine(-log(0.2),-5.3,-log(0.2),log(0.2*0.5))
line6.SetLineStyle(2)
line6.SetLineWidth(2)
line6.SetLineColor(2)
line6.Draw("same")

c1.SaveAs("lund_pythia.pdf")

#############################################
# lund jewel
#############################################

c1 = TCanvas("c1", "c1", 800, 800)

title = ";ln(1/R_{g});ln(z_{g}R_{g})"
gPad.SetLeftMargin(0.12)
gPad.SetRightMargin(0.12)
gPad.SetBottomMargin(0.1)
gPad.SetTopMargin(0.02)
frame = DrawFrame(0.55, 2.7, -5.3, -0.8, title, False)
frame.GetYaxis().SetLabelFont(43)
frame.GetYaxis().SetTitleFont(43)
frame.GetYaxis().SetLabelSize(25)
frame.GetYaxis().SetTitleSize(30)
frame.GetYaxis().SetTitleOffset(1.3)

frame.GetXaxis().SetLabelFont(43)
frame.GetXaxis().SetTitleFont(43)
frame.GetXaxis().SetLabelSize(25)
frame.GetXaxis().SetTitleSize(30)
frame.GetXaxis().SetTitleOffset(1.1)

h2_jewel.SetMaximum(3.2)
h2_jewel.Draw("colz same")

Tl = TLatex()
# Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.SetTextAngle(-30)
s1 = "z_{g}=0.5"
s2 = "z_{g}=0.2"
s3 = "z_{g}=0.1"

Tl.DrawLatex(2.37, -3.05, s1)
Tl.DrawLatex(2.37, -3.95, s2)
Tl.DrawLatex(2.37, -4.65, s3)


Tl2 = TLatex()
# Tl.SetNDC()
Tl2.SetTextFont(43)
Tl2.SetTextSize(28)
Tl2.SetTextAngle(90)
s1 = "R_{g}=0.4"
s2 = "R_{g}=0.2"
s3 = "R_{g}=0.1"

Tl2.DrawLatex(0.86, -5.15, s1)
Tl2.DrawLatex(1.55, -5.15, s2)
Tl2.DrawLatex(2.25, -5.15, s3)

if mult == 1700:
    s1 = "Jewel + Mid-central BG"
else:
    s1 = "Jewel + Most-central BG"
s2 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s3 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"


Tl3 = TLatex()
Tl3.SetNDC()
Tl3.SetTextFont(43)
Tl3.SetTextSize(28)
Tl3.DrawLatex(0.33,0.92, s1)
Tl3.DrawLatex(0.33,0.875, s2)
Tl3.DrawLatex(0.33,0.82, s3)

line1 = TLine(-log(0.4),log(0.4*0.5),-log(0.1),log(0.1*0.5))
line1.SetLineStyle(2)
line1.SetLineWidth(2)
line1.SetLineColor(2)
line1.Draw("same")

line2 = TLine(-log(0.4),log(0.4*0.1),-log(0.1),log(0.1*0.1))
line2.SetLineStyle(2)
line2.SetLineWidth(2)
line2.SetLineColor(2)
line2.Draw("same")

line3 = TLine(-log(0.4),log(0.4*0.2),-log(0.1),log(0.1*0.2))
line3.SetLineStyle(2)
line3.SetLineWidth(2)
line3.SetLineColor(2)
line3.Draw("same")

line4 = TLine(-log(0.4),-5.3,-log(0.4),log(0.4*0.5))
line4.SetLineStyle(2)
line4.SetLineWidth(2)
line4.SetLineColor(2)
line4.Draw("same")

line5 = TLine(-log(0.1),-5.3,-log(0.1),log(0.1*0.5))
line5.SetLineStyle(2)
line5.SetLineWidth(2)
line5.SetLineColor(2)
line5.Draw("same")

line6 = TLine(-log(0.2),-5.3,-log(0.2),log(0.2*0.5))
line6.SetLineStyle(2)
line6.SetLineWidth(2)
line6.SetLineColor(2)
line6.Draw("same")

c1.SaveAs("lund_jewel.pdf")


#############################################
# lund comp 1
#############################################

c1 = TCanvas("c1", "c1", 800, 800)

title = ";ln(1/R_{g});ln(z_{g}R_{g})"
gPad.SetLeftMargin(0.12)
gPad.SetRightMargin(0.12)
gPad.SetBottomMargin(0.1)
gPad.SetTopMargin(0.02)
frame = DrawFrame(0.55, 2.7, -5.3, -0.8, title, False)
frame.GetYaxis().SetLabelFont(43)
frame.GetYaxis().SetTitleFont(43)
frame.GetYaxis().SetLabelSize(25)
frame.GetYaxis().SetTitleSize(30)
frame.GetYaxis().SetTitleOffset(1.3)

frame.GetXaxis().SetLabelFont(43)
frame.GetXaxis().SetTitleFont(43)
frame.GetXaxis().SetLabelSize(25)
frame.GetXaxis().SetTitleSize(30)
frame.GetXaxis().SetTitleOffset(1.1)


th = h2_jewel_1.Clone()
th.Divide(h2_pythia_rebin)

th.SetMaximum(2)
th.SetMinimum(0)
th.Draw("colz same")


Tl = TLatex()
# Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.SetTextAngle(-30)
s1 = "z_{g}=0.5"
s2 = "z_{g}=0.2"
s3 = "z_{g}=0.1"

Tl.DrawLatex(2.37, -3.05, s1)
Tl.DrawLatex(2.37, -3.95, s2)
Tl.DrawLatex(2.37, -4.65, s3)


Tl2 = TLatex()
# Tl.SetNDC()
Tl2.SetTextFont(43)
Tl2.SetTextSize(28)
Tl2.SetTextAngle(90)
s1 = "R_{g}=0.4"
s2 = "R_{g}=0.2"
s3 = "R_{g}=0.1"

Tl2.DrawLatex(0.86, -5.15, s1)
Tl2.DrawLatex(1.55, -5.15, s2)
Tl2.DrawLatex(2.25, -5.15, s3)

# s1 = "Ratio: Jewel (LSTM > %0.3f) / Pythia8" % threshold
s1 = "Ratio: Jewel (Top 40%) / Pythia8"
if mult == 1700:
    s2 = "Mixed with Mid-central BG"
else:
    s2 = "Mixed with Most-central BG"
s3 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s4 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"


Tl3 = TLatex()
Tl3.SetNDC()
Tl3.SetTextFont(43)
Tl3.SetTextSize(27)
Tl3.DrawLatex(0.33,0.92, s1)
Tl3.DrawLatex(0.33,0.88, s2)
Tl3.DrawLatex(0.33,0.845, s3)
Tl3.DrawLatex(0.33,0.80, s4)


line1 = TLine(-log(0.4),log(0.4*0.5),-log(0.1),log(0.1*0.5))
line1.SetLineStyle(2)
line1.SetLineWidth(2)
line1.SetLineColor(2)
line1.Draw("same")

line2 = TLine(-log(0.4),log(0.4*0.1),-log(0.1),log(0.1*0.1))
line2.SetLineStyle(2)
line2.SetLineWidth(2)
line2.SetLineColor(2)
line2.Draw("same")

line3 = TLine(-log(0.4),log(0.4*0.2),-log(0.1),log(0.1*0.2))
line3.SetLineStyle(2)
line3.SetLineWidth(2)
line3.SetLineColor(2)
line3.Draw("same")

line4 = TLine(-log(0.4),-5.3,-log(0.4),log(0.4*0.5))
line4.SetLineStyle(2)
line4.SetLineWidth(2)
line4.SetLineColor(2)
line4.Draw("same")

line5 = TLine(-log(0.1),-5.3,-log(0.1),log(0.1*0.5))
line5.SetLineStyle(2)
line5.SetLineWidth(2)
line5.SetLineColor(2)
line5.Draw("same")

line6 = TLine(-log(0.2),-5.3,-log(0.2),log(0.2*0.5))
line6.SetLineStyle(2)
line6.SetLineWidth(2)
line6.SetLineColor(2)
line6.Draw("same")


c1.SaveAs("lund_comp_1.pdf")


#############################################
# lund comp 2
#############################################

c1 = TCanvas("c1", "c1", 800, 800)

title = ";ln(1/R_{g});ln(z_{g}R_{g})"
gPad.SetLeftMargin(0.12)
gPad.SetRightMargin(0.12)
gPad.SetBottomMargin(0.1)
gPad.SetTopMargin(0.02)
frame = DrawFrame(0.55, 2.7, -5.3, -0.8, title, False)
frame.GetYaxis().SetLabelFont(43)
frame.GetYaxis().SetTitleFont(43)
frame.GetYaxis().SetLabelSize(25)
frame.GetYaxis().SetTitleSize(30)
frame.GetYaxis().SetTitleOffset(1.3)

frame.GetXaxis().SetLabelFont(43)
frame.GetXaxis().SetTitleFont(43)
frame.GetXaxis().SetLabelSize(25)
frame.GetXaxis().SetTitleSize(30)
frame.GetXaxis().SetTitleOffset(1.1)


th = h2_jewel_2.Clone()
th.Divide(h2_pythia_rebin)

th.SetMaximum(2)
th.SetMinimum(0)
th.Draw("colz same")


Tl = TLatex()
# Tl.SetNDC()
Tl.SetTextFont(43)
Tl.SetTextSize(28)
Tl.SetTextAngle(-30)
s1 = "z_{g}=0.5"
s2 = "z_{g}=0.2"
s3 = "z_{g}=0.1"

Tl.DrawLatex(2.37, -3.05, s1)
Tl.DrawLatex(2.37, -3.95, s2)
Tl.DrawLatex(2.37, -4.65, s3)


Tl2 = TLatex()
# Tl.SetNDC()
Tl2.SetTextFont(43)
Tl2.SetTextSize(28)
Tl2.SetTextAngle(90)
s1 = "R_{g}=0.4"
s2 = "R_{g}=0.2"
s3 = "R_{g}=0.1"

Tl2.DrawLatex(0.86, -5.15, s1)
Tl2.DrawLatex(1.55, -5.15, s2)
Tl2.DrawLatex(2.25, -5.15, s3)

# s1 = "Ratio: Jewel (LSTM < %0.3f) / Pythia8" % threshold
s1 = "Ratio: Jewel (Bot 60%) / Pythia8"
if mult == 1700:
    s2 = "Mixed with Mid-central BG"
else:
    s2 = "Mixed with Most-central BG"
s3 = "anti-k_{T} R = 0.4, p_{T,jet}#in [%.0f,%.0f] GeV" % (jetpt_min, jetpt_max)
s4 = "Soft Drop z_{cut}=0.1 #beta=0., R_{g} > 0.1"


Tl3 = TLatex()
Tl3.SetNDC()
Tl3.SetTextFont(43)
Tl3.SetTextSize(27)
Tl3.DrawLatex(0.33,0.92, s1)
Tl3.DrawLatex(0.33,0.88, s2)
Tl3.DrawLatex(0.33,0.845, s3)
Tl3.DrawLatex(0.33,0.80, s4)



line1 = TLine(-log(0.4),log(0.4*0.5),-log(0.1),log(0.1*0.5))
line1.SetLineStyle(2)
line1.SetLineWidth(2)
line1.SetLineColor(2)
line1.Draw("same")

line2 = TLine(-log(0.4),log(0.4*0.1),-log(0.1),log(0.1*0.1))
line2.SetLineStyle(2)
line2.SetLineWidth(2)
line2.SetLineColor(2)
line2.Draw("same")

line3 = TLine(-log(0.4),log(0.4*0.2),-log(0.1),log(0.1*0.2))
line3.SetLineStyle(2)
line3.SetLineWidth(2)
line3.SetLineColor(2)
line3.Draw("same")

line4 = TLine(-log(0.4),-5.3,-log(0.4),log(0.4*0.5))
line4.SetLineStyle(2)
line4.SetLineWidth(2)
line4.SetLineColor(2)
line4.Draw("same")

line5 = TLine(-log(0.1),-5.3,-log(0.1),log(0.1*0.5))
line5.SetLineStyle(2)
line5.SetLineWidth(2)
line5.SetLineColor(2)
line5.Draw("same")

line6 = TLine(-log(0.2),-5.3,-log(0.2),log(0.2*0.5))
line6.SetLineStyle(2)
line6.SetLineWidth(2)
line6.SetLineColor(2)
line6.Draw("same")


c1.SaveAs("lund_comp_2.pdf")
