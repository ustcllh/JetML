from src.JetML.Event import *
from src.JetTree.JetTree import *
import sys

from ROOT import TH1F, TH2F, TCanvas, gStyle
gStyle.SetOptStat(0000)

try:
    inputs = sys.argv[1:]
    print('Input Files' + ' '.join(inputs))
except:
    print('Error!')

th1 = TH2F('th1', 'Multiplicity;#eta;#phi', 100, -4, 4, 100, -.5, 6.5)

th2 = TH2F('th2', 'p_{t} Density;#eta;#phi', 100, -4, 4, 100, -.5, 6.5)

th3 = TH1F('th3', 'Multiplicity;#eta;#frac{1}{N_{event}}#frac{dN}{d#eta}', 100, -4, 4)

th4 = TH1F('th4', 'Pt Density;#eta;#frac{1}{N_{event}}#frac{dpt}{d#eta}', 100, -4, 4)

th5 = TH1F('th5', 'Multiplicity;#phi;#frac{1}{N_{event}}#frac{dN}{d#phi}', 100, -.5, 6.5)

th6 = TH1F('th6', 'Pt Density;#phi;#frac{1}{N_{event}}#frac{dpt}{d#phi}', 100, -.5, 6.5)

th1.GetXaxis().CenterTitle()
th1.GetYaxis().CenterTitle()
th1.GetYaxis().SetTitleOffset(1.2)
th2.GetXaxis().CenterTitle()
th2.GetYaxis().CenterTitle()
th2.GetYaxis().SetTitleOffset(1.2)
th3.GetXaxis().CenterTitle()
th3.GetYaxis().CenterTitle()
th3.GetYaxis().SetTitleOffset(1.5)
th4.GetXaxis().CenterTitle()
th4.GetYaxis().CenterTitle()
th4.GetYaxis().SetTitleOffset(1.5)
th5.GetXaxis().CenterTitle()
th5.GetYaxis().CenterTitle()
th5.GetYaxis().SetTitleOffset(1.5)
th6.GetXaxis().CenterTitle()
th6.GetYaxis().CenterTitle()
th6.GetYaxis().SetTitleOffset(1.5)

nevent = 0
for input in inputs:
    rd = Reader(input)
    dict_input, des_input = rd.next_event()
    nevent += 1

    while dict_input['0']:
        for p in dict_input['0']:
            th1.Fill(p.eta(), p.phi(), 1)
            th2.Fill(p.eta(), p.phi(), p.pt())

            th3.Fill(p.eta(), 1)
            th4.Fill(p.eta(), p.pt())

            th5.Fill(p.phi(), 1)
            th6.Fill(p.phi(), p.pt())

        dict_input, des_input = rd.next_event()
        nevent += 1


c1 = TCanvas('c1', 'c1', 600, 600)
c1.SetBottomMargin(0.1)
c1.SetLeftMargin(0.1)
c1.SetRightMargin(0.15)

c1.cd()
th1.Scale(1./nevent/0.08/0.07)
th1.Draw('colz')
c1.SaveAs('mult.pdf')

c1.cd()
th2.Scale(1./nevent/0.08/0.07)
th2.Draw('colz')
c1.SaveAs('ptdensity.pdf')

c2 = TCanvas('c2', 'c2', 600, 600)
c2.SetBottomMargin(0.1)
c2.SetLeftMargin(0.15)
c2.SetRightMargin(0.1)

c2.cd()
th3.Sumw2()
th3.Scale(1./nevent/0.08)
th3.Draw('E')
c2.SaveAs('mult_eta.pdf')

c2.cd()
th4.Sumw2()
th4.Scale(1./nevent/0.08)
th4.Draw('E')
c2.SaveAs('ptdensity_eta.pdf')

c2.cd()
th5.Sumw2()
th5.Scale(1./nevent/0.07)
th5.Draw('E')
c2.SaveAs('mult_phi.pdf')

c2.cd()
th6.Sumw2()
th6.Scale(1./nevent/0.07)
th6.Draw('E')
c2.SaveAs('ptdensity_phi.pdf')
