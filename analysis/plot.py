from ROOT import TFile, TTree, TCanvas
from ROOT import TH1F, TProfile, TLine
from ROOT import TLegend, TPad, TLatex
from ROOT import gStyle, gPad
import sys

gStyle.SetOptStat(1110)

# input
try:
    input = sys.argv[1]
    tree = sys.argv[2]
    prefix = sys.argv[3]
    suffix = sys.argv[4]

except:
    input = './resolution.root'
    tree = 'jetics'
    prefix = 'icsjet'
    suffix = 'default'

f = TFile(input, 'READ')
# tr = f.Get('jetcs')
# prefix = 'csjet_'
# prefix = 'CS Jet'
zcut = 0.1

tr = f.Get(tree)



def DrawJES():
    th = TH1F('th', 'Jet Energy Scale (JES)JES', 50, 0.5, 1.5)
    binsize = 1./50
    i=0
    max = tr.GetEntriesFast()
    while i<max:
        tr.GetEntry(i)
        if tr.depth==0:
            i+=1
            continue
        scale = tr.scale
        th.Fill(scale)
        i+=1
    c = TCanvas('c', 'c', 600, 600)
    c.cd()
    th.Scale(1./th.Integral()/binsize)
    th.SetLineColor(4)
    th.SetLineWidth(2)
    th.Draw()
    c.SaveAs(prefix+'_jes_'+suffix+'.pdf')

def DrawJESDiff():
    tp = TProfile('tp1', 'JES Profilejet pt', 20, 100, 500)
    i=0
    max = tr.GetEntriesFast()
    while i<max:
        tr.GetEntry(i)
        if tr.depth==0:
            i+=1
            continue
        scale = tr.scale
        jetpt = tr.jetpt
        tp.Fill(jetpt, scale, 1)
        i+=1

    c = TCanvas('c', 'c', 600, 600)
    c.cd()
    tp.SetMaximum(1.24)
    tp.SetMinimum(0.84)
    tp.SetLineColor(4)
    tp.SetLineWidth(2)

    tp.Draw()
    tl = TLine(100, 1, 500, 1)
    tl.SetLineStyle(7)
    tl.SetLineColor(2)
    tl.SetLineWidth(2)
    tl.Draw("same")
    c.SaveAs(prefix+'_jes_diff_'+suffix+'.pdf')

def Draw_z():
    h1 = TH1F('h1', 'zz', 50, 0, 0.5)
    h2 = TH1F('h2', 'zz', 50, 0, 0.5)
    i=0
    bin_size = 0.5/50
    njet=tr.GetEntriesFast()
    while i<njet:
        tr.GetEntry(i)
        if tr.depth==0:
            i+=1
            continue
        for id in range(tr.depth):
            if tr.z[id]>zcut:
                h1.Fill(tr.z[id])
        for id in range(tr.depthm):
            if tr.zm[id]>zcut:
                h2.Fill(tr.zm[id])
        i+=1
    c = TCanvas('c', 'c', 600, 600)
    gStyle.SetOptStat(0000)

    p1 = TPad("p1", "p1", 0, 0.25, 1., 1.)
    p1.SetBottomMargin(0)
    p1.SetLeftMargin(0.15)
    p1.SetLogy(1)
    p1.Draw()
    p1.cd()
    fr1 = p1.DrawFrame(-0.01, 6.5, 0.51, 2.1e2,"")

    fr1.GetXaxis().SetTitleSize(0.15/3)
    fr1.GetXaxis().SetLabelSize(0.15/3)


    fr1.GetYaxis().SetTitleSize(0.14/3)
    fr1.GetYaxis().SetTitleOffset(1.2)
    fr1.GetYaxis().SetLabelSize(0.14/3)
    fr1.GetYaxis().SetTitle("#frac{1}{N_{jet}} #frac{dN}{dz}")
    fr1.GetYaxis().CenterTitle()

    h1.Scale(1./njet/bin_size)
    h2.Scale(1./njet/bin_size)
    h1.SetLineColor(4)
    h1.SetLineWidth(2)
    h2.SetLineColor(2)
    h2.SetLineWidth(2)
    h1.Draw("same")
    h2.Draw("same")
    lg = TLegend(0.6, 0.7, 0.88, 0.88)
    lg.AddEntry(h1, 'Jet', 'le')
    lg.AddEntry(h2, prefix, 'le')
    lg.Draw('same')

    latex = TLatex()
    latex.SetTextSize(0.05)
    latex.SetTextAlign(13)
    latex.DrawLatex(.02,150,"SoftDrop z_{cut}=0.1 #beta=0.")
    latex.Draw("same")

    c.cd()
    p2 = TPad("p2", "p2", 0, 0, 1., 0.25)
    p2.SetTopMargin(0)
    p2.SetLeftMargin(0.15)
    p2.SetBottomMargin(0.3)
    # p2.SetGridy()
    p2.Draw()
    p2.cd()

    fr2 = p2.DrawFrame(-0.01,0.62,0.51,1.35,"")

    fr2.GetXaxis().SetLabelSize(0.15)
    fr2.GetXaxis().SetTitleSize(0.15)
    fr2.GetXaxis().SetTitle("z")
    fr2.GetXaxis().CenterTitle()

    fr2.GetYaxis().SetNdivisions(8)
    fr2.GetYaxis().SetLabelSize(0.14)
    fr2.GetYaxis().SetTitleSize(0.14)
    fr2.GetYaxis().SetTitleOffset(0.4)
    fr2.GetYaxis().SetTitle("#frac{icsJet}{Jet}")
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.01, 1, 0.51, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_z_'+suffix+'.pdf')

def Draw_delta():
    h1 = TH1F('h1', '#Delta#Delta', 40, 0, 0.4)
    h2 = TH1F('h2', '#Delta#Delta', 40, 0, 0.4)
    i=0
    bin_size = 0.4/40
    njet=tr.GetEntriesFast()
    while i<njet:
        tr.GetEntry(i)
        if tr.depth==0:
            i+=1
            continue
        for id in range(tr.depth):
            if tr.z[id]>zcut:
                h1.Fill(tr.delta[id])
        for id in range(tr.depthm):
            if tr.zm[id]>zcut:
                h2.Fill(tr.deltam[id])
        i+=1
    c = TCanvas('c', 'c', 600, 600)
    gStyle.SetOptStat(0000)

    p1 = TPad("p1", "p1", 0, 0.25, 1., 1.)
    p1.SetLogy(1)
    p1.SetBottomMargin(0)
    p1.SetLeftMargin(0.15)
    p1.Draw()
    p1.cd()
    fr1 = p1.DrawFrame(-0.01, 2., 0.41, 3e2,"")

    fr1.GetXaxis().SetTitleSize(0.15/3)
    fr1.GetXaxis().SetLabelSize(0.15/3)


    fr1.GetYaxis().SetTitleSize(0.14/3)
    fr1.GetYaxis().SetTitleOffset(1.2)
    fr1.GetYaxis().SetLabelSize(0.14/3)
    fr1.GetYaxis().SetTitle("#frac{1}{N_{jet}} #frac{dN}{d#Delta}")
    fr1.GetYaxis().CenterTitle()

    h1.Scale(1./njet/bin_size)
    h2.Scale(1./njet/bin_size)
    h1.SetLineColor(4)
    h1.SetLineWidth(2)
    h2.SetLineColor(2)
    h2.SetLineWidth(2)
    h1.Draw("same")
    h2.Draw("same")
    lg = TLegend(0.6, 0.7, 0.88, 0.88)
    lg.AddEntry(h1, 'Jet', 'le')
    lg.AddEntry(h2, prefix, 'le')
    lg.Draw('same')

    latex = TLatex()
    latex.SetTextSize(0.05)
    latex.SetTextAlign(13)
    latex.DrawLatex(.02,200,"SoftDrop z_{cut}=0.1 #beta=0.")
    latex.Draw("same")

    c.cd()
    p2 = TPad("p2", "p2", 0, 0, 1., 0.25)
    p2.SetTopMargin(0)
    p2.SetLeftMargin(0.15)
    p2.SetBottomMargin(0.3)

    # p2.SetGridy()
    p2.Draw()
    p2.cd()

    fr2 = p2.DrawFrame(-0.01,0.42,0.41,1.24,"")

    fr2.GetXaxis().SetLabelSize(0.15)
    fr2.GetXaxis().SetTitleSize(0.15)
    fr2.GetXaxis().SetTitle("#Delta")
    fr2.GetXaxis().CenterTitle()

    fr2.GetYaxis().SetNdivisions(8)
    fr2.GetYaxis().SetLabelSize(0.12)
    fr2.GetYaxis().SetTitleSize(0.12)
    fr2.GetYaxis().SetTitleOffset(0.4)
    fr2.GetYaxis().SetTitle("#frac{icsJet}{Jet}")
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.01, 1, 0.41, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_delta_'+suffix+'.pdf')

def Draw_depth():
    h1 = TH1F('h1', 'depth', 20, 0., 20.)
    h2 = TH1F('h2', 'depth', 20, 0., 20.)
    i=0
    bin_size = 20./20.
    njet=tr.GetEntriesFast()
    while i<njet:
        tr.GetEntry(i)
        if tr.depth==0:
            i+=1
            continue
        h1.Fill(tr.depth)
        h2.Fill(tr.depthm)
        i+=1
    c = TCanvas('c', 'c', 600, 600)
    gStyle.SetOptStat(0000)

    p1 = TPad("p1", "p1", 0, 0.25, 1., 1.)
    p1.SetBottomMargin(0)
    p1.SetLeftMargin(0.15)
    p1.Draw()
    p1.cd()
    fr1 = p1.DrawFrame(-0.01, 0, 20.1, 0.3,"")
    fr1.GetXaxis().SetTitleSize(0.15/3)
    fr1.GetXaxis().SetLabelSize(0.15/3)


    fr1.GetYaxis().SetTitleSize(0.14/3)
    fr1.GetYaxis().SetTitleOffset(1.2)
    fr1.GetYaxis().SetLabelSize(0.14/3)
    fr1.GetYaxis().SetTitle("#frac{1}{N_{jet}} #frac{dN}{dz}")
    fr1.GetYaxis().CenterTitle()

    h1.Scale(1./njet/bin_size)
    h2.Scale(1./njet/bin_size)
    h1.SetLineColor(4)
    h1.SetLineWidth(2)
    h2.SetLineColor(2)
    h2.SetLineWidth(2)
    h1.Draw("same")
    h2.Draw("same")
    lg = TLegend(0.6, 0.7, 0.88, 0.88)
    lg.AddEntry(h1, 'Jet', 'le')
    lg.AddEntry(h2, prefix, 'le')
    lg.Draw('same')

    c.cd()
    p2 = TPad("p2", "p2", 0, 0, 1., 0.25)
    p2.SetTopMargin(0)
    p2.SetLeftMargin(0.15)
    p2.SetBottomMargin(0.3)
    # p2.SetGridy()
    p2.Draw()
    p2.cd()

    fr2 = p2.DrawFrame(-0.01,0.2,20.1,2.,"")

    fr2.GetXaxis().SetLabelSize(0.15)
    fr2.GetXaxis().SetTitleSize(0.15)
    fr2.GetXaxis().SetTitle("depth")
    fr2.GetXaxis().CenterTitle()

    fr2.GetYaxis().SetNdivisions(8)
    fr2.GetYaxis().SetLabelSize(0.14)
    fr2.GetYaxis().SetTitleSize(0.14)
    fr2.GetYaxis().SetTitleOffset(0.4)
    fr2.GetYaxis().SetTitle("#frac{icsJet}{Jet}")
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.01, 1, 12.1, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_depth_'+suffix+'.pdf')

DrawJES()
DrawJESDiff()
Draw_z()
Draw_delta()
Draw_depth()
