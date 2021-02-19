from ROOT import TFile, TTree, TCanvas, TVectorD
from ROOT import TH1F, TProfile, TF1, TGraphErrors
from ROOT import TLegend, TLine, TPad, TLatex
from ROOT import gStyle, gPad
import sys

gStyle.SetOptStat(1110)
gStyle.SetOptFit(1001)

# input
try:
    input = sys.argv[1]
    tree = sys.argv[2]
    # prefix = sys.argv[3]
    prefix = tree
    suffix = sys.argv[3]

except:
    input = './resolution.root'
    tree = 'icsjet'
    # prefix = 'icsjet'
    prefix = tree
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
    th = []
    tg_x = []
    tg_y = []
    tg_ex = []
    tg_ey = []
    tp = TProfile('tp', 'JES Profile;jetpt;JES', 20, 100, 500)

    for i in range(20):
        title = 'JES (%d<jetpt<%d); JES' % (100+20*i, 120+20*i)
        th.append(TH1F('', title, 200, 0.8, 1.2))
        tg_x.append(110+20*i)
        tg_ex.append(0)

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
        idx = int((jetpt-100)/20)
        if idx>0 and idx<20:
            th[idx].Fill(scale, 1)
        i+=1

    c = TCanvas('c', 'c', 600, 600)
    gStyle.SetOptStat(11)
    gStyle.SetOptFit(1001)
    for i in range(20):
        tf = TF1('tf', 'gaus', 0.5, 1.5)
        gaus_max = th[i].GetMaximum()
        gaus_mean = th[i].GetMean()
        gaus_dev = th[i].GetRMS()
        for j in range(10):
            tf.SetParameters(gaus_max, gaus_mean, gaus_dev)
            tf.SetRange(gaus_mean-2*gaus_dev, gaus_mean+2*gaus_dev)
            th[i].Fit('tf', 'QR')
            gaus_max = tf.GetParameter(0)
            gaus_mean = tf.GetParameter(1)
            gaus_dev = tf.GetParameter(2)
        tg_y.append(tf.GetParameter(1))
        tg_ey.append(tf.GetParError(1))
        c.cd()
        th[i].SetLineColor(4)
        th[i].SetLineWidth(2)
        th[i].Draw()
        tf.Draw("same")
        tl = TLine(100, 1, 500, 1)
        tl.SetLineStyle(7)
        tl.SetLineColor(2)
        tl.SetLineWidth(2)
        tl.Draw("same")
        # c.SaveAs(prefix+'_jes_diff_'+suffix+'%d.pdf'%(i))

    c = TCanvas('c', 'c', 600, 600)
    c.cd()
    c.SetBottomMargin(0.1)
    c.SetLeftMargin(0.15)

    tg = TGraphErrors()
    for i in range(20):
        tg.SetPoint(i, tg_x[i], tg_y[i])
        tg.SetPointError(i, 0, tg_ey[i])
    tg.SetMaximum(1.24)
    tg.SetMinimum(0.84)
    tg.SetLineColor(2)
    tg.SetLineWidth(2)
    tg.SetTitle('JES Profile;jetpt;JES')
    tg.GetYaxis().SetTitleSize(0.14/3)
    tg.GetYaxis().SetTitleOffset(1.2)
    tg.GetXaxis().CenterTitle()
    tg.GetYaxis().CenterTitle()
    tg.Draw('AP')

    tp.SetLineColor(4)
    tp.SetLineWidth(2)
    tp.Draw('same')

    tl = TLine(72, 1, 528, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(2)
    tl.Draw('same')

    lg = TLegend(0.2, 0.7, 0.5, 0.88)
    lg.AddEntry(tg, 'JES (Peak Fit)', 'le')
    lg.AddEntry(tp, 'JES (Profile)', 'le')
    lg.Draw('same')

    c.SaveAs(prefix+'_jes_diff_'+suffix+'.pdf')

def DrawJESProfile():
    tp = TProfile('tp', 'JES Profile;jetpt;JES', 20, 100, 500)
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
    c.SaveAs(prefix+'_jes_profile_'+suffix+'.pdf')

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
    # fr1 = p1.DrawFrame(-0.01, 0.5, 0.51, 2.1e1,"")

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
    lg.AddEntry(h1, 'jet', 'le')
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
    title_y = '#frac{' + tree + '}{jet}'
    fr2.GetYaxis().SetTitle(title_y)
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
    fr1 = p1.DrawFrame(-0.01, 2e-1, 0.41, 3e2,"")

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
    lg.AddEntry(h1, 'jet', 'le')
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
    title_y = '#frac{' + tree + '}{jet}'
    fr2.GetYaxis().SetTitle(title_y)
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
    lg.AddEntry(h1, 'jet', 'le')
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
    title_y = '#frac{' + tree + '}{jet}'
    fr2.GetYaxis().SetTitle(title_y)
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.01, 1, 20.1, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_depth_'+suffix+'.pdf')

def Draw_depthg():
    h1 = TH1F('h1', 'depth', 20, 0., 20.)
    h2 = TH1F('h2', 'depth', 20, 0., 20.)
    i=0
    bin_size = 20./20.
    njet=tr.GetEntriesFast()
    while i<njet:
        tr.GetEntry(i)
        if tr.depthg==0:
            i+=1
            continue
        h1.Fill(tr.depthg)
        h2.Fill(tr.depthgm)
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
    lg.AddEntry(h1, 'jet', 'le')
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
    fr2.GetXaxis().SetTitle("depthg")
    fr2.GetXaxis().CenterTitle()

    fr2.GetYaxis().SetNdivisions(8)
    fr2.GetYaxis().SetLabelSize(0.14)
    fr2.GetYaxis().SetTitleSize(0.14)
    fr2.GetYaxis().SetTitleOffset(0.4)
    title_y = '#frac{' + tree + '}{jet}'
    fr2.GetYaxis().SetTitle(title_y)
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.01, 1, 20.1, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_depthg_'+suffix+'.pdf')

def Draw_constituent_pt():
    h1 = TH1F('h1', '', 200, 0, 50)
    h2 = TH1F('h2', '', 200, 0, 50)
    i=0
    bin_size = 50./200
    njet=tr.GetEntriesFast()
    while i<njet:
        tr.GetEntry(i)
        for id in range(tr.cn):
            h1.Fill(tr.cpt[id])
        for id in range(tr.cnm):
            h2.Fill(tr.cptm[id])
        i+=1
    c = TCanvas('c', 'c', 600, 600)
    gStyle.SetOptStat(0000)

    p1 = TPad("p1", "p1", 0, 0.25, 1., 1.)
    p1.SetLogy(1)
    p1.SetBottomMargin(0)
    p1.SetLeftMargin(0.15)
    p1.Draw()
    p1.cd()
    fr1 = p1.DrawFrame(-0.1, 2e-5, 20.1, 3e5,"")

    fr1.GetXaxis().SetTitleSize(0.15/3)
    fr1.GetXaxis().SetLabelSize(0.15/3)

    fr1.GetYaxis().SetTitleSize(0.14/3)
    fr1.GetYaxis().SetTitleOffset(1.2)
    fr1.GetYaxis().SetLabelSize(0.14/3)
    fr1.GetYaxis().SetTitle("#frac{1}{N_{jet}} #frac{dN}{dcpt}")
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
    lg.AddEntry(h1, 'jet', 'le')
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

    fr2 = p2.DrawFrame(-0.1,0.42,20.1,1.24,"")

    fr2.GetXaxis().SetLabelSize(0.15)
    fr2.GetXaxis().SetTitleSize(0.15)
    fr2.GetXaxis().SetTitle("cpt")
    fr2.GetXaxis().CenterTitle()

    fr2.GetYaxis().SetNdivisions(8)
    fr2.GetYaxis().SetLabelSize(0.12)
    fr2.GetYaxis().SetTitleSize(0.12)
    fr2.GetYaxis().SetTitleOffset(0.4)
    title_y = '#frac{' + tree + '}{jet}'
    fr2.GetYaxis().SetTitle(title_y)
    fr2.GetYaxis().CenterTitle()

    h3 = h2.Clone()
    h3.Sumw2()
    h3.Divide(h1)
    h3.Draw("same")

    tl = TLine(-0.1, 1, 20.1, 1)
    tl.SetLineStyle(7)
    # tl.SetLineColor(2)
    tl.SetLineWidth(1)
    tl.Draw("same")

    c.SaveAs(prefix+'_cpt_'+suffix+'.pdf')

# DrawJES()
# DrawJESProfile()
DrawJESDiff()
Draw_z()
Draw_delta()
# Draw_depth()
# Draw_depthg()
Draw_constituent_pt()
