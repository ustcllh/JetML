# plot utils
from ROOT import gPad

def DrawFrame(xmin, xmax, ymin, ymax, Title, setMargins):

    if setMargins:
      gPad.SetLeftMargin(0.2)
      gPad.SetRightMargin(0.05)
      gPad.SetBottomMargin(0.1)
      gPad.SetTopMargin(0.05)

    frame = gPad.DrawFrame(xmin,ymin,xmax,ymax)
    frame.SetTitle(Title)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetXaxis().SetTitleSize(0.06)
    frame.GetYaxis().SetTitleSize(0.06)
    frame.GetXaxis().SetTitleOffset(0.8)
    frame.GetYaxis().SetTitleOffset(1.3)
    frame.GetXaxis().CenterTitle()
    frame.GetYaxis().CenterTitle()
    gPad.SetTicks(1,1)
    return frame

def SetMarkerStyle(hist, style, color, alpha, size):
    hist.SetMarkerStyle(style)
    hist.SetMarkerColorAlpha(color, alpha)
    hist.SetMarkerSize(size)

def SetLineStyle(hist, style, color, alpha, width):
    hist.SetLineStyle(style)
    hist.SetLineColorAlpha(color, alpha)
    hist.SetLineWidth(width)

def SetFillStyle(hist, style, color, alpha):
    hist.SetFillStyle(style)
    hist.SetFillColorAlpha(color, alpha)