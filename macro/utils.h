#include "TH1F.h"

TH1F* DrawFrame(double xmin, double xmax, double ymin, double ymax, TString Title, bool setMargins);
void SetMarkerStyle(TH1F* hist, int style, int color, double size);
void SetLineStyle(TH1F* hist, int style, int color,  double width);
void SetFillStyle(TH1F* hist, int style, int color, double alpha);



void SetMarkerStyle(TH1F* hist, int style, int color, double size){
  hist->SetMarkerStyle(style);
  hist->SetMarkerColor(color);
  hist->SetMarkerSize(size);
  return;
}

void SetLineStyle(TH1F* hist, int style, int color, double width){
  hist->SetLineStyle(style);
  hist->SetLineColor(color);
  hist->SetLineWidth(width);
  return;
}

void SetFillStyle(TH1F* hist, int style, int color, double alpha){
  hist->SetFillStyle(style);
  hist->SetFillColorAlpha(color, alpha);
}




TH1F* DrawFrame(double xmin, double xmax, double ymin, double ymax, TString Title, bool setMargins){

  if(setMargins) {
    gPad->SetLeftMargin(0.18);
    gPad->SetRightMargin(0.1);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.1);
  }

  TH1F* frame = gPad->DrawFrame(xmin,ymin,xmax,ymax);
  frame->SetTitle(Title.Data());
  frame->GetXaxis()->SetLabelSize(0.04);
  frame->GetYaxis()->SetLabelSize(0.04);
  frame->GetXaxis()->SetTitleSize(0.06);
  frame->GetYaxis()->SetTitleSize(0.06);
  frame->GetXaxis()->SetTitleOffset(0.8);
  frame->GetYaxis()->SetTitleOffset(1.3);
  frame->GetXaxis()->CenterTitle();
  frame->GetYaxis()->CenterTitle();

  gPad->SetTicks(1,1);

  return frame;
}
