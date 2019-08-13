#include <iostream>
#include <string>

#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TH1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TRoot.h"


int main(){
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  TString ifn_pythia = "../res/jetml_pythia.root";
  TFile* f_pythia = new TFile(ifn_pythia, "READ");
  TTree* jet_pythia_tr= (TTree*) f_pythia->Get("jet_groomed_binary_tree");
  int nEntries_pythia = jet_pythia_tr->GetEntries();

  TString ifn = "../res/jetml.root";
  TFile* f = new TFile(ifn, "READ");
  TTree* jet_tr= (TTree*) f->Get("jet_groomed_binary_tree");
  int nEntries = jet_tr->GetEntries();
  std::cout<<nEntries<<std::endl;

  std::string zg_title = "";
  std::string zg0_title = "z with lstm<0.2";
  std::string zg1_title = "z with lstm>0.8";
  std::string zg_x = "z";
  std::string zg_y = "#frac{1}{N} #frac{dN}{dz}";

  std::string delta_x = "#DeltaR_{12}";
  std::string delta_y = "#frac{1}{N} #frac{dN}{d#DeltaR_{12}}";

  // zg
  TH1F* zg_h = new TH1F("zg_h", Form("%s;%s;%s", zg_title.c_str(), zg_x.c_str(), zg_y.c_str()), 10, 0, 0.5);
  TH1F* zg0_h = new TH1F("zg0_h", Form("%s;%s;%s", zg0_title.c_str(), zg_x.c_str(), zg_y.c_str()), 10, 0, 0.5);
  TH1F* zg1_h = new TH1F("zg1_h", Form("%s;%s;%s", zg1_title.c_str(), zg_x.c_str(), zg_y.c_str()), 10, 0, 0.5);
  TH1F* zg_pythia_h = new TH1F("zg_pythia_h", Form("%s;%s;%s", "zg1_title.c_str()", zg_x.c_str(), zg_y.c_str()), 10, 0, 0.5);

  // delta
  TH1F* delta_h = new TH1F("delta_h", Form("%s;%s;%s", "", delta_x.c_str(), delta_y.c_str()), 9, 0, 0.45);
  TH1F* delta0_h = new TH1F("delta0_h", Form("%s;%s;%s", "", delta_x.c_str(), delta_y.c_str()), 9, 0, 0.45);
  TH1F* delta1_h = new TH1F("delta1_h", Form("%s;%s;%s", "", delta_x.c_str(), delta_y.c_str()), 9, 0, 0.45);
  TH1F* delta_pythia_h = new TH1F("delta_pythia_h", Form("%s;%s;%s", "", delta_x.c_str(), delta_y.c_str()), 9, 0, 0.45);


  float lnm, lndelta, lnz;
  float lstm;
  jet_tr->SetBranchAddress("lnm", &lnm);
  jet_tr->SetBranchAddress("lndelta", &lndelta);
  jet_tr->SetBranchAddress("lnz", &lnz);
  jet_tr->SetBranchAddress("lstm", &lstm);

  jet_pythia_tr->SetBranchAddress("lnm", &lnm);
  jet_pythia_tr->SetBranchAddress("lndelta", &lndelta);
  jet_pythia_tr->SetBranchAddress("lnz", &lnz);
  jet_pythia_tr->SetBranchAddress("lstm", &lstm);

  for(int i=0; i<nEntries_pythia; i++){
    jet_pythia_tr->GetEntry(i);
    float zg = TMath::Exp(lnz);
    float delta = TMath::Exp(lndelta);
    zg_pythia_h->Fill(zg);
    delta_pythia_h->Fill(delta);
  }

  for(int i=0; i<nEntries; i++){
    jet_tr->GetEntry(i);
    float zg = TMath::Exp(lnz);
    float delta = TMath::Exp(lndelta);
    zg_h->Fill(zg);
    delta_h->Fill(delta);
    if(lstm<0.2){
      zg0_h->Fill(zg);
      delta0_h->Fill(delta);
    }
    if(lstm>0.8){
      zg1_h->Fill(zg);
      delta1_h->Fill(delta);
    }
  }

  TCanvas* c = new TCanvas("c", "c", 600, 600);
  c->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  // TH1F * frame = c->DrawFrame(0,0.5,-1, 11);
  // gPad->SetLogy();
  // frame->Draw();
  int red_i = TColor::GetFreeColorIndex();
  TColor *red = new TColor(red_i, 255./255, 0./255, 0./255);

  int green_i = TColor::GetFreeColorIndex();
  TColor *green = new TColor(green_i, 0./255, 255./255, 0./255);

  int blue_i = TColor::GetFreeColorIndex();
  TColor *blue = new TColor(blue_i, 0./255, 0./255, 255./255);


  zg_h->Sumw2();
  zg_h->Scale(1./zg_h->Integral()/0.05);
  zg_h->GetYaxis()->SetRangeUser(-0.5, 11);
  zg_h->GetXaxis()->SetTitleOffset(1.0);
  zg_h->GetYaxis()->SetTitleOffset(1.1);
  zg_h->GetXaxis()->SetTitleSize(0.05);
  zg_h->GetYaxis()->SetTitleSize(0.05);
  zg_h->GetYaxis()->CenterTitle();
  zg_h->GetXaxis()->CenterTitle();


  zg_h->SetMarkerSize(1.3);
  zg_h->SetMarkerStyle(21);
  zg_h->SetMarkerColor(red_i);
  zg_h->SetLineColor(red_i);
  zg_h->SetFillStyle(1001);
  zg_h->SetFillColorAlpha(red_i, 0.4);
  zg_h->Draw("E2");



  zg0_h->Sumw2();
  zg0_h->Scale(1./zg0_h->Integral()/0.05);
  // zg0_h->GetYaxis()->SetRangeUser(-0.5, 11);
  zg0_h->SetMarkerSize(1.3);
  zg0_h->SetMarkerStyle(33);
  zg0_h->SetMarkerColor(green_i);
  zg0_h->SetLineColor(green_i);
  zg0_h->SetFillStyle(1001);
  zg0_h->SetFillColorAlpha(green_i, 0.4);
  zg0_h->Draw("E2 same");


  zg1_h->Sumw2();
  zg1_h->Scale(1./zg1_h->Integral()/0.05);
  // zg1_h->GetYaxis()->SetRangeUser(-0.5, 11);
  zg1_h->SetMarkerSize(1.3);
  zg1_h->SetMarkerStyle(34);
  zg1_h->SetMarkerColor(blue_i);
  zg1_h->SetLineColor(blue_i);
  zg1_h->SetFillStyle(1001);
  zg1_h->SetFillColorAlpha(blue_i, 0.4);
  zg1_h->Draw("E2 same");

  zg_pythia_h->Sumw2();
  zg_pythia_h->Scale(1./zg_pythia_h->Integral()/0.05);
  // zg1_h->GetYaxis()->SetRangeUser(-0.5, 11);
  zg_pythia_h->SetMarkerSize(1.3);
  zg_pythia_h->SetMarkerStyle(20);
  zg_pythia_h->SetMarkerColor(kBlack);
  zg_pythia_h->SetLineColor(kBlack);
  zg_pythia_h->SetFillStyle(1001);
  zg_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  zg_pythia_h->Draw("E2 same");


  TLegend* lg = new TLegend(0.35, 0.5, 0.88, 0.65);
  // zg_h->SetMarkerSize(0.1);
  // zg0_h->SetMarkerSize(0.2);
  // zg1_h->SetMarkerSize(0.3);
  lg->AddEntry(zg_pythia_h, "Pythia");
  lg->AddEntry(zg_h, "Jewel Reoil Off");
  lg->AddEntry(zg0_h, "Jewel Reoil Off lstm < 0.2");
  lg->AddEntry(zg1_h, "Jewel Reoil Off lstm > 0.8");

  lg->SetBorderSize(0);
  lg->SetTextSize(0.04);
  lg->Draw("same");


  TLatex latex;
  latex.SetTextSize(0.05);
  latex.SetTextAlign(13);
  latex.SetTextFont(42);
  latex.DrawLatex(0.03, 10.5, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  latex.DrawLatex(0.03, 9, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c->SaveAs("../plot/zg.pdf");

  TCanvas* c1 = new TCanvas("c1", "c1", 600, 600);
  c1->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  gPad->SetLogy();
  // TH1F * frame = c->DrawFrame(0,0.5,-1, 11);
  // gPad->SetLogy();
  // frame->Draw();

  delta_h->Sumw2();
  delta_h->Scale(1./delta_h->Integral()/0.05);
  delta_h->GetYaxis()->SetRangeUser(7E-3, 9E2);
  delta_h->GetXaxis()->SetTitleOffset(1.0);
  delta_h->GetYaxis()->SetTitleOffset(1.1);
  delta_h->GetXaxis()->SetTitleSize(0.05);
  delta_h->GetYaxis()->SetTitleSize(0.05);
  delta_h->GetYaxis()->CenterTitle();
  delta_h->GetXaxis()->CenterTitle();


  delta_h->SetMarkerSize(1.3);
  delta_h->SetMarkerStyle(21);
  delta_h->SetMarkerColor(kRed);
  delta_h->SetLineColor(kRed);
  delta_h->SetFillStyle(1001);
  delta_h->SetFillColorAlpha(kRed, 0.4);
  delta_h->Draw("E2");



  delta0_h->Sumw2();
  delta0_h->Scale(1./delta0_h->Integral()/0.05);
  // delta0_h->GetYaxis()->SetRangeUser(-0.5, 11);
  delta0_h->SetMarkerSize(1.3);
  delta0_h->SetMarkerStyle(33);
  delta0_h->SetMarkerColor(kGreen);
  delta0_h->SetLineColor(kGreen);
  delta0_h->SetFillStyle(1001);
  delta0_h->SetFillColorAlpha(kGreen, 0.4);
  delta0_h->Draw("E2 same");


  delta1_h->Sumw2();
  delta1_h->Scale(1./delta1_h->Integral()/0.05);
  // delta1_h->GetYaxis()->SetRangeUser(-0.5, 11);
  delta1_h->SetMarkerSize(1.3);
  delta1_h->SetMarkerStyle(34);
  delta1_h->SetMarkerColor(kBlue);
  delta1_h->SetLineColor(kBlue);
  delta1_h->SetFillStyle(1001);
  delta1_h->SetFillColorAlpha(kBlue, 0.4);
  delta1_h->Draw("E2 same");

  delta_pythia_h->Sumw2();
  delta_pythia_h->Scale(1./delta_pythia_h->Integral()/0.05);
  // delta1_h->GetYaxis()->SetRangeUser(-0.5, 11);
  delta_pythia_h->SetMarkerSize(1.3);
  delta_pythia_h->SetMarkerStyle(20);
  delta_pythia_h->SetMarkerColor(kBlack);
  delta_pythia_h->SetLineColor(kBlack);
  delta_pythia_h->SetFillStyle(1001);
  delta_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  delta_pythia_h->Draw("E2 same");


  TLegend* lg1 = new TLegend(0.35, 0.5, 0.88, 0.65);
  // delta_h->SetMarkerSize(0.1);
  // delta0_h->SetMarkerSize(0.2);
  // delta1_h->SetMarkerSize(0.3);
  lg1->AddEntry(delta_pythia_h, "Pythia");
  lg1->AddEntry(delta_h, "Jewel Reoil Off");
  lg1->AddEntry(delta0_h, "Jewel Reoil Off lstm < 0.2");
  lg1->AddEntry(delta1_h, "Jewel Reoil Off lstm > 0.8");

  lg1->SetBorderSize(0);
  lg1->SetTextSize(0.04);
  lg1->Draw("same");


  TLatex latex1;
  latex1.SetTextSize(0.05);
  latex1.SetTextAlign(13);
  latex1.SetTextFont(42);
  latex1.DrawLatex(0.03, 5E2, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  latex1.DrawLatex(0.03, 1.1E2, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c1->SaveAs("../plot/delta.pdf");
  return 0;
}
