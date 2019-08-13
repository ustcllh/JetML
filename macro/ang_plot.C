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
  TTree* jet_pythia_tr= (TTree*) f_pythia->Get("jet");
  int nEntries_pythia = jet_pythia_tr->GetEntries();

  TString ifn = "../res/jetml.root";
  TFile* f = new TFile(ifn, "READ");
  TTree* jet_tr= (TTree*) f->Get("jet");
  int nEntries = jet_tr->GetEntries();
  // std::cout<<nEntries<<std::endl;

  std::string thrust_title = "";
  // std::string thrust0_title = "thrust with lstm<0.2";
  // std::string thrust1_title = "thrust with lstm>0.8";
  std::string thrust_x = "thrust (#alpha=1 #beta=2)";
  std::string thrust_y = "#frac{1}{N} #frac{dN}{d thrust}";

  std::string width_x = "width (#alpha=1 #beta=1)";
  std::string width_y = "#frac{1}{N} #frac{dN}{d width}";

  std::string ptd_x = "ptd (#alpha=2 #beta=0)";
  std::string ptd_y = "#frac{1}{N} #frac{dN}{d ptd}";

  std::string m_x = "m";
  std::string m_y = "#frac{1}{N} #frac{dN}{d m}";

  // m
  int nbins_m = 50;
  float m_low = 0;
  float m_up = 50;
  TH1F* m_h = new TH1F("m_h", Form("%s;%s;%s", "m", m_x.c_str(), m_y.c_str()), nbins_m, m_low, m_up);
  TH1F* m0_h = new TH1F("m0_h", Form("%s;%s;%s", "m0", m_x.c_str(), m_y.c_str()), nbins_m, m_low, m_up);
  TH1F* m1_h = new TH1F("m1_h", Form("%s;%s;%s", "m1", m_x.c_str(), m_y.c_str()), nbins_m, m_low, m_up);
  TH1F* m_pythia_h = new TH1F("m_pythia_h", Form("%s;%s;%s", "m_pythia", m_x.c_str(), m_y.c_str()), nbins_m, m_low, m_up);

  // thrust
  int nbins_thrust = 10;
  float thrust_low = 0;
  float thrudt_up = 0.5;
  TH1F* thrust_h = new TH1F("thrust_h", Form("%s;%s;%s", "", thrust_x.c_str(), thrust_y.c_str()), nbins_thrust, thrust_low, thrudt_up);
  TH1F* thrust0_h = new TH1F("thrust0_h", Form("%s;%s;%s", "thrust0", thrust_x.c_str(), thrust_y.c_str()), nbins_thrust, thrust_low, thrudt_up);
  TH1F* thrust1_h = new TH1F("thrust1_h", Form("%s;%s;%s", "thrust1", thrust_x.c_str(), thrust_y.c_str()), nbins_thrust, thrust_low, thrudt_up);
  TH1F* thrust_pythia_h = new TH1F("thrust_pythia_h", Form("%s;%s;%s", "thrust_pythia", thrust_x.c_str(), thrust_y.c_str()), nbins_thrust, thrust_low, thrudt_up);

  // width
  int nbins_width = 10;
  float width_low = 0;
  float with_up = 2;
  TH1F* width_h = new TH1F("width_h", Form("%s;%s;%s", "", width_x.c_str(), width_y.c_str()), nbins_width, width_low, with_up);
  TH1F* width0_h = new TH1F("width0_h", Form("%s;%s;%s", "width0", width_x.c_str(), width_y.c_str()), nbins_width, width_low, with_up);
  TH1F* width1_h = new TH1F("width1_h", Form("%s;%s;%s", "width1", width_x.c_str(), width_y.c_str()), nbins_width, width_low, with_up);
  TH1F* width_pythia_h = new TH1F("width_pythia_h", Form("%s;%s;%s", "width_pythia", width_x.c_str(), width_y.c_str()), nbins_width, width_low, with_up);

  // ptd
  int nbins_ptd = 10;
  float ptd_low = 0;
  float ptd_up = 1;
  TH1F* ptd_h = new TH1F("ptd_h", Form("%s;%s;%s", "", ptd_x.c_str(), ptd_y.c_str()), nbins_ptd, ptd_low, ptd_up);
  TH1F* ptd0_h = new TH1F("ptd0_h", Form("%s;%s;%s", "ptd0", ptd_x.c_str(), ptd_y.c_str()), nbins_ptd, ptd_low, ptd_up);
  TH1F* ptd1_h = new TH1F("ptd1_h", Form("%s;%s;%s", "ptd1", ptd_x.c_str(), ptd_y.c_str()), nbins_ptd, ptd_low, ptd_up);
  TH1F* ptd_pythia_h = new TH1F("ptd_pythia_h", Form("%s;%s;%s", "ptd_pythia", ptd_x.c_str(), ptd_y.c_str()), nbins_ptd, ptd_low, ptd_up);

  float m, ang_mass, ang_width, ang_ptd;
  float lstm;
  jet_tr->SetBranchAddress("m", &m);
  jet_tr->SetBranchAddress("ang_mass", &ang_mass);
  jet_tr->SetBranchAddress("ang_width", &ang_width);
  jet_tr->SetBranchAddress("ang_ptd", &ang_ptd);
  jet_tr->SetBranchAddress("lstm", &lstm);

  jet_pythia_tr->SetBranchAddress("m", &m);
  jet_pythia_tr->SetBranchAddress("ang_mass", &ang_mass);
  jet_pythia_tr->SetBranchAddress("ang_width", &ang_width);
  jet_pythia_tr->SetBranchAddress("ang_ptd", &ang_ptd);
  jet_pythia_tr->SetBranchAddress("lstm", &lstm);

  for(int i=0; i<nEntries_pythia; i++){
    jet_pythia_tr->GetEntry(i);
    m_pythia_h->Fill(m);
    thrust_pythia_h->Fill(ang_mass);
    width_pythia_h->Fill(ang_width);
    ptd_pythia_h->Fill(ang_ptd);
  }

  for(int i=0; i<nEntries; i++){
    jet_tr->GetEntry(i);
    m_h->Fill(m);
    thrust_h->Fill(ang_mass);
    width_h->Fill(ang_width);
    ptd_h->Fill(ang_ptd);
    if(lstm<0.2){
      m0_h->Fill(m);
      thrust0_h->Fill(ang_mass);
      width0_h->Fill(ang_width);
      ptd0_h->Fill(ang_ptd);
    }
    if(lstm>0.8){
      m1_h->Fill(m);
      thrust1_h->Fill(ang_mass);
      width1_h->Fill(ang_width);
      ptd1_h->Fill(ang_ptd);
    }
  }

  TCanvas* c = new TCanvas("c", "c", 600, 600);
  c->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  gPad->SetLogy();

  thrust_h->Sumw2();
  thrust_h->Scale(1./thrust_h->Integral()/0.05);
  thrust_h->GetYaxis()->SetRangeUser(8E-3, 4E1);
  thrust_h->GetXaxis()->SetTitleOffset(1.0);
  thrust_h->GetYaxis()->SetTitleOffset(1.1);
  thrust_h->GetXaxis()->SetTitleSize(0.05);
  thrust_h->GetYaxis()->SetTitleSize(0.05);
  thrust_h->GetYaxis()->CenterTitle();
  thrust_h->GetXaxis()->CenterTitle();


  thrust_h->SetMarkerSize(1.3);
  thrust_h->SetMarkerStyle(21);
  thrust_h->SetMarkerColor(kRed);
  thrust_h->SetLineColor(kRed);
  thrust_h->SetFillStyle(1001);
  thrust_h->SetFillColorAlpha(kRed, 0.4);
  thrust_h->Draw("E2");



  thrust0_h->Sumw2();
  thrust0_h->Scale(1./thrust0_h->Integral()/0.05);
  thrust0_h->SetMarkerSize(1.3);
  thrust0_h->SetMarkerStyle(33);
  thrust0_h->SetMarkerColor(kGreen);
  thrust0_h->SetLineColor(kGreen);
  thrust0_h->SetFillStyle(1001);
  thrust0_h->SetFillColorAlpha(kGreen, 0.4);
  thrust0_h->Draw("E2 same");


  thrust1_h->Sumw2();
  thrust1_h->Scale(1./thrust1_h->Integral()/0.05);
  thrust1_h->SetMarkerSize(1.3);
  thrust1_h->SetMarkerStyle(34);
  thrust1_h->SetMarkerColor(kBlue);
  thrust1_h->SetLineColor(kBlue);
  thrust1_h->SetFillStyle(1001);
  thrust1_h->SetFillColorAlpha(kBlue, 0.4);
  thrust1_h->Draw("E2 same");

  thrust_pythia_h->Sumw2();
  thrust_pythia_h->Scale(1./thrust_pythia_h->Integral()/0.05);
  thrust_pythia_h->SetMarkerSize(1.3);
  thrust_pythia_h->SetMarkerStyle(20);
  thrust_pythia_h->SetMarkerColor(kBlack);
  thrust_pythia_h->SetLineColor(kBlack);
  thrust_pythia_h->SetFillStyle(1001);
  thrust_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  thrust_pythia_h->Draw("E2 same");


  TLegend* lg = new TLegend(0.35, 0.65, 0.88, 0.8);
  lg->AddEntry(thrust_pythia_h, "Pythia");
  lg->AddEntry(thrust_h, "Jewel Reoil Off");
  lg->AddEntry(thrust0_h, "Jewel Reoil Off lstm < 0.2");
  lg->AddEntry(thrust1_h, "Jewel Reoil Off lstm > 0.8");

  lg->SetBorderSize(0);
  lg->SetTextSize(0.04);
  lg->Draw("same");


  TLatex latex;
  latex.SetTextSize(0.05);
  latex.SetTextAlign(13);
  latex.SetTextFont(42);
  latex.DrawLatex(0.07, 25, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  // latex.DrawLatex(0.03, 9, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c->SaveAs("../plot/thrust.pdf");

  TCanvas* c1 = new TCanvas("c1", "c1", 600, 600);
  c1->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  gPad->SetLogy();


  width_h->Sumw2();
  width_h->Scale(1./width_h->Integral()/0.05);
  width_h->GetYaxis()->SetRangeUser(5E-2, 4E1);
  width_h->GetXaxis()->SetTitleOffset(1.0);
  width_h->GetYaxis()->SetTitleOffset(1.1);
  width_h->GetXaxis()->SetTitleSize(0.05);
  width_h->GetYaxis()->SetTitleSize(0.05);
  width_h->GetYaxis()->CenterTitle();
  width_h->GetXaxis()->CenterTitle();


  width_h->SetMarkerSize(1.3);
  width_h->SetMarkerStyle(21);
  width_h->SetMarkerColor(kRed);
  width_h->SetLineColor(kRed);
  width_h->SetFillStyle(1001);
  width_h->SetFillColorAlpha(kRed, 0.4);
  width_h->Draw("E2");



  width0_h->Sumw2();
  width0_h->Scale(1./width0_h->Integral()/0.05);
  width0_h->SetMarkerSize(1.3);
  width0_h->SetMarkerStyle(33);
  width0_h->SetMarkerColor(kGreen);
  width0_h->SetLineColor(kGreen);
  width0_h->SetFillStyle(1001);
  width0_h->SetFillColorAlpha(kGreen, 0.4);
  width0_h->Draw("E2 same");


  width1_h->Sumw2();
  width1_h->Scale(1./width1_h->Integral()/0.05);
  width1_h->SetMarkerSize(1.3);
  width1_h->SetMarkerStyle(34);
  width1_h->SetMarkerColor(kBlue);
  width1_h->SetLineColor(kBlue);
  width1_h->SetFillStyle(1001);
  width1_h->SetFillColorAlpha(kBlue, 0.4);
  width1_h->Draw("E2 same");

  width_pythia_h->Sumw2();
  width_pythia_h->Scale(1./width_pythia_h->Integral()/0.05);
  width_pythia_h->SetMarkerSize(1.3);
  width_pythia_h->SetMarkerStyle(20);
  width_pythia_h->SetMarkerColor(kBlack);
  width_pythia_h->SetLineColor(kBlack);
  width_pythia_h->SetFillStyle(1001);
  width_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  width_pythia_h->Draw("E2 same");


  TLegend* lg1 = new TLegend(0.35, 0.65, 0.88, 0.8);
  lg1->AddEntry(width_pythia_h, "Pythia");
  lg1->AddEntry(width_h, "Jewel Reoil Off");
  lg1->AddEntry(width0_h, "Jewel Reoil Off lstm < 0.2");
  lg1->AddEntry(width1_h, "Jewel Reoil Off lstm > 0.8");

  lg1->SetBorderSize(0);
  lg1->SetTextSize(0.04);
  lg1->Draw("same");


  TLatex latex1;
  latex1.SetTextSize(0.05);
  latex1.SetTextAlign(13);
  latex1.SetTextFont(42);
  latex1.DrawLatex(0.4, 30, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  // latex1.DrawLatex(0.03, 1.1E2, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c1->SaveAs("../plot/width.pdf");

  TCanvas* c2 = new TCanvas("c2", "c2", 600, 600);
  c2->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  // gPad->SetLogy();


  m_h->Sumw2();
  m_h->Scale(1./m_h->Integral()/0.05);
  m_h->GetYaxis()->SetRangeUser(0, 2.2);
  m_h->GetXaxis()->SetTitleOffset(1.0);
  m_h->GetYaxis()->SetTitleOffset(1.1);
  m_h->GetXaxis()->SetTitleSize(0.05);
  m_h->GetYaxis()->SetTitleSize(0.05);
  m_h->GetYaxis()->CenterTitle();
  m_h->GetXaxis()->CenterTitle();


  m_h->SetMarkerSize(1.3);
  m_h->SetMarkerStyle(21);
  m_h->SetMarkerColor(kRed);
  m_h->SetLineColor(kRed);
  m_h->SetFillStyle(1001);
  m_h->SetFillColorAlpha(kRed, 0.4);
  m_h->Draw("E2");


  m0_h->Sumw2();
  m0_h->Scale(1./m0_h->Integral()/0.05);
  m0_h->SetMarkerSize(1.3);
  m0_h->SetMarkerStyle(33);
  m0_h->SetMarkerColor(kGreen);
  m0_h->SetLineColor(kGreen);
  m0_h->SetFillStyle(1001);
  m0_h->SetFillColorAlpha(kGreen, 0.4);
  m0_h->Draw("E2 same");


  m1_h->Sumw2();
  m1_h->Scale(1./m1_h->Integral()/0.05);
  m1_h->SetMarkerSize(1.3);
  m1_h->SetMarkerStyle(34);
  m1_h->SetMarkerColor(kBlue);
  m1_h->SetLineColor(kBlue);
  m1_h->SetFillStyle(1001);
  m1_h->SetFillColorAlpha(kBlue, 0.4);
  m1_h->Draw("E2 same");

  m_pythia_h->Sumw2();
  m_pythia_h->Scale(1./m_pythia_h->Integral()/0.05);
  m_pythia_h->SetMarkerSize(1.3);
  m_pythia_h->SetMarkerStyle(20);
  m_pythia_h->SetMarkerColor(kBlack);
  m_pythia_h->SetLineColor(kBlack);
  m_pythia_h->SetFillStyle(1001);
  m_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  m_pythia_h->Draw("E2 same");


  TLegend* lg2 = new TLegend(0.35, 0.65, 0.88, 0.8);
  lg2->AddEntry(m_pythia_h, "Pythia");
  lg2->AddEntry(m_h, "Jewel Reoil Off");
  lg2->AddEntry(m0_h, "Jewel Reoil Off lstm < 0.2");
  lg2->AddEntry(m1_h, "Jewel Reoil Off lstm > 0.8");

  lg2->SetBorderSize(0);
  lg2->SetTextSize(0.04);
  lg2->Draw("same");


  TLatex latex2;
  latex2.SetTextSize(0.05);
  latex2.SetTextAlign(13);
  latex2.SetTextFont(42);
  latex2.DrawLatex(5, 2.1, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  // latex1.DrawLatex(0.03, 1.1E2, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c2->SaveAs("../plot/m.pdf");

  TCanvas* c3 = new TCanvas("c3", "c3", 600, 600);
  c3->cd();
  gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
  // gPad->SetLogy();


  ptd_h->Sumw2();
  ptd_h->Scale(1./ptd_h->Integral()/0.05);
  ptd_h->GetYaxis()->SetRangeUser(0, 14);
  ptd_h->GetXaxis()->SetTitleOffset(1.0);
  ptd_h->GetYaxis()->SetTitleOffset(1.1);
  ptd_h->GetXaxis()->SetTitleSize(0.05);
  ptd_h->GetYaxis()->SetTitleSize(0.05);
  ptd_h->GetYaxis()->CenterTitle();
  ptd_h->GetXaxis()->CenterTitle();


  ptd_h->SetMarkerSize(1.3);
  ptd_h->SetMarkerStyle(21);
  ptd_h->SetMarkerColor(kRed);
  ptd_h->SetLineColor(kRed);
  ptd_h->SetFillStyle(1001);
  ptd_h->SetFillColorAlpha(kRed, 0.4);
  ptd_h->Draw("E2");



  ptd0_h->Sumw2();
  ptd0_h->Scale(1./ptd0_h->Integral()/0.05);
  ptd0_h->SetMarkerSize(1.3);
  ptd0_h->SetMarkerStyle(33);
  ptd0_h->SetMarkerColor(kGreen);
  ptd0_h->SetLineColor(kGreen);
  ptd0_h->SetFillStyle(1001);
  ptd0_h->SetFillColorAlpha(kGreen, 0.4);
  ptd0_h->Draw("E2 same");


  ptd1_h->Sumw2();
  ptd1_h->Scale(1./ptd1_h->Integral()/0.05);
  ptd1_h->SetMarkerSize(1.3);
  ptd1_h->SetMarkerStyle(34);
  ptd1_h->SetMarkerColor(kBlue);
  ptd1_h->SetLineColor(kBlue);
  ptd1_h->SetFillStyle(1001);
  ptd1_h->SetFillColorAlpha(kBlue, 0.4);
  ptd1_h->Draw("E2 same");

  ptd_pythia_h->Sumw2();
  ptd_pythia_h->Scale(1./ptd_pythia_h->Integral()/0.05);
  ptd_pythia_h->SetMarkerSize(1.3);
  ptd_pythia_h->SetMarkerStyle(20);
  ptd_pythia_h->SetMarkerColor(kBlack);
  ptd_pythia_h->SetLineColor(kBlack);
  ptd_pythia_h->SetFillStyle(1001);
  ptd_pythia_h->SetFillColorAlpha(kBlack, 0.4);
  ptd_pythia_h->Draw("E2 same");


  TLegend* lg3 = new TLegend(0.35, 0.65, 0.88, 0.8);
  lg3->AddEntry(ptd_pythia_h, "Pythia");
  lg3->AddEntry(ptd_h, "Jewel Reoil Off");
  lg3->AddEntry(ptd0_h, "Jewel Reoil Off lstm < 0.2");
  lg3->AddEntry(ptd1_h, "Jewel Reoil Off lstm > 0.8");

  lg3->SetBorderSize(0);
  lg3->SetTextSize(0.04);
  lg3->Draw("same");


  TLatex latex3;
  latex3.SetTextSize(0.05);
  latex3.SetTextAlign(13);
  latex3.SetTextFont(42);
  latex3.DrawLatex(0.1, 13, "p_{T,jet} > 130 GeV, anti-k_{T}, R = 0.4");
  // latex1.DrawLatex(0.03, 1.1E2, "Soft Drop, z_{cut} = 0.5, #beta = 1.5");

  c3->SaveAs("../plot/ptd.pdf");
  return 0;
}
