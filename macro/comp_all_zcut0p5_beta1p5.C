#include "utils.h"


int comp_all_zcut0p5_beta1p5(){
  gStyle->SetOptStat(0);
  double zcut = 0.5;
  double beta = 1.5;
  double ptmin = 80;
  double rjet = 0.4;

  // TString cut = "delta>0 && z>0";
  // TString cut = "delta>0 && delta<0.2 && z>0";
  TString cut = "delta>0 && z>0";

  // pythia
  TString inf1 = "../results/ptmin80/pythia_zcut0p5_beta1p5.root";
  TFile* f1 = new TFile(inf1, "READ");
  TTree* tr_pythia = (TTree*) f1->Get("jet");

  tr_pythia->Draw("z>>h_z_pythia(10, 0, 0.5)", cut.Data(), "");
  TH1F* h_z_pythia = (TH1F*) gDirectory->Get("h_z_pythia");
  h_z_pythia->Sumw2();
  h_z_pythia->Scale(1./0.05/h_z_pythia->GetEntries());

  tr_pythia->Draw("delta>>h_delta_pythia(9, 0, 0.45)", cut.Data(), "");
  TH1F* h_delta_pythia = (TH1F*) gDirectory->Get("h_delta_pythia");
  h_delta_pythia->Sumw2();
  h_delta_pythia->Scale(1./0.05/h_delta_pythia->GetEntries());


  // jewel NR
  TString inf2 = "../results/ptmin80/jewel_NR_zcut0p5_beta1p5.root";
  TFile* f2 = new TFile(inf2, "READ");
  TTree* tr_NR = (TTree*) f2->Get("jet");

  tr_NR->Draw("z>>h_z_NR(10, 0, 0.5)", cut.Data(), "");
  TH1F* h_z_NR = (TH1F*) gDirectory->Get("h_z_NR");
  h_z_NR->Sumw2();
  h_z_NR->Scale(1./0.05/h_z_NR->GetEntries());

  tr_NR->Draw("delta>>h_delta_NR(9, 0, 0.45)", cut.Data(), "");
  TH1F* h_delta_NR = (TH1F*) gDirectory->Get("h_delta_NR");
  h_delta_NR->Sumw2();
  h_delta_NR->Scale(1./0.05/h_delta_NR->GetEntries());

  // jewel R
  TString inf3 = "../results/ptmin80/jewel_R_zcut0p5_beta1p5.root";
  TFile* f3 = new TFile(inf3, "READ");
  TTree* tr_R = (TTree*) f3->Get("jet");

  tr_R->Draw("z>>h_z_R(10, 0, 0.5)", cut.Data(), "");
  TH1F* h_z_R = (TH1F*) gDirectory->Get("h_z_R");
  h_z_R->Sumw2();
  h_z_R->Scale(1./0.05/h_z_R->GetEntries());

  tr_R->Draw("delta>>h_delta_R(9, 0, 0.45)", cut.Data(), "");
  TH1F* h_delta_R = (TH1F*) gDirectory->Get("h_delta_R");
  h_delta_R->Sumw2();
  h_delta_R->Scale(1./0.05/h_delta_R->GetEntries());

  // hybrid
  TString inf4 = "../results/ptmin80/hybrid_zcut0p5_beta1p5.root";
  TFile* f4 = new TFile(inf4, "READ");
  TTree* tr_hybrid = (TTree*) f4->Get("jet");

  tr_hybrid->Draw("z>>h_z_hybrid(10, 0, 0.5)", cut.Data(), "");
  TH1F* h_z_hybrid = (TH1F*) gDirectory->Get("h_z_hybrid");
  h_z_hybrid->Sumw2();
  h_z_hybrid->Scale(1./0.05/h_z_hybrid->GetEntries());

  tr_hybrid->Draw("delta>>h_delta_hybrid(9, 0, 0.45)", cut.Data(), "");
  TH1F* h_delta_hybrid = (TH1F*) gDirectory->Get("h_delta_hybrid");
  h_delta_hybrid->Sumw2();
  h_delta_hybrid->Scale(1./0.05/h_delta_hybrid->GetEntries());

  // division

  TH1F* div11 = (TH1F*) h_z_NR->Clone();
  div11->Sumw2();
  div11->Divide(h_z_pythia);

  TH1F* div12 = (TH1F*) h_z_R->Clone();
  div12->Sumw2();
  div12->Divide(h_z_pythia);

  TH1F* div13 = (TH1F*) h_z_hybrid->Clone();
  div13->Sumw2();
  div13->Divide(h_z_pythia);

  TH1F* div21 = (TH1F*) h_delta_NR->Clone();
  div21->Sumw2();
  div21->Divide(h_delta_pythia);

  TH1F* div22 = (TH1F*) h_delta_R->Clone();
  div22->Sumw2();
  div22->Divide(h_delta_pythia);

  TH1F* div23 = (TH1F*) h_delta_hybrid->Clone();
  div23->Sumw2();
  div23->Divide(h_delta_pythia);


  // Marker style color size
  // Line style color width
  // fill style color alpha
  // pythia 20 1 1.3, 1  1  1, 1001 1 0.4
  // NR 21 2 1.3, 1 2 1, 1001 2 0.4
  // R 33 4 1.3, 1 4 1, 1001 4 0.4
  // hybrid 34 3 1.3, 1 3 1, 1001 3 0.4

  TCanvas* c1 = new TCanvas("c1", "c1", 600, 800);
  c1->cd();

  TPad* p11 = new TPad("p11", "p11", 0., 0.3, 1., 1.);
  p11->SetBottomMargin(0);
  p11->SetLeftMargin(0.18);
  p11->Draw();
  p11->cd();
  TString title1 = ";; #frac{1}{N} #frac{dN}{dz}";
  TH1F* frame11 = DrawFrame(0., 0.5, -0.5, 11, title1, false);
  frame11->GetYaxis()->SetLabelFont(43);
  frame11->GetYaxis()->SetTitleFont(43);
  frame11->GetYaxis()->SetLabelSize(25);
  frame11->GetYaxis()->SetTitleSize(25);
  frame11->GetYaxis()->SetTitleOffset(2.);


  SetMarkerStyle(h_z_pythia, 20, 1, 1.3);
  SetLineStyle(h_z_pythia, 1, 1, 1);
  SetFillStyle(h_z_pythia, 1001, 1, 0.4);
  h_z_pythia->Draw("E2 same");

  SetMarkerStyle(h_z_NR, 21, 2, 1.3);
  SetLineStyle(h_z_NR, 1, 2, 1);
  SetFillStyle(h_z_NR, 1001, 2, 0.4);
  h_z_NR->Draw("E2 same");

  SetMarkerStyle(h_z_R, 33, 4, 1.3);
  SetLineStyle(h_z_R, 1, 4, 1);
  SetFillStyle(h_z_R, 1001, 4, 0.4);
  h_z_R->Draw("E2 same");

  SetMarkerStyle(h_z_hybrid, 34, kGreen+2, 1.3);
  SetLineStyle(h_z_hybrid, 1, 3, 1);
  SetFillStyle(h_z_hybrid, 1001, 3, 0.4);
  h_z_hybrid->Draw("E2 same");

  TLegend* lg1 = new TLegend(0.52, 0.3, 0.87, 0.5);
  lg1->AddEntry(h_z_pythia, "Pythia8");
  lg1->AddEntry(h_z_NR, "Jewel Recoil Off");
  lg1->AddEntry(h_z_R, "Jewel Recoil On");
  lg1->AddEntry(h_z_hybrid, "Hybrid");
  lg1->SetBorderSize(0);
  lg1->SetTextSize(0.04);
  lg1->Draw("same");

  TLatex Tl;
  Tl.SetTextFont(43);
  Tl.SetTextSize(30);
  Tl.DrawLatex(0.04,9.5,"p_{T,jet}> 80GeV, anti-k_{T}, R=0.4");
  Tl.DrawLatex(0.04,8,"Soft Drop z_{cut}=0.5, #beta=1.5");
  Tl.DrawLatex(0.04,6.5, "All Soft Drop");


  c1->cd();
  TPad* p12 = new TPad("", "", 0., 0., 1., 0.3);
  p12->SetLeftMargin(0.18);
  p12->SetTopMargin(0);
  p12->SetBottomMargin(0.22);
  p12->Draw();
  p12->cd();
  TH1F* frame12 = DrawFrame(0., 0.5, -0.2, 2.2, "; z; #frac{Model}{Pythia8}", false);
  frame12->GetYaxis()->SetLabelFont(43);
  frame12->GetYaxis()->SetTitleFont(43);
  frame12->GetYaxis()->SetLabelSize(25);
  frame12->GetYaxis()->SetTitleSize(25);
  frame12->GetYaxis()->SetTitleOffset(2.);
  frame12->GetYaxis()->SetNdivisions(505);

  frame12->GetXaxis()->SetLabelFont(43);
  frame12->GetXaxis()->SetTitleFont(43);
  frame12->GetXaxis()->SetLabelSize(20);
  frame12->GetXaxis()->SetTitleSize(30);
  frame12->GetXaxis()->SetTitleOffset(2.5);

  div11->Draw("E2 same");
  SetMarkerStyle(div11, 21, 2, 1.3);
  SetLineStyle(div11, 1, 2, 1);
  SetFillStyle(div11, 1001, 2, 0.4);

  div12->Draw("E2 same");
  SetMarkerStyle(div12, 33, 4, 1.3);
  SetLineStyle(div12, 1, 4, 1);
  SetFillStyle(div12, 1001, 4, 0.4);

  div13->Draw("E2 same");
  SetMarkerStyle(div13, 34, kGreen+2, 1.3);
  SetLineStyle(div13, 1, 3, 1);
  SetFillStyle(div13, 1001, 3, 0.4);

  TLine *line1 = new TLine(0,1,0.5,1);
  line1->SetLineStyle(2);
  line1->SetLineWidth(2);
  line1->SetLineColor(kBlack);
  line1->Draw("same");

  c1->SaveAs("../plot/z_all_zcut0p5_beta1p5.pdf");



  TCanvas* c2 = new TCanvas("c2", "c2", 600, 800);
  c2->cd();

  TPad* p21 = new TPad("p11", "p11", 0., 0.3, 1., 1.);
  p21->SetBottomMargin(0);
  p21->SetLeftMargin(0.18);
  p21->Draw();
  p21->cd();
  p21->SetLogy();
  TString title2 = ";; #frac{1}{N} #frac{dN}{d#Delta}";
  TH1F* frame21 = DrawFrame(0., 0.42, 5e-3, 5e3, title2, false);
  frame21->GetYaxis()->SetLabelFont(43);
  frame21->GetYaxis()->SetTitleFont(43);
  frame21->GetYaxis()->SetLabelSize(25);
  frame21->GetYaxis()->SetTitleSize(25);
  frame21->GetYaxis()->SetTitleOffset(2.);

  SetMarkerStyle(h_delta_pythia, 20, 1, 1.3);
  SetLineStyle(h_delta_pythia, 1, 1, 1);
  SetFillStyle(h_delta_pythia, 1001, 1, 0.4);
  h_delta_pythia->Draw("E2 same");

  SetMarkerStyle(h_delta_NR, 21, 2, 1.3);
  SetLineStyle(h_delta_NR, 1, 2, 1);
  SetFillStyle(h_delta_NR, 1001, 2, 0.4);
  h_delta_NR->Draw("E2 same");

  SetMarkerStyle(h_delta_R, 33, 4, 1.3);
  SetLineStyle(h_delta_R, 1, 4, 1);
  SetFillStyle(h_delta_R, 1001, 4, 0.4);
  h_delta_R->Draw("E2 same");

  SetMarkerStyle(h_delta_hybrid, 34, kGreen+2, 1.3);
  SetLineStyle(h_delta_hybrid, 1, 3, 1);
  SetFillStyle(h_delta_hybrid, 1001, 3, 0.4);
  h_delta_hybrid->Draw("E2 same");

  TLegend* lg2 = new TLegend(0.52, 0.4, 0.87, 0.6);
  lg2->AddEntry(h_delta_pythia, "Pythia8");
  lg2->AddEntry(h_delta_NR, "Jewel Recoil Off");
  lg2->AddEntry(h_delta_R, "Jewel Recoil On");
  lg2->AddEntry(h_delta_hybrid, "Hybrid");
  lg2->SetBorderSize(0);
  lg2->SetTextSize(0.04);
  lg2->Draw("same");

  Tl.DrawLatex(0.04,1e3,"p_{T,jet}> 80GeV, anti-k_{T}, R=0.4");
  Tl.DrawLatex(0.04,3e2,"Soft Drop z_{cut}=0.5, #beta=1.5");
  Tl.DrawLatex(0.04,1e2, "All Soft Drop");

  c2->cd();
  TPad* p22 = new TPad("", "", 0., 0., 1., 0.3);
  p22->SetLeftMargin(0.18);
  p22->SetTopMargin(0);
  p22->SetBottomMargin(0.22);
  p22->Draw();
  p22->cd();
  TH1F* frame22 = DrawFrame(0., 0.42, -0.2, 2.2, "; #Delta; #frac{Model}{Pythia8}", false);
  frame22->GetYaxis()->SetLabelFont(43);
  frame22->GetYaxis()->SetTitleFont(43);
  frame22->GetYaxis()->SetLabelSize(25);
  frame22->GetYaxis()->SetTitleSize(25);
  frame22->GetYaxis()->SetTitleOffset(2.);
  frame22->GetYaxis()->SetNdivisions(505);

  frame22->GetXaxis()->SetLabelFont(43);
  frame22->GetXaxis()->SetTitleFont(43);
  frame22->GetXaxis()->SetLabelSize(20);
  frame22->GetXaxis()->SetTitleSize(30);
  frame22->GetXaxis()->SetTitleOffset(2.5);

  div21->Draw("E2 same");
  SetMarkerStyle(div21, 21, 2, 1.3);
  SetLineStyle(div21, 1, 2, 1);
  SetFillStyle(div21, 1001, 2, 0.4);

  div22->Draw("E2 same");
  SetMarkerStyle(div22, 33, 4, 1.3);
  SetLineStyle(div22, 1, 4, 1);
  SetFillStyle(div22, 1001, 4, 0.4);

  div23->Draw("E2 same");
  SetMarkerStyle(div23, 34, kGreen+2, 1.3);
  SetLineStyle(div23, 1, 3, 1);
  SetFillStyle(div23, 1001, 3, 0.4);

  TLine *line2 = new TLine(0,1,0.42,1);
  line2->SetLineStyle(2);
  line2->SetLineWidth(2);
  line2->SetLineColor(kBlack);
  line2->Draw("same");


  c2->SaveAs("../plot/delta_all_zcut0p5_beta1p5.pdf");


  return 0;
}
