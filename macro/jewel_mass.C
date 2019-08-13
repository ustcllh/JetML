

void jewel_mass(){
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  TString ifn = "../res/jetml.root";
  TFile* f = new TFile(ifn, "READ");
  TTree* jet_tr= (TTree*) f->Get("jet");
  int nEntries = jet_tr->GetEntries();
  std::cout<<nEntries<<std::endl;

  float m;
  float lstm;
  int isgluon;

  jet_tr->SetBranchAddress("m", &m);
  jet_tr->SetBranchAddress("lstm", &lstm);
  jet_tr->SetBranchAddress("isgluon", &isgluon);

  TH1F* h0 = new TH1F("h0", "h0", 50, 0, 50);
  TH1F* h1 = new TH1F("h1", "h1", 50, 0, 50);

  for(int i=0; i<nEntries; i++){
    jet_tr->GetEntry(i);
    if(isgluon==1) h1->Fill(m);
    else if(isgluon==0) h0->Fill(m);
    else continue;
  }
  h0->Sumw2();
  h0->Scale(1./h0->Integral());
  h1->Sumw2();
  h1->Scale(1./h1->Integral());

  h0->SetMarkerColor(kRed);
  h1->SetMarkerColor(kBlue);
  h0->SetMarkerStyle(21);
  h1->SetMarkerStyle(22);

  TCanvas* c = new TCanvas("c", "c", 600, 600);
  c->cd();
  h0->Draw("E1P");
  h1->Draw("E1PSAME");
  c->SaveAs("../plot/m_qgjets.pdf");

  c->cd();
  jet_tr->Draw("ratio:lstm>>h2(40, 0.1, 0.9, 40, 0, 2)", "isgluon==1", "colz");
  c->SaveAs("../plot/ratio_lstm_gluon.pdf");
  jet_tr->Draw("ratio:lstm>>h3(40, 0.1, 0.9, 40, 0, 2)", "isgluon==0", "colz");
  c->SaveAs("../plot/ratio_lstm_quark.pdf");

}
