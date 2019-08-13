int lund_plot(){
    gStyle->SetOptStat(0);
    // gStyle->SetPadTickX(1);
    // gStyle->SetPadTickY(1);

    TString ifn = "../res/jetml.root";
    TFile* f = new TFile(ifn, "READ");
    TTree* jet_tr= (TTree*) f->Get("jet_binary_tree");
    // TTree* jet_groomed_tr= (TTree*) f->Get("jet_groomed_binary_tree");
    // int nEntries = jet_tr->GetEntries();
    // std::cout<<nEntries<<std::endl;

    TCanvas* c1 = new TCanvas("c1", "c1", 600, 600);
    gPad->SetMargin(0.1, 0.1, 0.12, 0.08);
    jet_tr->Draw("lnz+lndelta:-lndelta>>lund0(100, 1, 7, 100, -10, -2)", "lstm<0.2", "colz");
    TH2F* lund0 = (TH2F*) gDirectory->Get("lund0");
    lund0->GetXaxis()->SetTitle("ln(1/#theta)");
    lund0->GetYaxis()->SetTitle("lnz#theta");
    lund0->GetXaxis()->CenterTitle();
    lund0->GetYaxis()->CenterTitle();
    lund0->SetTitle("");
    lund0->GetXaxis()->SetTitleSize(0.05);
    lund0->GetYaxis()->SetTitleSize(0.05);
    lund0->GetYaxis()->SetTitleOffset(0.8);
    c1->SaveAs("../plot/lund0.pdf");

    TCanvas* c2 = new TCanvas("c2", "c2", 600, 600);
    gPad->SetMargin(0.1, 0.1, 0.12, 0.08);
    jet_tr->Draw("lnz+lndelta:-lndelta>>lund1(100, 1, 7, 100, -10, -2)", "lstm>0.8", "colz");
    TH2F* lund1 = (TH2F*) gDirectory->Get("lund1");
    lund1->GetXaxis()->SetTitle("ln(1/#theta)");
    lund1->GetYaxis()->SetTitle("lnz#theta");
    lund1->GetXaxis()->CenterTitle();
    lund1->GetYaxis()->CenterTitle();
    lund1->SetTitle("");
    lund1->GetXaxis()->SetTitleSize(0.05);
    lund1->GetYaxis()->SetTitleSize(0.05);
    lund1->GetYaxis()->SetTitleOffset(0.8);
    c2->SaveAs("../plot/lund1.pdf");



    int color_i = TColor::GetFreeColorIndex();
    TColor *color = new TColor(color_i, 68./255, 114./255, 196./255);

    TCanvas* c3 = new TCanvas("c3", "c3", 600, 600);
    gPad->SetMargin(0.14, 0.06, 0.12, 0.08);
    jet_tr->Draw("lstm>>lstm(40, 0, 1)");
    TH1F* lstm = (TH1F*) gDirectory->Get("lstm");
    lstm->Scale(1./lstm->Integral()/0.025);
    lstm->GetXaxis()->SetTitle("LSTM Prediction");
    lstm->GetYaxis()->SetTitle("dP(x)/dx");
    lstm->GetXaxis()->CenterTitle();
    lstm->GetYaxis()->CenterTitle();
    lstm->SetTitle("");
    lstm->GetXaxis()->SetTitleSize(0.05);
    lstm->GetYaxis()->SetTitleSize(0.05);
    lstm->GetYaxis()->SetTitleOffset(1.2);
    lstm->SetFillColorAlpha(color_i,1.);
    lstm->SetLineColorAlpha(color_i,1.);
    lstm->Draw("HIST");
    c3->SaveAs("../plot/lstm.pdf");
    return 0;

}
