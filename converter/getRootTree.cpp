#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include "TFile.h"
#include "TTree.h"

int main(int argc, char* argv[]){

  std::string ifn_jet = "./res/results.txt";
  std::string ifn_jet_btree = "./res/btree.txt";
  std::string ifn_jet_groomed = "./res/results_groomed.txt";
  std::string ifn_jet_btree_groomed = "./res/btree_groomed.txt";

  std::string ofn = "./res/jetml.root";
  TFile* ofs = new TFile(ofn.c_str(), "RECREATE");

  // jet variables
  float px, py, pz, e;
  float eta, phi;
  float lstm, cnn;
  float m, zg, delta;
  float ang_ptd, ang_mass, ang_width;
  float ratio, dr;
  int isgluon;
  double pid;


  TTree* jet_tr = new TTree("jet", "jet");
  jet_tr->Branch("px", &px, "px/F");
  jet_tr->Branch("py", &py, "py/F");
  jet_tr->Branch("pz", &pz, "pz/F");
  jet_tr->Branch("e", &e, "e/F");
  jet_tr->Branch("eta", &eta, "eta/F");
  jet_tr->Branch("phi", &phi, "phi/F");
  jet_tr->Branch("lstm", &lstm, "lstm/F");
  jet_tr->Branch("cnn", &cnn, "cnn/F");
  jet_tr->Branch("m", &m, "m/F");
  jet_tr->Branch("zg", &zg, "zg/F");
  jet_tr->Branch("delta", &delta, "delta/F");
  jet_tr->Branch("ang_ptd", &ang_ptd, "ang_ptd/F");
  jet_tr->Branch("ang_mass", &ang_mass, "ang_mass/F");
  jet_tr->Branch("ang_width", &ang_width, "ang_width/F");
  jet_tr->Branch("ratio", &ratio, "ratio/F");
  jet_tr->Branch("dr", &dr, "dr/F");
  jet_tr->Branch("isgluon", &isgluon, "isgluon/I");

  // jet binary tree variables

  unsigned long idx;
  int primary;
  float lnm, lnkt, lnz, lndelta, lnkappa, psi;

  TTree* jet_btr = new TTree("jet_binary_tree", "jet_binary_tree");
  // jet_btr->Branch("idx", &idx, "idx/l");
  // jet_btr->Branch("primary", &primary, "primary/I");
  jet_btr->Branch("lnm", &lnm, "lnm/F");
  jet_btr->Branch("lnkt", &lnkt, "lnkt/F");
  jet_btr->Branch("lnz", &lnz, "lnz/F");
  jet_btr->Branch("lndelta", &lndelta, "lndelta/F");
  jet_btr->Branch("lnkappa", &lnkappa, "lnkappa/F");
  jet_btr->Branch("psi", &psi, "psi/F");
  jet_btr->Branch("lstm", &lstm, "lstm/F");
  jet_btr->Branch("cnn", &cnn, "cnn/F");

  // groomed jet

  TTree* jet_tr_groomed = new TTree("jet_groomed", "jet_groomed");
  jet_tr_groomed->Branch("px", &px, "px/F");
  jet_tr_groomed->Branch("py", &py, "py/F");
  jet_tr_groomed->Branch("pz", &pz, "pz/F");
  jet_tr_groomed->Branch("e", &e, "e/F");
  jet_tr_groomed->Branch("lstm", &lstm, "lstm/F");
  jet_tr_groomed->Branch("cnn", &cnn, "cnn/F");
  jet_tr_groomed->Branch("m", &m, "m/F");
  jet_tr_groomed->Branch("zg", &zg, "zg/F");
  jet_tr_groomed->Branch("delta", &delta, "delta/F");

  // groomed jet binary tree

  TTree* jet_btr_groomed = new TTree("jet_groomed_binary_tree", "jet_groomed_binary_tree");
  // jet_btr_groomed->Branch("idx", &idx, "idx/l");
  // jet_btr_groomed->Branch("primary", &primary, "primary/I");
  jet_btr_groomed->Branch("lnm", &lnm, "lnm/F");
  jet_btr_groomed->Branch("lnkt", &lnkt, "lnkt/F");
  jet_btr_groomed->Branch("lnz", &lnz, "lnz/F");
  jet_btr_groomed->Branch("lndelta", &lndelta, "lndelta/F");
  jet_btr_groomed->Branch("lnkappa", &lnkappa, "lnkappa/F");
  jet_btr_groomed->Branch("psi", &psi, "psi/F");
  jet_btr_groomed->Branch("lstm", &lstm, "lstm/F");
  jet_btr_groomed->Branch("cnn", &cnn, "cnn/F");

  std::ifstream ifs1(ifn_jet);
  while(1){
    ifs1 >> px >> py >> pz >> e >> eta >> phi >> lstm >> cnn >> m >> zg >> delta >> ang_ptd >> ang_mass >> ang_width >> ratio >> dr >> pid;
    if(!ifs1.good()) break;
    isgluon = 0;
    if(pid>20 && pid<22) isgluon = 1;
    if(pid<-1)  isgluon = -9999;
    jet_tr->Fill();
  }

  std::ifstream ifs2(ifn_jet_btree);
  while(1){
    ifs2 >> lnm >> lnkt >> lnz >> lndelta >> lnkappa >> psi >> lstm >> cnn;
    if(!ifs2.good()) break;
    jet_btr->Fill();
  }

  std::ifstream ifs3(ifn_jet_groomed);
  while(1){
    ifs3 >> px >> py >> pz >> e >> lstm >> cnn >> m >> zg >> delta;
    if(!ifs3.good()) break;
    jet_tr_groomed->Fill();
  }

  std::ifstream ifs4(ifn_jet_btree_groomed);
  while(1){
    ifs4 >> lnm >> lnkt >> lnz >> lndelta >> lnkappa >> psi >> lstm >> cnn;
    if(!ifs4.good()) break;
    jet_btr_groomed->Fill();
  }

  ofs->cd();
  jet_tr->Write();
  jet_btr->Write();
  jet_tr_groomed->Write();
  jet_btr_groomed->Write();
  ofs->Close();

  return 0;
}
