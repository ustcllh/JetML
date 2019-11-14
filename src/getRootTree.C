#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "TFile.h"
#include "TTree.h"

#define MAX_DEPTH 100

int main(int argc, char* argv[]){

  if(argc!=2){
    std::cout << "Usage: ./getRootTree <input prefix>" << std::endl;
    return -1;
  }

  std::string ifn_prefix = argv[1];

  std::string ifn_jet = ifn_prefix + "_jet";
  std::string ifn_strc = ifn_prefix + "_strc";


  std::string ofn = ifn_prefix + ".root";
  TFile* ofs = new TFile(ofn.c_str(), "RECREATE");

  // jet variables
  float px, py, pz, e;
  float eta, phi;
  float lstm;
  float ang_ptd, ang_mass, ang_width;
  float ratio, dr;
  bool isgluon;
  // double pid;

  // structure variables
  int depth;
  float lnm[MAX_DEPTH], lnz[MAX_DEPTH], lndelta[MAX_DEPTH], lnkt[MAX_DEPTH], psi[MAX_DEPTH], lnkappa[MAX_DEPTH];


  // jet
  TTree* jet_tr = new TTree("jet", "jet");
  jet_tr->Branch("px", &px, "px/F");
  jet_tr->Branch("py", &py, "py/F");
  jet_tr->Branch("pz", &pz, "pz/F");
  jet_tr->Branch("e", &e, "e/F");
  jet_tr->Branch("eta", &eta, "eta/F");
  jet_tr->Branch("phi", &phi, "phi/F");
  jet_tr->Branch("lstm", &lstm, "lstm/F");

  //parton
  jet_tr->Branch("ratio", &ratio, "ratio/F");
  jet_tr->Branch("dr", &dr, "dr/F");
  jet_tr->Branch("isgluon", &isgluon, "isgluon/O");

  // angularities
  jet_tr->Branch("ang_ptd", &ang_ptd, "ang_ptd/F");
  jet_tr->Branch("ang_mass", &ang_mass, "ang_mass/F");
  jet_tr->Branch("ang_width", &ang_width, "ang_width/F");

  // jet structure
  jet_tr->Branch("depth", &depth, "depth/I");
  jet_tr->Branch("lnz", lnz, "lnz[depth]/F");
  jet_tr->Branch("lndelta", lndelta, "lndelta[depth]/F");
  jet_tr->Branch("lnkt", lnkt, "lnkt[depth]/F");
  jet_tr->Branch("lnm", lnm, "lnm[depth]/F");
  jet_tr->Branch("psi", psi, "psi[depth]/F");
  jet_tr->Branch("lnkappa", lnkappa, "lnkappa[depth]/F");


  std::ifstream ifs_jet(ifn_jet);
  std::ifstream ifs_strc(ifn_strc);

  std::string line_jet, line_strc;
  while(std::getline(ifs_jet, line_jet) && std::getline(ifs_strc, line_strc)){
    std::stringstream ss;

    ss.str(line_strc);
    int icell = 0;
    std::string cell;
    while(std::getline(ss, cell, ',')){
      int mod = icell / 6;
      int rem = icell % 6;
      if(rem==0){
        lnz[mod] = std::stof(cell);
      }
      else if(rem==1){
        lndelta[mod] = std::stof(cell);
      }
      else if(rem==2){
        lnkt[mod] = std::stof(cell);
      }
      else if(rem==3){
        lnm[mod] = std::stof(cell);
      }
      else if(rem==4){
        psi[mod] = std::stof(cell);
      }
      else if(rem==5){
        lnkappa[mod] = std::stof(cell);
      }
      icell++;
    }
    depth = icell / 6;

    // std::cout << depth << std::endl;
    ss.clear();
    ss.str(line_jet);
    icell = 0;
    while(std::getline(ss, cell, ',')){
      if(icell==0){
        px = std::stof(cell);
      }
      else if(icell==1){
        py = std::stof(cell);
      }
      else if(icell==2){
        pz = std::stof(cell);
      }
      else if(icell==3){
        e = std::stof(cell);
      }
      else if(icell==4){
        eta = std::stof(cell);
      }
      else if(icell==5){
        phi = std::stof(cell);
      }
      else if(icell==6){
        lstm = std::stof(cell);
      }
      else if(icell==7){
        ratio = std::stof(cell);
      }
      else if(icell==8){
        dr = std::stof(cell);
      }
      else if(icell==9){
        int pid = std::stoi(cell);
        if(pid==21)
          isgluon = 1;
        else
          isgluon = 0;
      }
      else if(icell==10){
        ang_ptd = std::stof(cell);
      }
      else if(icell==11){
        ang_mass = std::stof(cell);
      }
      else if(icell==12){
        ang_width = std::stof(cell);
      }
      icell++;
      // ss >> px >> py >> pz >> e >> eta >> phi >> lstm >> ang_ptd >> ang_mass >> ang_width >> ratio >> dr >> isgluon;
    }
    jet_tr->Fill();
  }

  ofs->cd();
  jet_tr->Write();
  ofs->Close();

  return 0;
}
