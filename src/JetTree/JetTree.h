#ifndef __JETTREE_H__
#define __JETTREE_H__

#include "fastjet/PseudoJet.hh"
#include <vector>
#include <cstddef>

//************************************************
//              GhostSubtractor
//************************************************
class GhostSubtractor{
private:
  std::vector<fastjet::PseudoJet> _ghosts;
  bool _do_subtraction;
public:
  // constructor
  GhostSubtractor(std::vector<fastjet::PseudoJet> ghosts):_ghosts(ghosts),_do_subtraction(true){};
  GhostSubtractor(std::vector<fastjet::PseudoJet> ghosts, bool do_subtraction):_ghosts(ghosts),_do_subtraction(do_subtraction){};

  // generate dummy particles from ghosts
  std::vector<fastjet::PseudoJet> getDummies();

  // do ghost subtraction
  fastjet::PseudoJet getCorrectedJet(fastjet::PseudoJet pseudojet);

};


//************************************************
//              SoftDropGroomer
//************************************************
class SoftDropGroomer{
private:
  double _zcut;
  double _beta;
  double _r0;

  int _nsd;

public:
  SoftDropGroomer(double zcut=0.1, double beta=0., double r0=0.4):_zcut(zcut),_beta(beta),_r0(r0),_nsd(0){};

  bool softdrop(fastjet::PseudoJet j1, fastjet::PseudoJet j2);

  int nsd();

  void reset();

};

//************************************************
//              JetTree
//************************************************
class JetTree{
private:

  // binary tree
  fastjet::PseudoJet _pseudojet;
  JetTree* _harder;
  JetTree* _softer;
  JetTree* _child;

public:

  // constructor
  JetTree(fastjet::PseudoJet& pseudojet, JetTree* child=NULL, GhostSubtractor* gs=NULL);

  // binary tree
  fastjet::PseudoJet pseudojet();
  JetTree* harder();
  JetTree* softer();
  JetTree* child();
  void swap();

  bool has_structure();
  void remove_soft(bool do_recursive_correction=false);

  // variables
  double e1();
  double e2();
  double pt1();
  double pt2();
  double eta1();
  double eta2();
  double phi1();
  double phi2();

  double z();
  double delta();
  double kperp();

  // grooming
  void groom(SoftDropGroomer* groomer, bool do_recursive_correction=false);
  double zg(SoftDropGroomer* groomer);
  double deltag(SoftDropGroomer* groomer);

  // test
  double hello_world();
  void show();

};

#endif
