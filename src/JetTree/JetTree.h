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
public:
  // constructor
  GhostSubtractor(std::vector<fastjet::PseudoJet> ghosts):_ghosts(ghosts){};
  GhostSubtractor(std::vector<fastjet::PseudoJet> ghosts, bool do_subtraction):_ghosts(ghosts){};

  // generate dummy particles from ghosts
  std::vector<fastjet::PseudoJet> getDummies();

  // do ghost subtraction
  // used to corrent 4-mom from medium response
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
  // constructor
  SoftDropGroomer(double zcut=0.1, double beta=0., double r0=0.4):_zcut(zcut),_beta(beta),_r0(r0),_nsd(0){};

  // true if the branch doesn't satisfy soft drop condition
  bool softdrop(fastjet::PseudoJet j1, fastjet::PseudoJet j2);

  int nsd();

  void reset();

};

//************************************************
//              JetTree
//************************************************
class JetTree{
private:

  // jet/subjet
  fastjet::PseudoJet _pseudojet;

  // parent trees
  JetTree* _harder;
  JetTree* _softer;

  // child tree
  JetTree* _child;

public:

  // constructor destructor
  JetTree(fastjet::PseudoJet& pseudojet, JetTree* child=NULL, GhostSubtractor* gs=NULL);
  ~JetTree();

  // getter
  fastjet::PseudoJet pseudojet();
  JetTree* harder();
  JetTree* softer();
  JetTree* child();
  void swap();

  bool has_structure();
  void remove_soft();

  // variables
  double e1();
  double e2();
  double pt1();
  double pt2();
  double eta1();
  double eta2();
  double phi1();
  double phi2();
  double area1();
  double area2();

  double z();
  double delta();
  double kperp();
  double m();

  // grooming
  void groom(SoftDropGroomer* groomer);


};

#endif
