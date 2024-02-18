#include "JetTree.h"
#include "fastjet/PseudoJet.hh"
#include <vector>
#include <iostream>
#include <cstddef>
#include <cmath>


//************************************************
//              JetTree
//************************************************
JetTree::JetTree(fastjet::PseudoJet& pseudojet, JetTree* child, GhostSubtractor* gs){
  _harder = NULL;
  _softer = NULL;
  _child = child;

  if(gs){
    auto corrected_jet = gs->getCorrectedJet(pseudojet);
    if(corrected_jet.e()<0 or corrected_jet.m()<0) return;
    else _pseudojet = corrected_jet;
  }
  else{
    _pseudojet = pseudojet;
  }

  fastjet::PseudoJet j1;
  fastjet::PseudoJet j2;
  // retrive two subjets from ClusterSequence
  if(pseudojet.has_parents(j1, j2)){

    // construct parent trees
    _harder = new JetTree(j1, this, gs);
    _softer = new JetTree(j2, this, gs);
    if(_harder->pseudojet().pt()<_softer->pseudojet().pt()) this->swap();
  }

};

JetTree::~JetTree(){
  _child = NULL;
  delete _harder;
  delete _softer;
}

fastjet::PseudoJet JetTree::pseudojet(){
  return _pseudojet;
};

bool JetTree::has_structure(){
  if(_harder && _softer){
    return true;
  }
  else{
    return false;
  }
};


JetTree* JetTree::harder(){
  return _harder;
};

JetTree* JetTree::softer(){
  return _softer;
};

JetTree* JetTree::child(){
  return _child;
};

void JetTree::swap(){
  auto temp = _harder;
  _harder = _softer;
  _softer = temp;
  return;
};

// variables

double JetTree::e1(){
  if(this->has_structure()){
    auto subjet = this->harder()->pseudojet();
    return subjet.e();
  }
  else return -999;
};

double JetTree::e2(){
  if(this->has_structure()){
    auto subjet = this->softer()->pseudojet();
    return subjet.e();
  }
  else return -999;
};

double JetTree::pt1(){
  if(this->has_structure()){
    auto subjet = this->harder()->pseudojet();
    return subjet.pt();
  }
  else return -999;
};

double JetTree::pt2(){
  if(this->has_structure()){
    auto subjet = this->softer()->pseudojet();
    return subjet.pt();
  }
  else return -999;
};

double JetTree::eta1(){
  if(this->has_structure()){
    auto subjet = this->harder()->pseudojet();
    return subjet.eta();
  }
  else return -999;
};

double JetTree::eta2(){
  if(this->has_structure()){
    auto subjet = this->softer()->pseudojet();
    return subjet.eta();
  }
  else return -999;
};

double JetTree::phi1(){
  if(this->has_structure()){
    auto subjet = this->harder()->pseudojet();
    return subjet.phi();
  }
  else return -999;
};

double JetTree::phi2(){
  if(this->has_structure()){
    auto subjet = this->softer()->pseudojet();
    return subjet.phi();
  }
  else return -999;
};

double JetTree::area1(){
  if(this->has_structure()){
    auto subjet = this->harder()->pseudojet();
    return subjet.area();
  }
  else return -999;
}

double JetTree::area2(){
  if(this->has_structure()){
    auto subjet = this->softer()->pseudojet();
    return subjet.area();
  }
  else return -999;
}

double JetTree::z(){
  if(this->has_structure()){
    double pt1 = this->pt1();
    double pt2 = this->pt2();
    if((pt1+pt2)<0) return -999;
    double z = pt2 / (pt1 + pt2);
    return z;
  }
  else return -999;
};

double JetTree::delta(){
  if(this->has_structure()){
    auto j1 = this->harder()->pseudojet();
    auto j2 = this->softer()->pseudojet();
    double delta = j1.delta_R(j2);
    return delta;
  }
  else return -999;
};

double JetTree::kperp(){
  if(this->has_structure()){
    double delta = this->delta();
    double pt2 = this->pt2();
    double kperp = delta * pt2;
    return kperp;
  }
  else return -999;
};

double JetTree::m(){
  auto temp = this->pseudojet();
  return temp.m();
};

// grooming

void JetTree::remove_soft(){
  // remove soft branch by replacing current branch with harder branch
  _pseudojet = _harder->pseudojet();

  _softer = _harder->softer()? _harder->softer() : NULL;
  _harder = _harder->harder()? _harder->harder() : NULL;

  if(_harder) _harder->_child = this;
  if(_softer) _softer->_child = this;


};

// soft drop
void JetTree::groom(SoftDropGroomer* groomer){

  if(this->has_structure()){
    auto harder = _harder->pseudojet();
    auto softer = _softer->pseudojet();
    if(groomer->softdrop(harder, softer)){
      // doesn't satisfy soft drop condition
      this->remove_soft();
      this->groom(groomer);
    }
  }
};

// recursive soft drop
/*
void JetTree::groom(SoftDropGroomer* groomer){

  if(this->has_structure()){
    auto harder = _harder->pseudojet();
    auto softer = _softer->pseudojet();
    if(groomer->softdrop(harder, softer)){
      this->remove_soft();
      this->groom(groomer, do_recursive_correction);
    }
    else{
      this->harder()->groom(groomer);
      this->softer()->groom(groomer);
    }
  }
};

*/


//************************************************
//              SoftDropGroomer
//************************************************
bool SoftDropGroomer::softdrop(fastjet::PseudoJet j1, fastjet::PseudoJet j2){
  double delta = j1.delta_R(j2);
  double pt1 = j1.pt();
  double pt2 = j2.pt();

  double softdrop = pt2/(pt1+pt2) - _zcut * pow(delta/_r0, _beta);

  if(softdrop<0){
    // doesn't satisfy soft drop condition
    return true;
  }
  else{
    _nsd++;
    return false;
  }
};

int SoftDropGroomer::nsd(){
  return _nsd;
};

void SoftDropGroomer::reset(){
  _nsd = 0;
};


//************************************************
//              GhostSubtractor
//************************************************
fastjet::PseudoJet GhostSubtractor::getCorrectedJet(fastjet::PseudoJet pseudojet){

  auto ghosts = _ghosts;
  auto correction = fastjet::PseudoJet(0., 0., 0., 0.);
  for(auto&& particle : pseudojet.constituents()){

    // not dummy particle
    if(particle.pt()>0.01) continue;

    for(auto&& ghost : ghosts){
      auto deta = particle.eta() - ghost.eta();
      auto dphi = particle.delta_phi_to(ghost);
      auto dr = std::sqrt(deta*deta+dphi*dphi);
      if(dr<1e-5){
        // match dummy particle to thermal parton (ghost)
        correction += ghost;
        ghost.reset(0., 0., 0., 0.);
      }
    }
  }
  auto temp = pseudojet - correction;
  return temp;
};

std::vector<fastjet::PseudoJet> GhostSubtractor::getDummies(){
  std::vector<fastjet::PseudoJet> dummies;
  for(auto&& ghost : _ghosts){
    auto eta = ghost.eta();
    auto phi = ghost.phi();
    auto e = 1e-6;
    auto px = e * cos(phi) / cosh(eta);
    auto py = e * sin(phi) / cosh(eta);
    auto pz = e * tanh(eta);
    dummies.push_back(fastjet::PseudoJet(px, py, pz, e));
  }
  return dummies;
};
