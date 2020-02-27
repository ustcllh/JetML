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
    _pseudojet = gs->getCorrectedJet(pseudojet);
  }
  else{
    _pseudojet = pseudojet;
  }

  fastjet::PseudoJet j1;
  fastjet::PseudoJet j2;
  if(pseudojet.has_parents(j1, j2)){
    _harder = new JetTree(j1, this, gs);
    _softer = new JetTree(j2, this, gs);
    if(gs){
      auto j1_sub = gs->getCorrectedJet(j1);
      auto j2_sub = gs->getCorrectedJet(j2);
      if(j1_sub.pt()<j2_sub.pt()) this->swap();
    }
    else{
      if(j1.pt()<j2.pt()) this->swap();
    }
  }

};

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

double JetTree::zg(SoftDropGroomer* groomer){
  double zg = -999;
  auto temp = this;
  while(temp && temp->has_structure()){
    auto harder = temp->harder()->pseudojet();
    auto softer = temp->softer()->pseudojet();
    if(groomer->softdrop(harder, softer)){
      temp = temp->harder();
    }
    else{
      zg = temp->z();
      return zg;
    }
  }
  return zg;
};

double JetTree::deltag(SoftDropGroomer* groomer){
  double deltag = -999;
  auto temp = this;
  while(temp && temp->has_structure()){
    auto harder = temp->harder()->pseudojet();
    auto softer = temp->softer()->pseudojet();
    if(groomer->softdrop(harder, softer)){
      temp = temp->harder();
    }
    else{
      deltag = temp->delta();
      return deltag;
    }
  }
  return deltag;
};

void JetTree::remove_soft(bool do_recursive_correction){
  auto correction = this->softer()->pseudojet();
  _softer = _harder->softer()? _harder->softer() : NULL;
  _harder = _harder->harder()? _harder->harder() : NULL;

  if(_harder) _harder->_child = this;
  if(_softer) _softer->_child = this;

  if(do_recursive_correction){
    auto child = this;
    while(child){
      auto temp = child->pseudojet();
      temp -= correction;
      child->_pseudojet = temp;
      if(child->has_structure() && child->pt1()<=child->pt2()) child->swap();
      child = child->child();

    }
  }

};

void JetTree::groom(SoftDropGroomer* groomer, bool do_recursive_correction){

  if(this->has_structure()){
    auto harder = _harder->pseudojet();
    auto softer = _softer->pseudojet();
    if(groomer->softdrop(harder, softer)){
      this->remove_soft(do_recursive_correction);
      this->groom(groomer, do_recursive_correction);
    }
    else{
      this->harder()->groom(groomer, do_recursive_correction);
      this->softer()->groom(groomer, do_recursive_correction);
    }
  }
};

// test
double JetTree::hello_world(){
  std::cout << "Hello World From JetTree!" << std::endl;
  return 0;
};

void JetTree::show(){
  if(this->has_structure()){
    std::cout << this->pt1() << " " << this->pt2() << std::endl;
  }

  if(_harder) _harder->show();
  else{
    std::cout << "end" << std::endl;
  }
  return;
}


//************************************************
//              SoftDropGroomer
//************************************************
bool SoftDropGroomer::softdrop(fastjet::PseudoJet j1, fastjet::PseudoJet j2){
  double delta = j1.delta_R(j2);
  double pt1 = j1.pt();
  double pt2 = j2.pt();

  double softdrop = pt2/(pt1+pt2) - _zcut * pow(delta/_r0, _beta);

  if(softdrop<0){
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
    if(particle.pt()>1e-5) continue;

    for(auto&& ghost : ghosts){
      auto dr = particle.delta_R(ghost);
      if(dr<1e-5){
        correction += ghost;
        ghost.reset(0., 0., 0., 0.);
      }
    }
  }
  if(_do_subtraction){
    auto temp = pseudojet - correction;
    return temp;
  }
  else{
    auto temp = pseudojet + correction;
    return temp;
  }
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
