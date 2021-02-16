#!/opt/conda/bin/python

########################################
#               parsing
########################################
import sys, argparse

parser = argparse.ArgumentParser(description='JetML CML Parser')
parser.add_argument('-i','--input', help='Input file name', required=True)
parser.add_argument('-b','--background',help='Background file name', required=False)
parser.add_argument('-o','--output',help='Output file name', required=True)
parser.add_argument('-n',help='number of events', required=False)

parser.add_argument('-g',help='do ghost subtraction', required=False, default=False, action='store_true')

args = parser.parse_args()

# print values
print('*********Initialization**********')
print('Input file:\t%s' % args.input)
print('Background file:\t%s' % args.background)
print('Output file: \t%s' % args.output)
if args.n:
    print('Number of events:]\t%s' % args.n)
else:
    print('Number of events:\tAll')
print('Do ghost subtraction:\t%r' % args.g)
print('*********************************')

########################################
#              Root Tree
########################################

from ROOT import TFile, TTree
from array import array

output = TFile(args.output, 'recreate')

tr = TTree('jet', 'jet')
trcsj = TTree('csjjet', 'cs jet by jet')
trcse = TTree('csejet', 'cs event wide')
trics = TTree('icsjet', 'cs iterative')


maxn = 100
# event variables
x = array('f', [0.])
y = array('f', [0.])
weight = array('f', [0.])
event = array('i', [0])

# jet variables
eta = array('f', [0.])
phi = array('f', [0.])
jetpt = array('f', [0.])
depth = array('i', [0])
jetm = array('f', [0.])

# constituents
cn = array('i', [0])
cpt = array('f', 500 * [0.])
cdeta = array('f', 500 * [0.])
cdphi = array('f', 500 * [0.])

# groomed jet variables
jetmg = array('f', [0.])
depthg = array('i', [0])

# sequential variables
z = array('f', maxn * [0.])
delta = array('f', maxn * [0.])
kperp = array('f', maxn * [0.])
m = array('f', maxn * [0.])

def set_branches(tr):
    # event variables
    tr.Branch('event', event, 'event/I')
    tr.Branch('x', x, 'x/F')
    tr.Branch('y', y, 'y/F')
    tr.Branch('weight', weight, 'weight/F')

    # jet variables
    tr.Branch('eta', eta, 'eta/F')
    tr.Branch('phi', phi, 'phi/F')
    tr.Branch('jetpt', jetpt, 'jetpt/F')
    tr.Branch('depth', depth, 'depth/I')
    tr.Branch('jetm', jetm, 'jetm/F')

    # constituents
    tr.Branch('cn', cn, 'cn/I')
    tr.Branch('cpt', cpt, 'cpt[cn]/F')
    tr.Branch('cdeta', cdeta, 'cdeta[cn]/F')
    tr.Branch('cdphi', cdphi, 'cdphi[cn]/F')

    # groomed jet variables
    tr.Branch('depthg', depthg, 'depthg/I')
    tr.Branch('jetmg', jetmg, 'jetmg/F')

    # substructure variables without grooming
    tr.Branch('z', z, 'z[depth]/F')
    tr.Branch('delta', delta, 'delta[depth]/F')
    tr.Branch('kperp', kperp, 'kperp[depth]/F')
    tr.Branch('m', m, 'm[depth]/F')


set_branches(tr)
set_branches(trcsj)
set_branches(trcse)
set_branches(trics)


########################################
#                Jet
########################################

fastjet_dir='/workspace/fastjet/lib/python3.6/site-packages'
sys.path.append(fastjet_dir)
import fastjet as fj
from src.JetML.Event import *
from src.JetTree.JetTree import *

# constituent subtraction python module
constituent_subtractor_dir='/workspace/ConstituentSubtractor'
sys.path.append(constituent_subtractor_dir)
import ConstituentSubtractor as cs
import IterativeConstituentSubtractor as ics


# soft drop groomer
zcut=0.1
beta=0.
sd = SoftDropGroomer(zcut=0.1, beta=0.)
print('Soft Drop Groomer: zcut=%.1f, beta=%.1f' % (zcut, beta))

# CS parameters
max_distance = 1
alpha = 1
ghost_area = 0.01

print('Constituent Subtraction [max_distance, alpha, ghost_area]: [%.1f, %.1f, %.2f]' % (max_distance, alpha, ghost_area))

# ICS parameters
max_distances = [0.2, 0.05]
alphas = [1, 1]

print('Iterative Constituent Subtraction [max_distances, alphas, ghost_area]: [[%.1f, %.1f], [%.1f, %.1f], %.2f]' % (max_distances[0], max_distances[1], alphas[0], alphas[1], ghost_area))

# jet clustering w/o background
def do_clustering(hard_event, ptmin=100.):
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq= fj.ClusterSequence(hard_event, jet_def)
    jets = clust_seq.inclusive_jets(ptmin)
    return jets, clust_seq

# cs jet by jet
def do_cs_jet_by_jet(full_event, ptmin=100.):
    # clustering with ghosts and get the jets
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    ghost_RapMax = 3.
    ghost_spec = fj.GhostedAreaSpec(ghost_RapMax, 1, ghost_area)
    area_def = fj.AreaDefinition(fj.active_area_explicit_ghosts, ghost_spec)
    clust_seq_full = fj.ClusterSequenceArea(full_event, jet_def, area_def)
    full_jets = clust_seq_full.inclusive_jets(ptmin)

    # background estimation
    jet_def_rho= fj.JetDefinition(fj.kt_algorithm, 0.4)
    area_def_rho = fj.AreaDefinition(fj.active_area_explicit_ghosts, ghost_spec)

    rho_range = fj.SelectorAbsRapMax(3.0)
    bge_rho = fj.JetMedianBackgroundEstimator(rho_range, jet_def_rho, area_def_rho)
    bge_rho.set_particles(full_event)

    # subtractor
    subtractor = cs.ConstituentSubtractor()
    subtractor.set_distance_type(0)
    subtractor.set_max_distance(max_distance)
    subtractor.set_alpha(alpha)
    subtractor.set_max_eta(3.0)
    subtractor.set_background_estimator(bge_rho)


    # do subtraction
    corrected_jets = cs.PseudoJetVec()
    for jet in full_jets:
        subtracted_jet = subtractor.result(jet)
        corrected_jets.push_back(subtracted_jet)

    # pt cut
    selector = fj.SelectorPtMin(ptmin)
    return selector(corrected_jets), []

# jet clustering with cs
def do_cs_event_wide(full_event, ptmin=100.):
    max_eta = 3.

    # background estimator
    bge_rho = fj.GridMedianBackgroundEstimator(max_eta, 0.5)
    subtractor = cs.ConstituentSubtractor()
    # enum Distance {
    #     deltaR,
    #     angle
    # }
    subtractor.set_distance_type(0)

    subtractor.set_max_distance(max_distance)
    subtractor.set_alpha(alpha)
    subtractor.set_ghost_area(ghost_area)
    subtractor.set_max_eta(max_eta)
    subtractor.set_background_estimator(bge_rho)
    # sel_max_pt = fj.SelectorPtMax(15)
    # subtractor.set_particle_selector(sel_max_pt)
    subtractor.initialize()

    bge_rho.set_particles(full_event)

    corrected_event = subtractor.subtract_event(full_event)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq_corr = fj.ClusterSequence(corrected_event, jet_def)
    corrected_jets = clust_seq_corr.inclusive_jets(ptmin)

    return corrected_jets, clust_seq_corr

# jet clustering with ics
def do_cs_iterative(full_event, ptmin=100.):
    max_eta = 3.

    bge_rho = fj.GridMedianBackgroundEstimator(3., 0.5)
    subtractor = ics.IterativeConstituentSubtractor()
    subtractor.set_distance_type(0)

    max_distances_vec = ics.DoubleVec()
    max_distances_vec.push_back(max_distances[0])
    max_distances_vec.push_back(max_distances[1])

    alphas_vec = ics.DoubleVec()
    alphas_vec.push_back(alphas[0])
    alphas_vec.push_back(alphas[1])

    subtractor.set_parameters(max_distances_vec,alphas_vec)
    subtractor.set_ghost_removal(True)
    subtractor.set_ghost_area(ghost_area)
    subtractor.set_max_eta(max_eta);
    subtractor.set_background_estimator(bge_rho)

    # sel_max_pt = fj.SelectorPtMax(15)
    # only particles with pt<15 will be corrected - the other particles will be copied without any changes.

    # subtractor.set_particle_selector(sel_max_pt)
    subtractor.initialize()

    bge_rho.set_particles(full_event)

    # clustering
    corrected_event = subtractor.subtract_event(full_event)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq_corr = fj.ClusterSequence(corrected_event, jet_def)
    corrected_jets = clust_seq_corr.inclusive_jets(ptmin)

    return corrected_jets, clust_seq_corr


########################################
#                Running
########################################

# hard event
rd = Reader(args.input)
dict_input, des_input = rd.next_event()

# thermal event

try:
    rd_bkg = Reader(args.background)
    dict_bkg, des_bkg = rd_bkg.next_event()

except:
    print("No background files.")

# jet recluster
jr = JetFinder(algorithm=fj.cambridge_algorithm, R=999., ptmin=0)

# mix event
def mix_event(dict_input, dict_bkg):
    hard_event = cs.PseudoJetVec()
    full_event = cs.PseudoJetVec()

    hard_event.clear()
    full_event.clear()
    for p in dict_input['0']:
        hard_event.push_back(p)
        full_event.push_back(p)
    for p in dict_bkg['1']:
        full_event.push_back(p)

    return hard_event, full_event

def delta_phi(phi1, phi2):
    pi = 3.1415926
    dphi = phi1 - phi2
    if dphi > pi:
        dphi -= 2*pi
    if dphi < -pi:
        dphi += 2*pi
    return dphi

def delta_eta(eta1, eta2):
    deta = eta1 - eta2
    return deta

nevent = 0

try:
    nevent_max = int(args.n)
except:
    nevent_max = 100

while dict_input['0'] and dict_bkg['1']:
    if nevent == nevent_max:
        break

    dict_input, des_input = rd.next_event()
    dict_bkg, des_dkg = rd_bkg.next_event()
    nevent+=1

    event[0] = nevent
    hard_event, full_event = mix_event(dict_input, dict_bkg)
    ghosts = PseudoJetVec()

    for ghost in dict_input['-1']:
        ghosts.push_back(ghost)
    gs = GhostSubtractor(ghosts)

    # cuts
    ptmin = 130
    ptmin_cs = 130
    jet_eta_cut = 3.

    # hard event only
    jets, clust_seq = do_clustering(hard_event, ptmin)
    for jet in jets:
        if abs(jet.eta())>jet_eta_cut:
            continue
        constituents = [i for i in jet.constituents()]

        cn[0] = len(constituents)
        for i in range(cn[0]):
            cdeta[i] = delta_eta(constituents[i].eta(), jet.eta())
            cdphi[i] = delta_phi(constituents[i].phi(), jet.phi())
            cpt[i] = constituents[i].pt()

        rjets = jr(constituents)
        rjet = rjets[0]

        x[0] = 0
        y[0] = 0

        try:
            weight[0] = float(des_input['weight'])
        except:
            weight[0] = 1

        depth[0] = 0
        depthg[0] = 0

        jtr = None
        if args.g == 'False':
            jtr = JetTree(rjet)
        else:
            jtr = JetTree(rjet, gs=gs)

        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        jetm[0] = jtr.pseudojet().m()

        temp = jtr
        while temp.has_structure():
            z[depth[0]] = temp.z()
            delta[depth[0]] = temp.delta()
            kperp[depth[0]] = temp.kperp()
            m[depth[0]] = temp.m()
            depth[0] += 1
            temp = temp.harder()

        temp = jtr
        temp.groom(groomer=sd, do_recursive_correction=True)
        jetmg[0] = temp.pseudojet().m()
        while temp.has_structure():
            depthg[0] += 1
            temp = temp.harder()
        tr.Fill()

    # cs jet by jet
    corrected_jets, clust_seq_corr = do_cs_jet_by_jet(full_event, ptmin)

    for jet in corrected_jets:
        if abs(jet.eta())>3:
            continue
        constituents = [i for i in jet.constituents()]

        cn[0] = len(constituents)
        for i in range(cn[0]):
            cdeta[i] = delta_eta(constituents[i].eta(), jet.eta())
            cdphi[i] = delta_phi(constituents[i].phi(), jet.phi())
            cpt[i] = constituents[i].pt()

        rjets = jr(constituents)
        rjet = rjets[0]

        x[0] = 0
        y[0] = 0

        try:
            weight[0] = float(des_input['weight'])
        except:
            weight[0] = 1

        depth[0] = 0
        depthg[0] = 0

        jtr = None
        if args.g == 'False':
            jtr = JetTree(rjet)
        else:
            jtr = JetTree(rjet, gs=gs)

        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        jetm[0] = jtr.pseudojet().m()

        temp = jtr
        while temp.has_structure():
            z[depth[0]] = temp.z()
            delta[depth[0]] = temp.delta()
            kperp[depth[0]] = temp.kperp()
            m[depth[0]] = temp.m()
            depth[0] += 1
            temp = temp.harder()

        temp = jtr
        temp.groom(groomer=sd, do_recursive_correction=True)
        jetmg[0] = temp.pseudojet().m()
        while temp.has_structure():
            depthg[0] += 1
            temp = temp.harder()
        trcsj.Fill()

    # cs event wide
    corrected_jets, clust_seq_corr = do_cs_event_wide(full_event, ptmin)
    for jet in corrected_jets:
        if abs(jet.eta())>3:
            continue
        constituents = [i for i in jet.constituents()]

        cn[0] = len(constituents)
        for i in range(cn[0]):
            cdeta[i] = delta_eta(constituents[i].eta(), jet.eta())
            cdphi[i] = delta_phi(constituents[i].phi(), jet.phi())
            cpt[i] = constituents[i].pt()

        rjets = jr(constituents)
        rjet = rjets[0]

        x[0] = 0
        y[0] = 0

        try:
            weight[0] = float(des_input['weight'])
        except:
            weight[0] = 1

        depth[0] = 0
        depthg[0] = 0

        jtr = None
        if args.g == 'False':
            jtr = JetTree(rjet)
        else:
            jtr = JetTree(rjet, gs=gs)

        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        jetm[0] = jtr.pseudojet().m()

        temp = jtr
        while temp.has_structure():
            z[depth[0]] = temp.z()
            delta[depth[0]] = temp.delta()
            kperp[depth[0]] = temp.kperp()
            m[depth[0]] = temp.m()
            depth[0] += 1
            temp = temp.harder()

        temp = jtr
        temp.groom(groomer=sd, do_recursive_correction=True)
        jetmg[0] = temp.pseudojet().m()
        while temp.has_structure():
            depthg[0] += 1
            temp = temp.harder()
        trcse.Fill()

    # jet ics
    corrected_jets, clust_seq_corr = do_cs_iterative(full_event, ptmin_cs)
    for jet in corrected_jets:
        if abs(jet.eta())>3:
            continue
        constituents = [i for i in jet.constituents()]

        cn[0] = len(constituents)
        for i in range(cn[0]):
            cdeta[i] = delta_eta(constituents[i].eta(), jet.eta())
            cdphi[i] = delta_phi(constituents[i].phi(), jet.phi())
            cpt[i] = constituents[i].pt()

        rjets = jr(constituents)
        rjet = rjets[0]

        x[0] = 0
        y[0] = 0

        try:
            weight[0] = float(des_input['weight'])
        except:
            weight[0] = 1

        depth[0] = 0
        depthg[0] = 0

        jtr = None
        if args.g == 'False':
            jtr = JetTree(rjet)
        else:
            jtr = JetTree(rjet, gs=gs)

        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        jetm[0] = jtr.pseudojet().m()

        temp = jtr
        while temp.has_structure():
            z[depth[0]] = temp.z()
            delta[depth[0]] = temp.delta()
            kperp[depth[0]] = temp.kperp()
            m[depth[0]] = temp.m()
            depth[0] += 1
            temp = temp.harder()

        temp = jtr
        temp.groom(groomer=sd, do_recursive_correction=True)
        jetmg[0] = temp.pseudojet().m()
        while temp.has_structure():
            depthg[0] += 1
            temp = temp.harder()
        trics.Fill()

    dict_input, des_input = rd.next_event()
    dict_bkg, des_dkg = rd_bkg.next_event()
    nevent += 1
    if nevent % 100 == 0:
        print('%d events completed!' % nevent)

output.Write()
output.Close()
