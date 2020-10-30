# root python module
from ROOT import TFile, TTree, TCanvas
from array import array
import math
import sys





try:
    input = sys.argv[1]
    output = sys.argv[2]
except:
    input = '../default.root'
    output = './resolution.root'

print('input file: %s' % input)
print('output file: %s' % output)

f = TFile(input, 'READ')
tr = f.Get('jet')
tr_cs = f.Get('csjet')
tr_ics = f.Get('icsjet')

#output
of = TFile(output, 'recreate')
otr_cs = TTree('csjet', 'csjet')
otr_ics = TTree('icsjet', 'icsjet')

jetpt = array('f', [0.])
jetptcs = array('f', [0.])


scale = array('f', [0.])
# res = array('f', [0.])

deta = array('f', [0.])
dphi = array('f', [0.])

# sub structure
depth = array('i', [0])
z = array('f', 40*[0.])
delta = array('f', 40*[0.])
kperp = array('f', 40*[0.])
m = array('f', 40*[0.])

# matched sub structure
depthm = array('i', [0])
zm = array('f', 40*[0.])
deltam = array('f', 40*[0.])
kperpm = array('f', 40*[0.])
mm = array('f', 40*[0.])

# sub structure matched

# branches
otr_cs.Branch('jetpt', jetpt, 'jetpt/F')
otr_cs.Branch('jetptcs', jetptcs, 'jetptcs/F')
otr_cs.Branch('scale', scale, 'scale/F')
# otr_cs.Branch('jer', jer, 'jer/F')
otr_cs.Branch('deta', deta, 'deta/F')
otr_cs.Branch('dphi', dphi, 'dphi/F')

otr_cs.Branch('depth', depth, 'depth/I')
otr_cs.Branch('z', z, 'z[depth]/F')
otr_cs.Branch('delta', delta, 'delta[depth]/F')
otr_cs.Branch('kperp', kperp, 'kperp[depth]/F')
otr_cs.Branch('m', m, 'm[depth]/F')

otr_cs.Branch('depthm', depthm, 'depthm/I')
otr_cs.Branch('zm', zm, 'zm[depthm]/F')
otr_cs.Branch('deltam', deltam, 'deltam[depthm]/F')
otr_cs.Branch('kperpm', kperpm, 'kperpm[depthm]/F')
otr_cs.Branch('mm', m, 'mm[depthm]/F')

otr_ics.Branch('jetpt', jetpt, 'jetpt/F')
otr_ics.Branch('jetptcs', jetptcs, 'jetptcs/F')
otr_ics.Branch('scale', scale, 'scale/F')
# otr_ics.Branch('jer', jer, 'jer/F')
otr_ics.Branch('deta', deta, 'deta/F')
otr_ics.Branch('dphi', dphi, 'dphi/F')

otr_ics.Branch('depth', depth, 'depth/I')
otr_ics.Branch('z', z, 'z[depth]/F')
otr_ics.Branch('delta', delta, 'delta[depth]/F')
otr_ics.Branch('kperp', kperp, 'kperp[depth]/F')
otr_ics.Branch('m', m, 'm[depth]/F')

otr_ics.Branch('depthm', depthm, 'depthm/I')
otr_ics.Branch('zm', zm, 'zm[depthm]/F')
otr_ics.Branch('deltam', deltam, 'deltam[depthm]/F')
otr_ics.Branch('kperpm', kperpm, 'kperpm[depthm]/F')
otr_ics.Branch('mm', m, 'mm[depthm]/F')


#matching
idx = 0
idx_cs = 0
idx_ics = 0
max = tr.GetEntriesFast()
# max = 10

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


def match(tr, tr_match, idx, idx_matched=0):
    tr.GetEntry(idx)
    max = tr_match.GetEntriesFast()
    if not tr.event:
        return -999
    while tr.event >= tr_match.event and idx_matched<max:
        tr_match.GetEntry(idx_matched)
        if tr.event==tr_match.event:
            eta1 = tr.eta
            phi1 = tr.phi
            eta2 = tr_match.eta
            phi2 = tr_match.phi
            deta = delta_eta(eta1, eta2)
            dphi = delta_phi(phi1, phi2)
            dr = math.sqrt(deta*deta + dphi*dphi)
            if dr<0.4:
                return idx_matched
        idx_matched += 1
    return -999

while idx<max:
    # cs matching

    temp = match(tr, tr_cs, idx, idx_cs)
    if temp>=0:
        idx_cs = temp
        # print('%d/%d'%(idx, idx_cs))

        tr.GetEntry(idx)
        tr_cs.GetEntry(idx_cs)
        jetpt[0] = tr.jetpt
        jetptcs[0] = tr_cs.jetpt
        deta[0] = tr_cs.eta - tr.eta
        dphi[0] = tr_cs.phi - tr.phi
        scale[0] = jetptcs[0]/jetpt[0]

        depth[0] = tr.depth
        # print(tr.depth)
        for i in range(depth[0]):
            z[i] = tr.z[i]
            delta[i] = tr.delta[i]
            kperp[i] = tr.kperp[i]
            m[i] = tr.m[i]

        depthm[0] = tr_cs.depth
        for i in range(depthm[0]):
            zm[i] = tr_cs.z[i]
            deltam[i] = tr_cs.delta[i]
            kperpm[i] = tr_cs.kperp[i]
            mm[i] = tr_cs.m[i]

        otr_cs.Fill()

    # ics matching
    temp = match(tr, tr_ics, idx, idx_ics)
    if temp>=0:
        idx_ics = temp
        # print('%d/%d'%(idx, idx_ics))

        tr.GetEntry(idx)
        tr_ics.GetEntry(idx_ics)
        jetpt[0] = tr.jetpt
        jetptcs[0] = tr_ics.jetpt
        deta[0] = tr_ics.eta - tr.eta
        dphi[0] = tr_ics.phi - tr.phi
        scale[0] = jetptcs[0]/jetpt[0]

        depth[0] = tr.depth
        for i in range(depth[0]):
            z[i] = tr.z[i]
            delta[i] = tr.delta[i]
            kperp[i] = tr.kperp[i]
            m[i] = tr.m[i]

        depthm[0] = tr_ics.depth
        for i in range(depthm[0]):
            zm[i] = tr_ics.z[i]
            deltam[i] = tr_ics.delta[i]
            kperpm[i] = tr_ics.kperp[i]
            mm[i] = tr_ics.m[i]

        otr_ics.Fill()

    idx+=1

of.Write()
of.Close()
