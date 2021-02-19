# root python module
from ROOT import TFile, TTree, TCanvas
from array import array
import math
import sys





try:
    target = sys.argv[1]
    inputs = sys.argv[2:]
except:
    target = './resolution.root'
    inputs = ['../default.root']

print('target: %s' % target)
print('inputs:' + ' '.join(inputs))

#output
of = TFile(target, 'recreate')
otr_csj = TTree('csjjet', 'csjjet')
otr_cse = TTree('csejet', 'csejet')
otr_ics = TTree('icsjet', 'icsjet')

jetpt = array('f', [0.])
jetptm = array('f', [0.])

jetm = array('f', [0.])
jetmm = array('f', [0.])

jetmg = array('f', [0.])
jetmgm = array('f', [0.])

scale = array('f', [0.])

deta = array('f', [0.])
dphi = array('f', [0.])

# sub structure
depth = array('i', [0])
depthg = array('i', [0])
z = array('f', 40*[0.])
delta = array('f', 40*[0.])
kperp = array('f', 40*[0.])
m = array('f', 40*[0.])

# matched sub structure
depthm = array('i', [0])
depthgm = array('i', [0])
zm = array('f', 40*[0.])
deltam = array('f', 40*[0.])
kperpm = array('f', 40*[0.])
mm = array('f', 40*[0.])

# constituents
cn = array('i', [0])
cpt = array('f', 500 * [0.])
cdeta = array('f', 500 * [0.])
cdphi = array('f', 500 * [0.])
cnm = array('i', [0])
cptm = array('f', 500 * [0.])
cdetam = array('f', 500 * [0.])
cdphim = array('f', 500 * [0.])

# branches
otr_csj.Branch('jetpt', jetpt, 'jetpt/F')
otr_csj.Branch('jetptm', jetptm, 'jetptm/F')
otr_csj.Branch('jetm', jetm, 'jetm/F')
otr_csj.Branch('jetmm', jetmm, 'jetmm/F')
otr_csj.Branch('jetmg', jetmg, 'jetmg/F')
otr_csj.Branch('jetmgm', jetmgm, 'jetmgm/F')
otr_csj.Branch('scale', scale, 'scale/F')
otr_csj.Branch('deta', deta, 'deta/F')
otr_csj.Branch('dphi', dphi, 'dphi/F')
otr_csj.Branch('depth', depth, 'depth/I')
otr_csj.Branch('depthg', depthg, 'depthg/I')
otr_csj.Branch('z', z, 'z[depth]/F')
otr_csj.Branch('delta', delta, 'delta[depth]/F')
otr_csj.Branch('kperp', kperp, 'kperp[depth]/F')
otr_csj.Branch('m', m, 'm[depth]/F')
otr_csj.Branch('depthm', depthm, 'depthm/I')
otr_csj.Branch('depthgm', depthgm, 'depthgm/I')
otr_csj.Branch('zm', zm, 'zm[depthm]/F')
otr_csj.Branch('deltam', deltam, 'deltam[depthm]/F')
otr_csj.Branch('kperpm', kperpm, 'kperpm[depthm]/F')
otr_csj.Branch('mm', m, 'mm[depthm]/F')
# constituents
otr_csj.Branch('cn', cn, 'cn/I')
otr_csj.Branch('cpt', cpt, 'cpt[cn]/F')
otr_csj.Branch('cdeta', cdeta, 'cdeta[cn]/F')
otr_csj.Branch('cdphi', cdphi, 'cdphi[cn]/F')
otr_csj.Branch('cnm', cnm, 'cnm/I')
otr_csj.Branch('cptm', cptm, 'cptm[cnm]/F')
otr_csj.Branch('cdetam', cdetam, 'cdetam[cn]/F')
otr_csj.Branch('cdphim', cdphim, 'cdphim[cn]/F')

otr_cse.Branch('jetpt', jetpt, 'jetpt/F')
otr_cse.Branch('jetptm', jetptm, 'jetptm/F')
otr_cse.Branch('jetm', jetm, 'jetm/F')
otr_cse.Branch('jetmm', jetmm, 'jetmm/F')
otr_cse.Branch('jetmg', jetmg, 'jetmg/F')
otr_cse.Branch('jetmgm', jetmgm, 'jetmgm/F')
otr_cse.Branch('scale', scale, 'scale/F')
otr_cse.Branch('deta', deta, 'deta/F')
otr_cse.Branch('dphi', dphi, 'dphi/F')
otr_cse.Branch('depth', depth, 'depth/I')
otr_cse.Branch('depthg', depthg, 'depthg/I')
otr_cse.Branch('z', z, 'z[depth]/F')
otr_cse.Branch('delta', delta, 'delta[depth]/F')
otr_cse.Branch('kperp', kperp, 'kperp[depth]/F')
otr_cse.Branch('m', m, 'm[depth]/F')
otr_cse.Branch('depthm', depthm, 'depthm/I')
otr_cse.Branch('depthgm', depthgm, 'depthgm/I')
otr_cse.Branch('zm', zm, 'zm[depthm]/F')
otr_cse.Branch('deltam', deltam, 'deltam[depthm]/F')
otr_cse.Branch('kperpm', kperpm, 'kperpm[depthm]/F')
otr_cse.Branch('mm', m, 'mm[depthm]/F')
# constituents
otr_cse.Branch('cn', cn, 'cn/I')
otr_cse.Branch('cpt', cpt, 'cpt[cn]/F')
otr_cse.Branch('cdeta', cdeta, 'cdeta[cn]/F')
otr_cse.Branch('cdphi', cdphi, 'cdphi[cn]/F')
otr_cse.Branch('cnm', cnm, 'cnm/I')
otr_cse.Branch('cptm', cptm, 'cptm[cnm]/F')
otr_cse.Branch('cdetam', cdetam, 'cdetam[cn]/F')
otr_cse.Branch('cdphim', cdphim, 'cdphim[cn]/F')

otr_ics.Branch('jetpt', jetpt, 'jetpt/F')
otr_ics.Branch('jetptm', jetptm, 'jetptm/F')
otr_ics.Branch('jetm', jetm, 'jetm/F')
otr_ics.Branch('jetmm', jetmm, 'jetmm/F')
otr_ics.Branch('jetmg', jetmg, 'jetmg/F')
otr_ics.Branch('jetmgm', jetmgm, 'jetmgm/F')
otr_ics.Branch('scale', scale, 'scale/F')
otr_ics.Branch('deta', deta, 'deta/F')
otr_ics.Branch('dphi', dphi, 'dphi/F')
otr_ics.Branch('depth', depth, 'depth/I')
otr_ics.Branch('depthg', depthg, 'depthg/I')
otr_ics.Branch('z', z, 'z[depth]/F')
otr_ics.Branch('delta', delta, 'delta[depth]/F')
otr_ics.Branch('kperp', kperp, 'kperp[depth]/F')
otr_ics.Branch('m', m, 'm[depth]/F')
otr_ics.Branch('depthm', depthm, 'depthm/I')
otr_ics.Branch('depthgm', depthgm, 'depthgm/I')
otr_ics.Branch('zm', zm, 'zm[depthm]/F')
otr_ics.Branch('deltam', deltam, 'deltam[depthm]/F')
otr_ics.Branch('kperpm', kperpm, 'kperpm[depthm]/F')
otr_ics.Branch('mm', m, 'mm[depthm]/F')
# constituents
otr_ics.Branch('cn', cn, 'cn/I')
otr_ics.Branch('cpt', cpt, 'cpt[cn]/F')
otr_ics.Branch('cdeta', cdeta, 'cdeta[cn]/F')
otr_ics.Branch('cdphi', cdphi, 'cdphi[cn]/F')
otr_ics.Branch('cnm', cnm, 'cnm/I')
otr_ics.Branch('cptm', cptm, 'cptm[cnm]/F')
otr_ics.Branch('cdetam', cdetam, 'cdetam[cn]/F')
otr_ics.Branch('cdphim', cdphim, 'cdphim[cn]/F')


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

for input in inputs:
    f = TFile(input, 'READ')
    tr = f.Get('jet')
    tr_csj = f.Get('csjjet')
    tr_cse = f.Get('csejet')
    tr_ics = f.Get('icsjet')

    #matching
    idx = 0
    idx_csj = 0
    idx_cse = 0
    idx_ics = 0
    max = tr.GetEntriesFast()
    # max = 10

    while idx<max:
        # csj matching
        temp = match(tr, tr_csj, idx, idx_csj)
        if temp>=0:
            idx_csj = temp
            tr.GetEntry(idx)
            tr_csj.GetEntry(idx_csj)
            jetpt[0] = tr.jetpt
            jetptm[0] = tr_csj.jetpt
            jetm[0] = tr.jetm
            jetmm[0] = tr_csj.jetm
            jetmg[0] = tr.jetmg
            jetmgm[0] = tr_csj.jetmg
            deta[0] = tr_csj.eta - tr.eta
            dphi[0] = delta_phi(tr_csj.phi, tr.phi)
            scale[0] = jetptm[0]/jetpt[0]
            depth[0] = tr.depth
            depthm[0] = tr_csj.depth
            depthg[0] = tr.depthg
            depthgm[0] = tr_csj.depthg
            cn[0] = tr.cn
            cnm[0] = tr_csj.cn

            for i in range(depth[0]):
                z[i] = tr.z[i]
                delta[i] = tr.delta[i]
                kperp[i] = tr.kperp[i]
                m[i] = tr.m[i]

            for i in range(depthm[0]):
                zm[i] = tr_csj.z[i]
                deltam[i] = tr_csj.delta[i]
                kperpm[i] = tr_csj.kperp[i]
                mm[i] = tr_csj.m[i]

            for i in range(cn[0]):
                cpt[i] = tr.cpt[i]
                cdeta[i] = tr.cdeta[i]
                cdphi[i] = tr.cdphi[i]

            for i in range(cnm[0]):
                cptm[i] = tr_csj.cpt[i]
                cdetam[i] = tr_csj.cdeta[i]
                cdphim[i] = tr_csj.cdphi[i]

            otr_csj.Fill()

        # cse matching
        temp = match(tr, tr_cse, idx, idx_cse)
        if temp>=0:
            idx_cse = temp
            tr.GetEntry(idx)
            tr_cse.GetEntry(idx_cse)
            jetpt[0] = tr.jetpt
            jetptm[0] = tr_cse.jetpt
            jetm[0] = tr.jetm
            jetmm[0] = tr_cse.jetm
            jetmg[0] = tr.jetmg
            jetmgm[0] = tr_cse.jetmg
            deta[0] = tr_cse.eta - tr.eta
            dphi[0] = delta_phi(tr_cse.phi, tr.phi)
            scale[0] = jetptm[0]/jetpt[0]
            depth[0] = tr.depth
            depthm[0] = tr_cse.depth
            depthg[0] = tr.depthg
            depthgm[0] = tr_cse.depthg
            cn[0] = tr.cn
            cnm[0] = tr_cse.cn

            for i in range(depth[0]):
                z[i] = tr.z[i]
                delta[i] = tr.delta[i]
                kperp[i] = tr.kperp[i]
                m[i] = tr.m[i]

            for i in range(depthm[0]):
                zm[i] = tr_cse.z[i]
                deltam[i] = tr_cse.delta[i]
                kperpm[i] = tr_cse.kperp[i]
                mm[i] = tr_cse.m[i]

            for i in range(cn[0]):
                cpt[i] = tr.cpt[i]
                cdeta[i] = tr.cdeta[i]
                cdphi[i] = tr.cdphi[i]

            for i in range(cnm[0]):
                cptm[i] = tr_cse.cpt[i]
                cdetam[i] = tr_cse.cdeta[i]
                cdphim[i] = tr_cse.cdphi[i]

            otr_cse.Fill()

        # ics matching
        temp = match(tr, tr_ics, idx, idx_ics)
        if temp>=0:
            idx_ics = temp
            tr.GetEntry(idx)
            tr_ics.GetEntry(idx_ics)
            jetpt[0] = tr.jetpt
            jetptm[0] = tr_ics.jetpt
            jetm[0] = tr.jetm
            jetmm[0] = tr_cse.jetm
            jetmg[0] = tr.jetmg
            jetmgm[0] = tr_cse.jetmg
            deta[0] = tr_ics.eta - tr.eta
            dphi[0] = delta_phi(tr_ics.phi, tr.phi)
            scale[0] = jetptm[0]/jetpt[0]
            depth[0] = tr.depth
            depthm[0] = tr_ics.depth
            depthg[0] = tr.depthg
            depthgm[0] = tr_ics.depthg
            cn[0] = tr.cn
            cnm[0] = tr_ics.cn

            for i in range(depth[0]):
                z[i] = tr.z[i]
                delta[i] = tr.delta[i]
                kperp[i] = tr.kperp[i]
                m[i] = tr.m[i]

            for i in range(depthm[0]):
                zm[i] = tr_ics.z[i]
                deltam[i] = tr_ics.delta[i]
                kperpm[i] = tr_ics.kperp[i]
                mm[i] = tr_ics.m[i]

            for i in range(cn[0]):
                cpt[i] = tr.cpt[i]
                cdeta[i] = tr.cdeta[i]
                cdphi[i] = tr.cdphi[i]

            for i in range(cnm[0]):
                cptm[i] = tr_ics.cpt[i]
                cdetam[i] = tr_ics.cdeta[i]
                cdphim[i] = tr_ics.cdphi[i]

            otr_ics.Fill()

        idx+=1

of.Write()
of.Close()
