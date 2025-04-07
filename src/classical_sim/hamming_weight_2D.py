import numpy as np
import scipy

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford, Clifford
from qiskit.circuit.library import ZGate, RZGate
from qiskit_aer import StatevectorSimulator

import quimb.tensor as qtn
from itertools import combinations
from numba import njit,int64
from numba.typed.typedlist import List
from numba.typed import Dict
import matplotlib.pyplot as plt

def list_to_int(x):
    return sum([x[i]*2**i for i in range(len(x))])

def int_to_str(x,b=2,m=4):
    ans=[]
    while x>0:
        ans.append(str(x%b))
        x//=b
    while len(ans)<m:
        ans.append('0')
    # n=len(ans)
    # for i in range(n//2):
    #     t=ans[i]
    #     ans[i]=ans[n-i-1]
    #     ans[n-i-1]=t
    return ''.join(ans)

def hamming_ball(n,r,x):
    ans=[]
    for i in range(r+1):
        for comb in combinations(list(range(n)),i):
            y=x
            for idx in comb:
                y^=(1<<idx)
            ans.append(y)
    return np.array(ans,dtype=np.int64)

def generate_RQC_gate_sequence(R,C,d):
    gs=[]
    gs_raw=[]
    while d>0:
        for offset in range(0,min(d,2,C-1)):
            for r in range(R):
                for c in range(offset,C-1,2):
                    gs.append([random_clifford(2),[r*C+c,r*C+(c+1)%C]])
                    gs_raw.append([gs[-1][0].to_matrix(),[r*C+c,r*C+(c+1)%C]])
            d-=1
        for offset in range(0,min(d,2,R-1)):
            for c in range(C):
                for r in range(offset,R-1,2):
                    gs.append([random_clifford(2),[r*C+c,((r+1)%R)*C+c]])
                    gs_raw.append([gs[-1][0].to_matrix(),[r*C+c,((r+1)%R)*C+c]])
            d-=1
    return gs,gs_raw

def U_to_U_dagger_P_U(U,R,C,is_raw):
    V=[]
    for gate,idx in U:
        V.append([gate if is_raw else gate.to_instruction(),idx])
    for r in range(R):
        for c in range(C):
            V.append([ZGate().to_matrix() if is_raw else ZGate(),[r*C+c]])
    for gate,idx in reversed(U):
        V.append([gate.conj().T if is_raw else gate.adjoint().to_instruction(),idx])
    return V

def circuit_from_gate_sequence(gate_sequence,R,C,is_raw):
    qc=QuantumCircuit(R*C)
    for gate,idx in gate_sequence:
        if is_raw:
            qc.unitary(gate,idx)
        else:
            qc.append(gate,idx)
    return qc

def insert_Z_rotations(U,theta):
    ans=[]
    for gate,idx in U:
        ans.append([gate,idx])
        ans.append([RZGate(theta).to_matrix(),[idx[0]]])
        ans.append([RZGate(theta).to_matrix(),[idx[1]]])
    return ans

def compute_H_terms(V,R,C,bounds):
    nn=R*C
    terms=[]
    mapping=[]
    p00=np.array([[1,0],[0,0]])
    for k in range(nn):
        min_R=bounds[k][0][0]
        max_R=bounds[k][0][1]+1
        min_C=bounds[k][1][0]
        max_C=bounds[k][1][1]+1
        tensors=[]
        label=[0 for i in range(nn)]
        for gate,idx in V:
            if len(idx)==2:
                i,j=idx
                if min_R<=i//C<max_R and min_C<=i%C<max_C and min_R<=j//C<max_R and min_C<=j%C<=max_C:
                    tensors.append(qtn.Tensor(gate.reshape((2,2,2,2)),
                                              inds=[".".join([str(i),str(label[i]+1)]),'.'.join([str(j),str(label[j]+1)]),
                                                    '.'.join([str(i),str(label[i])]),'.'.join([str(j),str(label[j])])]))
                    label[i]+=1
                    label[j]+=1
            else:
                i=idx[0]
                if min_R<=i//C<max_R and min_C<=i%C<max_C:
                    tensors.append(qtn.Tensor(gate.reshape((2,2)),inds=['.'.join([str(i),str(label[i]+1)]),'.'.join([str(i),str(label[i])])]))
                    label[i]+=1
        tensors.append(qtn.Tensor(p00,inds=['.'.join([str(k),str(label[k]+1)]),'.'.join([str(k),str(label[k])])]))  # type: ignore
        label[k]+=1
        for gate,idx in V:
            if len(idx)==2:
                i,j=idx
                if min_R<=i//C<max_R and min_C<=i%C<max_C and min_R<=j//C<max_R and min_C<=j%C<=max_C:
                    tensors.append(qtn.Tensor(gate.reshape((2,2,2,2)),
                                              inds=[".".join([str(i),str(label[i]+1)]),'.'.join([str(j),str(label[j]+1)]),
                                                    '.'.join([str(i),str(label[i])]),'.'.join([str(j),str(label[j])])]))
                    label[i]+=1
                    label[j]+=1
            else:
                i=idx[0]
                if min_R<=i//C<max_R and min_C<=i%C<max_C:
                    tensors.append(qtn.Tensor(gate.reshape((2,2)),inds=['.'.join([str(i),str(label[i]+1)]),'.'.join([str(i),str(label[i])])]))
                    label[i]+=1
        TN=tensors[0]
        for t in tensors[1:]:
            TN&=t
        TN=TN.contract()
        mapping.append([int(x.split(".")[0]) for x in TN.inds])
        terms.append(TN.data/nn)
    return mapping,terms

def compute_lightcone(V,R,C):
    ans=[]
    for j in range(R*C):
        lightcone=set([j])
        for gate,idx in V:
            if len(idx)==2:
                if idx[0] in lightcone or idx[1] in lightcone:
                    lightcone.add(idx[0])
                    lightcone.add(idx[1])
        lightcone=np.array(list(lightcone))
        rs=lightcone//C
        cs=lightcone%C
        ans.append([[min(rs),max(rs)],[min(cs),max(cs)]])
    return ans

def get_mask(R,C,min_R,max_R,min_C,max_C):
    mask=0
    for i in range(R*C-1,-1,-1):
        mask<<=1
        if min_R<=i//C<max_R and min_C<=i%C<max_C:
            mask|=1
    return mask

@njit
def compute_G_j(row,col,data,idx,term,dim,hb,hb_reverse,R,C,min_R,max_R,min_C,max_C,mask):
    N=(max_R-min_R)*(max_C-min_C)
    bucket=List()
    uncollapse=np.zeros(2**N,dtype=np.int64)
    for i in range(2**N):
        bucket.append(set([-1]))
    for i in range(dim):
        mid=hb[i]&mask
        collapsed_mid=0
        x=mid
        pos=0
        for j in range(R*C):
            if min_R<=j//C<max_R and min_C<=j%C<max_C:
                collapsed_mid=collapsed_mid|((x&1)<<pos)
                pos+=1
            x>>=1
        bucket[collapsed_mid].add(hb[i]-mid)
        uncollapse[collapsed_mid]=mid
    for i in range(2**N):
        bucket[i].remove(-1)
    for y in range(2**N):
        for x in range(y+1):
            if abs(term[y,x])>1e-15:
                for other in bucket[y]:
                    if other in bucket[x]:
                        bra=other+uncollapse[y]
                        ket=other+uncollapse[x]
                        i=hb_reverse[bra]
                        k=hb_reverse[ket]
                        if i>=k:
                            row[idx]=i
                            col[idx]=k
                            data[idx]=term[y,x]
                        else:
                            row[idx]=k
                            col[idx]=i
                            data[idx]=term[x,y]
                        if i==k:
                            data[idx]/=2
                        idx+=1
    return idx

def compute_zV0(n,V,z):
    qc=qtn.circuit.Circuit(n)
    for g in V:
        qc.apply_gate_raw(g[0],g[1])
    return abs(qc.amplitude(z))  # type: ignore

def reverse_permute(perm):
    perm_sorted=sorted(perm)
    reduced_perm=[perm_sorted.index(x) for x in perm]
    nn=len(perm)-1
    return np.array([nn-x for x in reduced_perm])

def hamming_weight_simulation(U,R,C,max_idx,theta,W):
    l1_diff_limit=2
    nn=R*C
    ZU=insert_Z_rotations(U,theta)
    ZV=U_to_U_dagger_P_U(ZU,R,C,True)
    p_psi = np.empty(0)
    if nn<=l1_diff_limit:
        backend=StatevectorSimulator()
        qc=circuit_from_gate_sequence(ZV,R,C,True)
        result=backend.run(transpile(qc, backend)).result()
        psi=np.array(result.get_statevector(qc))
        p_psi=np.array([abs(x)**2 for x in psi])
    for g in ZV:
        g[1]=list(reversed(g[1]))
    zV0=compute_zV0(nn,ZV,int_to_str(max_idx,2,nn))
    print(zV0,flush=True)
    max_non_zero=4*10**8
    bounds=compute_lightcone(ZV,R,C)
    mapping,terms=compute_H_terms(ZV,R,C,bounds)
    hb=hamming_ball(nn,W,max_idx)
    dim=len(hb)
    print(dim,flush=True)
    
    hb_reverse=Dict.empty(key_type=int64,value_type=int64)
    for i in range(len(hb)):
        hb_reverse[hb[i]]=i
    
    row=np.zeros(max_non_zero,dtype=np.int64)
    col=np.zeros(max_non_zero,dtype=np.int64)
    data=np.zeros(max_non_zero,dtype=complex)
    idx=0
    for j in range(nn):
        min_R=bounds[j][0][0]
        max_R=bounds[j][0][1]+1
        min_C=bounds[j][1][0]
        max_C=bounds[j][1][1]+1
        N=(max_R-min_R)*(max_C-min_C)
        term=np.moveaxis(terms[j],list(range(N)),reverse_permute(mapping[j][:N]))
        term=np.moveaxis(term,list(range(N,2*N)),reverse_permute(mapping[j][N:])+N).reshape(1<<N,1<<N)
        mask=get_mask(R,C,min_R,max_R,min_C,max_C)
        idx=compute_G_j(row,col,data,idx,term,dim,hb,hb_reverse,R,C,min_R,max_R,min_C,max_C,mask)
    
    G=scipy.sparse.csr_array((data,(row,col)),shape=(dim,dim),dtype=complex)
    G=G+G.conj().T
    
    su,sv=scipy.sparse.linalg.eigs(G,k=1)
    print(G.nnz,flush=True)
    print(su[0].real,flush=True)
    sphi=sv[:,-1]
    if nn<=l1_diff_limit:
        sp_phi=np.array([abs(x)**2 for x in sphi])
        l1_diff=np.linalg.norm(sp_phi-p_psi[hb],1)+np.linalg.norm(p_psi[list(set(range(2**nn)).difference(hb))],1)
        print(l1_diff)
        return su[0].real,l1_diff,np.sqrt(sp_phi[0]),zV0
    
    hb1=hamming_ball(nn,W-1,max_idx)
    return su[0].real,abs(sphi[0]),abs(sphi[0])*np.linalg.norm(sphi[:len(hb1)]),zV0

def experiment_n_W(RC_list,d,theta,W_list):
    largest_eig_list=[[] for i in range(len(W_list))]
    estimated_peakedness_list=[[] for i in range(len(W_list))]
    normalized_list=[[] for i in range(len(W_list))]
    peakedness_list=[]
    for R,C in RC_list:
        U,U_raw=generate_RQC_gate_sequence(R,C,d)
        V=U_to_U_dagger_P_U(U,R,C,False)
        qc=circuit_from_gate_sequence(V,R,C,False)
        max_arr=Clifford(qc).phase[R*C:]
        max_idx=list_to_int(max_arr)
        for i in range(len(W_list)):
            largest_eig,estimated_peakedness,normalized,peakedness=hamming_weight_simulation(U_raw,R,C,max_idx,theta,W_list[i])
            largest_eig_list[i].append(largest_eig)
            estimated_peakedness_list[i].append(estimated_peakedness)
            normalized_list[i].append(normalized)
            if i==0:
                peakedness_list.append(peakedness)
    
    n_list=[R*C for R,C in RC_list]
    np.save("%d_%.2f_%s.npy"%(d,theta,W_list),{"n_list":n_list,"peakedness_list":peakedness_list,
             "estimated_peakedness_list":estimated_peakedness_list,"largest_eig_list":largest_eig_list,
             "normalized_list":normalized_list})  # type: ignore

def plot_together(d,theta,W_list,square):
    data=np.load("%d_%.2f_%s.npy"%(d,theta,W_list),allow_pickle=True)[()]
    n_list=data["n_list"]
    peakedness_list=data["peakedness_list"]
    estimated_peakedness_list=data["estimated_peakedness_list"]
    largest_eig_list=data["largest_eig_list"]
    # normalized_list=data["normalized_list"]
    
    colour_cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig,axs=plt.subplots(2,figsize=(8,10))
    axs[0].set_title(r"2D Random Clifford Circuit with $R(\theta)$ Rotations, $d$=%d, $\theta$=%.2f"%(d,theta),fontsize=14)
    axs[0].set_xticks(n_list)
    if square:
        axs[0].set_ylabel(r"$|\langle z|V(\theta)|0^n\rangle|^2$",fontsize=12)
    else:
        axs[0].set_ylabel(r"$|\langle z|V(\theta)|0^n\rangle|$",fontsize=12)
    if square:
        peakedness_list=[x**2 for x in peakedness_list]
    axs[0].plot(n_list,peakedness_list,marker='x',linestyle='',label="Exact")
    for i in range(len(W_list)):
        if square:
            estimated_peakedness_list[i]=[x**2 for x in estimated_peakedness_list[i]]
        axs[0].plot(n_list,estimated_peakedness_list[i],marker='x',linestyle='',color=colour_cycle[i+1],label=r"Estimated $W$=%d"%W_list[i])
        # axs[0].plot(n_list,normalized_list[i],color=colour_cycle[i+1],linestyle='dashed')
    axs[0].legend(fontsize=12)
    
    axs[1].set_xticks(n_list)
    axs[1].set_ylabel(r"$\lambda_1(G)$",fontsize=12)
    axs[1].set_xlabel(r"$n$",fontsize=16)
    for i in range(len(W_list)):
        axs[1].plot(n_list,largest_eig_list[i],marker='x',linestyle='',color=colour_cycle[i+1],label=r"$W$=%d"%W_list[i])
    axs[1].legend(fontsize=12)
    
    if square:
        plt.savefig("together_n_list_w_list_%d_%.2f_square.png"%(d,theta),bbox_inches='tight',dpi=300)
    else:
        plt.savefig("together_n_list_w_list_%d_%.2f.png"%(d,theta),bbox_inches='tight',dpi=300)
    plt.show()

def plot_separate(d,theta,W_list,square):
    data=np.load("%d_%.2f_%s.npy"%(d,theta,W_list),allow_pickle=True)[()]
    n_list=data["n_list"]
    peakedness_list=data["peakedness_list"]
    estimated_peakedness_list=data["estimated_peakedness_list"]
    largest_eig_list=data["largest_eig_list"]
    # normalized_list=data["normalized_list"]
    
    colour_cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.title(r"2D Random Clifford Circuit with $R(\theta)$ Rotations, $d$=%d, $\theta$=%.2f"%(d,theta),fontsize=14)
    plt.xticks(n_list)
    if square:
        plt.ylabel(r"$|\langle z|V(\theta)|0^n\rangle|^2$",fontsize=12)
    else:
        plt.ylabel(r"$|\langle z|V(\theta)|0^n\rangle|$",fontsize=12)
    plt.xlabel(r"$n$",fontsize=16)
    if square:
        peakedness_list=[x**2 for x in peakedness_list]
    plt.plot(n_list,peakedness_list,marker='x',linestyle='',label="Exact")
    for i in range(len(W_list)):
        if square:
            estimated_peakedness_list[i]=[x**2 for x in estimated_peakedness_list[i]]
        plt.plot(n_list,estimated_peakedness_list[i],marker='x',linestyle='',color=colour_cycle[i+1],label=r"Estimated $W$=%d"%W_list[i])
        # axs[0].plot(n_list,normalized_list[i],color=colour_cycle[i+1],linestyle='dashed')
    plt.legend(fontsize=12)
    if square:
        plt.savefig("zV0_n_list_w_list_%d_%.2f_square.png"%(d,theta),bbox_inches='tight',dpi=300)
    else:
        plt.savefig("zV0_n_list_w_list_%d_%.2f.png"%(d,theta),bbox_inches='tight',dpi=300)
    plt.show()
    
    plt.title(r"2D Random Clifford Circuit with $R(\theta)$ Rotations, $d$=%d, $\theta$=%.2f"%(d,theta),fontsize=14)
    plt.xticks(n_list)
    plt.ylabel(r"$\lambda_1(G)$",fontsize=12)
    plt.xlabel(r"$n$",fontsize=16)
    for i in range(len(W_list)):
        plt.plot(n_list,largest_eig_list[i],marker='x',linestyle='',color=colour_cycle[i+1],label=r"$W$=%d"%W_list[i])
    plt.legend(fontsize=12)
    if square:
        plt.savefig("lambda1_n_list_w_list_%d_%.2f_square.png"%(d,theta),bbox_inches='tight',dpi=300)
    else:
        plt.savefig("lambda1_n_list_w_list_%d_%.2f.png"%(d,theta),bbox_inches='tight',dpi=300)
    plt.show()
    
def go():
    RC_list=[[5,6],[6,6],[7,6],[6,8],[7,8]]
    d=3
    W_list=[3,4,5]
    W_list=[4,5,6]
    theta=0.2
    experiment_n_W(RC_list,d,theta,W_list)
    plot_together(d,theta,W_list,True)
    plot_separate(d,theta,W_list,True)
    
go()
