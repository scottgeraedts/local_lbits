#try method corresponding to rahul's May 2nd email

import numpy as np
import local_library as ll
import scipy.linalg as spl
import time
import cProfile
import random
import sys
sys.path.append("/home/sgeraedt/TenPy/")
from algorithms.linalg import np_conserved as npc
np.set_printoptions(suppress=True,linewidth=150,precision=5,threshold=20000)


def make_operators(operators,v2,Hclass):
	Op_basis_size=operators.shape[0]
	Op=[]
	for j in range(Op_basis_size): Op.append(Hclass.project(operators[j],v2))
	K=np.zeros((Op_basis_size,Op_basis_size))
	for m in range(Op_basis_size):
		for n in range(Op_basis_size):
			if(Hclass.conserve):
				K[n,m]=npc.tensordot(Op[n],Op[m],([0,1],[0,1])).real
			else:
				K[n,m]=np.tensordot(Op[n],Op[m],([0,1],[0,1])).real
	
	for m in range(Op_basis_size):
		K[:,m]=K[:,m]
	kw,kv=spl.eigh(K)

	#make Op a set of orthogonal vectors that might not be local
	
#	for j in range(Op_basis_size): Op[j]=Hclass.to_ndarray(Op[j])
	Op=np.array(Op)
	Op=np.tensordot(kv,Op,([0],[0]))
#	print Op
	
	return Op,kw[-1]

kwfile=open("kw","w")

Ns=[10]
h=float(sys.argv[1])
ROD=40
#t=time.clock()
print_times=False
rfile=open("r","a")
for N in Ns:
	if(print_times): t=time.clock()
	Hclass=ll.Hamiltonian(N,pbc=True,conserve=True,level=2)
	if(print_times): print "time to init",time.clock()-t
	Ntruncs=[2**(N-5),2**(N-4),2**(N-3),2**(N-2)]
	#Ntruncs=[2**(N-2)]
	Mout=np.zeros([len(Ntruncs),N])
	newMout=np.zeros([len(Ntruncs),N])
#	Mout=np.zeros([ROD,N])
	failures=np.zeros(len(Ntruncs))

	r=0
	for seed in range(ROD):
		if(print_times): t=time.clock()
		Hclass.MPO_construction(h,seed)
		r+=Hclass.print_r()
		if(print_times): print "time to make and solve H",time.clock()-t

#		if(Hclass.conserve): v=Hclass.to_ndarray(Hclass.v)
#		v=Hclass.v

		for t_ind,Ntrunc in enumerate(Ntruncs):
			if(print_times): t=time.clock()
			try:
				v2=Hclass.truncate(Hclass.v.copy(),Ntrunc)
			except:
				failures[t_ind]+=1
				
			if(print_times): print "time to truncate",time.clock()-t	
	#		v2=np.dot(v2,v2.transpose())
#			if(Hclass.conserve):
#				q_flat=ll.make_q_flat(N)
#				perm,v2= npc.array.from_ndarray_flat(v2,[q_flat,q_flat],bunch=True,sort=True,q_conj=[1,-1])
			M=np.zeros([N,N])
			M2=np.zeros([N,N])
			best_ops=[]
			for i in range(N):
				if(print_times): t=time.clock()
				tempops,kw=make_operators(Hclass.twobody_ops[i],v2,Hclass)
				print >>kwfile,N,Ntrunc,i,kw
				best_ops.append(tempops[-1])
				if(print_times): print "time to find the best operators",time.clock()-t
	#			print tempops

			best_ops=np.array(best_ops)
			diagops=[np.diagonal(best_op.to_ndarray()) for best_op in best_ops]
			#compute the Mij
			if(print_times): ti=time.clock()
		
			for i in range(N):
				for j in range(N): 
					if(Hclass.conserve):
						M[i,j-i]=M[i,j-i]+npc.tensordot(best_ops[i],best_ops[j],[(0,1),(1,0)]).real
						M2[i,j-i]=M2[i,j-i]+np.sum(np.diagonal(best_ops[i].to_ndarray())*np.diagonal(best_ops[j].to_ndarray()))
						#M2[i,j-i]+=np.dot(diagops[i],diagops[j])
					else:
						M[i,j-i]=M[i,j-i]+np.tensordot(best_ops[i],best_ops[j],[(0,1),(1,0)]).real
						M2[i,j-i]=M2[i,j-i]+np.sum(np.diagonal(best_ops[i])*np.diagonal(best_ops[j]))
			Mout[t_ind,:]=Mout[t_ind,:]+np.mean(abs(M),axis=0)
			newMout[t_ind,:]=newMout[t_ind,:]+np.mean(abs(M2),axis=0)
			if(print_times): print "time to get the Ms",time.clock()-ti
	for t_ind,failure in enumerate(failures):
		Mout[t_ind,:]=Mout[t_ind,:]/(ROD-failure)
		newMout[t_ind,:]/=(ROD-failure)
	print "failures at N=",":",failures	
	print >>rfile,h,r/ROD
	rfile.close()
	np.savetxt("M"+str(N),Mout.transpose())
	np.savetxt("newM"+str(N),newMout.transpose())
kwfile.close()	
