#try method corresponding to rahul's May 2nd email
import numpy as np
import scipy.linalg as spl
import time
import cProfile
import random
import sys
sys.path.append("/home/sgeraedt/TenPy/")
sys.path.append("/home/geraedts/TenPy/")
sys.path.append("/home/geraedts/local_lbits/local_lbits/")
from algorithms.linalg import np_conserved as npc
import local_library as ll
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

def commuting_lambda(operators,v2,Hclass,diagop=True):
	lam=[]
	for i,op in enumerate(operators):
		if(diagop): op=np.diag(op)
		else: op=op.to_ndarray()
		outop=np.zeros(op.shape)
		for Hop in Hclass.twobody_ops[i]:
			Hop=Hclass.project(Hop,v2)
			Hop=Hop.to_ndarray()
			outop+=np.tensordot(op,Hop,([0,1],[0,1]))*Hop
		lam.append(spl.norm(outop)/spl.norm(op))
	return lam
Ns=[4,6,8]
h=float(sys.argv[1])
ROD=40
#t=time.time()
print_times=False

newkwfile=open("newkw","w")

for N in Ns:
	kwfile=open("kw"+str(N),"w")
	if(print_times): t=time.time()
	Hclass=ll.Hamiltonian(N,pbc=True,conserve=True,level=5)
	print "built Hamiltonian"
	if(print_times): print "time to init",time.time()-t
	Ntruncs=[2**(N-5),2**(N-4),2**(N-3),2**(N-2)]
	#Ntruncs=[2**(N-2)]
	Mout=np.zeros([len(Ntruncs),N])
	newMout=np.zeros([len(Ntruncs),N])
#	Mout=np.zeros([ROD,N])
	failures=np.zeros(len(Ntruncs))

	newkw=np.zeros(len(Ntruncs))
	for seed in range(ROD):
		if(print_times): t=time.time()
		Hclass.MPO_construction(h,seed)
		if(print_times): print  "time to make and solve H",time.time()-t

#		if(Hclass.conserve): v=Hclass.to_ndarray(Hclass.v)
#		v=Hclass.v

		for t_ind,Ntrunc in enumerate(Ntruncs):
			if(print_times): t=time.time()
			try:
				v2=Hclass.truncate(Hclass.v.copy(),Ntrunc)
			except:
				failures[t_ind]+=1
				
			if(print_times): print "time to truncate",time.time()-t	
	#		v2=np.dot(v2,v2.transpose())
#			if(Hclass.conserve):
#				q_flat=ll.make_q_flat(N)
#				perm,v2= npc.array.from_ndarray_flat(v2,[q_flat,q_flat],bunch=True,sort=True,q_conj=[1,-1])
			M=np.zeros([N,N])
			M2=np.zeros([N,N])
			best_ops=[]
			for i in range(N):
				if(print_times): t=time.time()
				tempops,kw=make_operators(Hclass.twobody_ops[i],v2,Hclass)
				print >>kwfile,N,Ntrunc,i,kw
				best_ops.append(tempops[-1])
				if(print_times): print "time to find the best operators",time.time()-t
	#			print tempops

			best_ops=np.array(best_ops)
			diagops=[np.diagonal(best_op.to_ndarray()) for best_op in best_ops]
			#compute the Mij
			if(print_times): ti=time.time()
#			print commuting_lambda(best_ops,v2,Hclass,False)
			newkw[t_ind]+=np.mean(commuting_lambda(diagops,v2,Hclass))
		
			for i in range(N):
				for j in range(N): 
					if(Hclass.conserve):
						M[i,j-i]=M[i,j-i]+npc.tensordot(best_ops[i],best_ops[j],[(0,1),(1,0)]).real
						#M2[i,j-i]=M2[i,j-i]+np.sum(np.diagonal(best_ops[i].to_ndarray())*np.diagonal(best_ops[j].to_ndarray()))
						M2[i,j-i]+=np.dot(diagops[i],diagops[j])
					else:
						M[i,j-i]=M[i,j-i]+np.tensordot(best_ops[i],best_ops[j],[(0,1),(1,0)]).real
						M2[i,j-i]=M2[i,j-i]+np.sum(np.diagonal(best_ops[i])*np.diagonal(best_ops[j]))
			Mout[t_ind,:]=Mout[t_ind,:]+np.mean(abs(M),axis=0)
			newMout[t_ind,:]=newMout[t_ind,:]+np.mean(abs(M2),axis=0)
			if(print_times): print  "time to get the Ms",time.time()-ti
	
	for t_ind,failure in enumerate(failures):
		Mout[t_ind,:]=Mout[t_ind,:]/(ROD-failure)
		newMout[t_ind,:]/=(ROD-failure)
	print "failures at N=",":",failures	
	print >>rfile,h,r/ROD
	rfile.close()
	np.savetxt("M"+str(N),Mout.transpose())
	np.savetxt("newM"+str(N),newMout.transpose())
	
	print >>newkwfile,N,
	for i in range(len(newkw)):
		print >>newkwfile,newkw[i]/(ROD-failure),
	print >>newkwfile
	kwfile.close()	
newkwfile.close
