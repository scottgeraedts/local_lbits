#this code optimizes a local lbit to make it commute with the Hamiltonian
import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize
import random
import time
import sys
sys.path.append("/home/sgeraedt/TenPy/")
from algorithms.linalg import np_conserved as npc
np.set_printoptions(suppress=True,linewidth=150,precision=8)

#define pauli matrices

class Hamiltonian:
	def __init__(self,N,pbc=True,conserve=True,level=2):
		self.N=N
		self.pbc=pbc
		self.conserve=conserve
#		self.H_onebody=[]
#		self.H_twobody=np.zeros([pow(2,N),pow(2,N)],dtype=np.float)

		self.sx=sx=np.zeros([2,2],dtype=np.float)
		sx[0,1]=1; sx[1,0]=1;
		self.sy=sy=np.zeros([2,2],dtype=np.complex)
		sy[0,1]=1J; sy[1,0]=-1J;
		self.sz=sz=np.zeros([2,2],dtype=np.float)
		sz[0,0]=1; sz[1,1]=-1;
		self.Id=Id=np.eye(2,dtype=np.float)
		self.sp=sp=np.zeros([2,2],dtype=np.float)
		sp[0,1]=1;
		self.sm=sm=np.zeros([2,2],dtype=np.float)
		sm[1,0]=1;

		#make pbc term
		self.H_pbc=np.zeros([pow(2,N),pow(2,N)],dtype=np.float)
		if(pbc):
			op1=[0.25*sz,0.5*sp,0.5*sm]
			op2=[sz,sm,sp]

			link=np.zeros((1,2,2,1))
			link[0,0,0,0]=link[0,1,1,0]=1
			full_link=link
			for j in range(N-3):
				full_link=np.tensordot(full_link,link,([-1],[0]))
			for i,op in enumerate(op1):
				out=op.reshape((2,2,1))
				out=np.tensordot(out,full_link,([-1],[0]))
				out=np.tensordot(out,op2[i].reshape((1,2,2)),([-1],[0]))
				out=out.transpose(range(0,2*N,2)+range(1,2*N,2))
				out=out.reshape([2**N,2**N])

				self.H_pbc+=out	
#		if(pbc):
#			self.twobody_pbc=[]
#			for j,op in enumerate([self.sx,self.sy,self.sz]):
#				temp=0.25*op;
#				for i in range(self.N-2):
#					temp=spl.kron(temp,self.Id)
#				temp=spl.kron(temp,op)
#				self.H_pbc=self.H_pbc+temp.real
#				self.twobody_pbc.append(temp.real)

		if level!=-1:
			self.twobody_ops=[]
			for i in range(N): self.twobody_ops.append(self.make_twobody(i,level))

	def MPO_construction(self,h,seed=0,rans=None,alphax=0):
		t=time.clock()
		if rans is None:
			random.seed(seed)
			rans=[]
			ransx=[]
			for i in range(self.N): rans.append(random.uniform(-1,1))
			for i in range(self.N): ransx.append(random.uniform(-1,1))
		else: ransx=rans
		
		first=np.zeros((2,2,5),dtype=float)
		last=np.zeros((2,2,5),dtype=float)
		first[:,:,0]=self.Id
		first[:,:,1]=0.25*self.sz
		first[:,:,2]=0.5*self.sp
		first[:,:,3]=0.5*self.sm
		first[:,:,4]=0.5*h*(self.sz*rans[0]+alphax*self.sx*ransx[0])

		last[:,:,0]=0.5*h*(self.sz*rans[-1]+alphax*self.sx*ransx[-1])
		last[:,:,1]=self.sz
		last[:,:,2]=self.sm
		last[:,:,3]=self.sp
		last[:,:,4]=self.Id
		
		out=first
		for i in range(1,self.N-1):
			mid=np.zeros((2,2,5,5),dtype=float)
			
			mid[:,:,0,0]=self.Id
			mid[:,:,0,1]=0.25*self.sz
			mid[:,:,0,2]=0.5*self.sp
			mid[:,:,0,3]=0.5*self.sm
			mid[:,:,0,4]=0.5*h*(self.sz*rans[i]+alphax*self.sx*ransx[i])
			mid[:,:,1,4]=self.sz
			mid[:,:,2,4]=self.sm
			mid[:,:,3,4]=self.sp
			mid[:,:,4,4]=self.Id
			
			out=np.tensordot(mid,out,([2],[2]))

		
		out=np.tensordot(last,out,([2],[2]))
		out=out.transpose(range(0,2*self.N,2)+range(1,2*self.N,2))
		out=out.reshape((2**self.N,2**self.N))

		if(self.pbc): out=out+self.H_pbc
		
		if(self.conserve):
			q_flat=make_q_flat(self.N)
			self.perm,H= npc.array.from_ndarray_flat(out,[q_flat,q_flat],bunch=True,sort=True,q_conj=[1,-1])
			self.H=H
			self.w,self.v=npc.eigh(H)
			
			self.energy_keys=np.argsort(self.w)
			self.energy_keys=np.argsort(self.energy_keys)
		else:
			self.H=out
			self.w,self.v=spl.eigh(out)
			
	def to_ndarray(self,x):
		if  not self.conserve: return x
		v=x.to_ndarray()
		if self.perm is not None: 
			keys=np.argsort(self.perm[0])
			v=v[keys,:]
			v=v[:,keys]
		return v
	
	def make_twobody(self,i,level):
		N=self.N
		Id=np.zeros((2,2,1,1))
		Id[0,0,0,0]=Id[1,1,0,0]=1/np.sqrt(2)
		Sz=np.zeros((2,2,1,1))
		Sz[0,0,0,0]=1/np.sqrt(2); Sz[1,1,0,0]=-1/np.sqrt(2)
		Sp=np.zeros((2,2,1,1))
		Sp[0,1,0,0]=1
		Sm=np.zeros((2,2,1,1))
		Sm[1,0,0,0]=1
		Sx=np.zeros((2,2,1,1))
		Sx[:,:,0,0]=self.sx/np.sqrt(2)
		Sy=np.zeros((2,2,1,1),dtype=complex)
		Sy[:,:,0,0]=self.sy/np.sqrt(2)

		out=[]
		#level 1
		out.append(self.assemble([(i,Sz)]))
#		if(not self.conserve):
#			out.append(self.assemble([(i,Sx)]))
#			out.append(self.assemble([(i,Sy)]))
		#level 2
		if(level>=2):
			if(self.conserve):
				out.append(self.assemble([((i+1)%N,Sz)]))
				out.append(self.assemble([(i,Sz),((i+1)%N,Sz)]))
				out.append(  (self.assemble([(i,Sm),((i+1)%N,Sp)])+self.assemble([(i,Sp),((i+1)%N,Sm)]))/np.sqrt(2)  )
			else:
				sops=[Sx,Sy,Sz]
				for op1 in sops:
					out.append(self.assemble([((i+1)%N,op1)]))
					for op2 in sops:
						out.append(self.assemble([(i,op1),((i+1)%N,op2)]))
		#level 3
		if(level>=3):
			out.append(self.assemble([((i-1)%N,Sz)]))		
			out.append(self.assemble([(i,Sz),((i-1)%N,Sz)]))
			out.append(  (self.assemble([(i,Sm),((i-1)%N,Sp)])+self.assemble([(i,Sp),((i-1)%N,Sm)]))/np.sqrt(2)  )
		#level 4
		if(level>=4):
			out.append(self.assemble([((i-1)%N,Sz),((i+1)%N,Sz)]))
			out.append(  (self.assemble([(i+1,Sm),((i-1)%N,Sp)])+self.assemble([(i+1,Sp),((i-1)%N,Sm)]))/np.sqrt(2)  )
		#level 5
		if(level>=5):
			out.append(self.assemble([ ((i-1)%N,Sz),(i,Sz),((i+1)%N,Sz)   ]))
			out.append( (self.assemble([ ((i-1)%N,Sz),(i,Sp),((i+1)%N,Sm)   ])+self.assemble([ ((i-1)%N,Sz),(i,Sm),((i+1)%N,Sp)]))/np.sqrt(2))
			out.append( (self.assemble([ ((i-1)%N,Sp),(i,Sz),((i+1)%N,Sm)   ])+self.assemble([ ((i-1)%N,Sm),(i,Sz),((i+1)%N,Sp)]))/np.sqrt(2))
			out.append( (self.assemble([ ((i-1)%N,Sp),(i,Sm),((i+1)%N,Sz)   ])+self.assemble([ ((i-1)%N,Sm),(i,Sp),((i+1)%N,Sz)])) /np.sqrt(2))
		return np.array(out)
	
	def make_flippers(self):
		Sx=np.zeros((2,2,1,1))
		Sx[:,:,0,0]=self.sx
		self.flippers=[]
		for i in range(self.N): self.flippers.append(self.assemble([(i,Sx)],norm=False,conserve=False))
		self.flippers=np.array(self.flippers)
					
	def assemble(self,ops,N=None,norm=True,conserve=None):
		if N is None: N=self.N
		if conserve is None: conserve=self.conserve
		Id=np.zeros((2,2,1,1))
		if norm: Id[0,0,0,0]=Id[1,1,0,0]=1/np.sqrt(2)
		else: Id[0,0,0,0]=Id[1,1,0,0]=1
		sites,ops=zip(*ops)
		temp=np.ones((1))
		for i in range(N):
			try:
				k=sites.index(i)
				temp=np.tensordot(temp,ops[k],([-1],[2]))
			except ValueError:
				temp=np.tensordot(temp,Id,([-1],[2]))
				
		temp=temp.reshape([2]*(2*N))
		temp=temp.transpose(range(0,2*N,2)+range(1,2*N,2))
		temp=temp.reshape([2**N]*2)
		if(conserve):
			q_flat=make_q_flat(N)
			perm,temp= npc.array.from_ndarray_flat(temp,[q_flat,q_flat],bunch=True,sort=True,q_conj=[1,-1])
		return temp
	
	def visualize(self,state):
		N=self.N
		out=np.zeros(N)
		for i in range(N):
#			print self.twobody_ops[i][0]
#			print np.dot(np.tensordot(self.twobody_ops[i][0],state,([1],[0])),state)
			out[i]=np.dot(state,np.tensordot(self.twobody_ops[i][0],state,([1],[0])))
		return out

	def project(self,op,v):
		if(self.conserve):
			vtemp=v.shallow_copy()
			vtemp.q_conj=np.array([-1,1])
			return npc.tensordot(npc.tensordot(vtemp,op,([0],[0])),v,([1],[0]))
		else:
			return np.tensordot(np.tensordot(v,op,([0],[0])),v,([1],[0]))

	def truncate(self,v,Ntrunc):
		#decide which indices to project out. 
		start=v.shape[1]//2-Ntrunc//2
		if(Ntrunc%2): start=start-1
		end=v.shape[1]//2+Ntrunc//2
#		start=v.shape[1]-Ntrunc
#		end=v.shape[1]
#		print start,end,Ntrunc
#		print self.energy_keys
#		print v.charge
		nonempty_sectors=[]
		if self.conserve:
			qind_counter=0
			otheraxis_counter=0
			v.q_ind[1]=v.q_ind[1].copy()
			newdat=[]; newqdat=[]
			qdat_counter=0
			for charge,dat in enumerate(v.dat):
#				print np.where(self.energy_keys[v.q_ind[1][charge,0]:v.q_ind[1][charge,1]]  <start)
#				print np.where(self.energy_keys[v.q_ind[1][charge,0]:v.q_ind[1][charge,1]]  >=end)
				temp=v.dat[charge][:,np.concatenate([np.where(self.energy_keys[v.q_ind[1][charge,0]:v.q_ind[1][charge,1]]  <start)[0],
					np.where(self.energy_keys[v.q_ind[1][charge,0]:v.q_ind[1][charge,1]]  >=end)[0]])]
				v.q_ind[1][charge,0]=qind_counter
				qind_counter=temp.shape[1]+qind_counter
				v.q_ind[1][charge,1]=qind_counter
				if temp.shape[1]!=0: 
					nonempty_sectors.append(charge)
					newdat.append(temp)
#					newqdat.append(v.q_dat[charge,:])
					newqdat.append([charge,qdat_counter])
					qdat_counter+=1
				else: otheraxis_counter=otheraxis_counter+v.dat[charge].shape[0]
					

				#fix the phase				
				for i in range(v.dat[charge].shape[1]):
					if v.dat[charge][np.argmax(abs(v.dat[charge][:,i])),i]<0: v.dat[charge][:,i]=v.dat[charge][:,i]*-1
					
			#eliminate charge sectors that now have no states
			nonempty_sectors=np.array(nonempty_sectors)
			v.q_ind[1]=v.q_ind[1][nonempty_sectors,:]
			v.dat=newdat
			v.q_dat=np.array(newqdat,dtype=np.uint)
			v.shape[1]=qind_counter
						
#			print v.q_ind[0]		
#			print v.q_ind[1]
#			print v.q_dat
			v.check_sanity()
			return v			
		else:
			for i in range(v.shape[1]):
				if v[np.argmax(abs(v[:,i])),i]<0: v[:,i]=v[:,i]*-1
			return np.hstack([v[:,:start],v[:,end:]])
		
	def chandranize(self,op,v):
		if self.conserve :
			raise NotImplementedError
		else:
			out=np.zeros(v.shape[1])
			for i in range(v.shape[1]):
				#print np.tensordot(np.tensordot(v[:,i],op,([0],[0])),v[:,i],([0],[0]))
				out[i]=np.tensordot(np.tensordot(v[:,i],op,([0],[0])),v[:,i],([0],[0])).real
			out=np.diag(out)
			return out
			
	def print_r(self):
		def getr(x1,x2):
			if abs(x1)<1e-10 and abs(x2)<1e-10: return 1
			if x1<x2: 
			#	print x1,x2
				return x1/x2
			else: return x2/x1

		vec_getr=np.vectorize(getr)		

		out=0
		total_size=0
		q=self.v.q_ind[0]
		for charge in range(q.shape[0]):
		 	e=q[charge,1]
		 	s=q[charge,0]
		 	
		 	if (e-s)<3: continue
		 	spacings=self.w[s+1:e]-self.w[s:e-1]
		 	out+=np.mean(vec_getr(spacings[1:],spacings[:-1])*(e-s))
		 	total_size+=e-s
		return out/total_size

def make_q_flat(N):
	q_flat=[]
	for i in range(pow(2,N)):
		q_flat.append([bin(i).count("1")])
	return np.array(q_flat)

