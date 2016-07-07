#include "local_lbits.h"

//using namespace std::placeholders;

//  Matrix-vector multiplication w <- M*v.
void LOCAL::TFIM_conserved(double* v, double* w, double Jz){
	int countJ=0;
	int sign;
	double countH;
	for(int in=0; in<rows; in++) w[in]=0.;
	for(int in=0; in<rows; in++){
		//off-diagonal elements
		countH=0.;
		for(int i=0; i<N; i++){
			if(bittest(states[in],i)) sign=1;
			else sign=-1;

			//hz
			countH+=0.5*sign*alphaz[i];
		}
		
		//diagonal elements
		countJ=0;
		for(int i=0; i<N;i++){
			if(bittest(states[in],i) != bittest(states[in],next(i)) ){
				w[ lookup_flipped(in , states ,2, i,next(i))  ]+=2*v[in]*0.25;
				countJ--;
			}else
				countJ++;	
		}
		w[in]+=(countJ*Jz*0.25+countH)*v[in];	
	}
} //  MultMv.

void LOCAL::Sz(double* v, double* w, int site){
	int sign;
	for(int in=0; in<rows; in++) w[in]=0.;
	for(int in=0; in<rows; in++){
		if(!bittest(states[in],site)) sign=1;
		else sign=-1;
		w[in]+=sign*v[in];	
	}
}
void LOCAL::projUp(double* v, double* w, int site){
	int sign;
	for(int in=0; in<rows; in++) w[in]=0.;
	for(int in=0; in<rows; in++){
		if(!bittest(states[in],site)) sign=2;
		else sign=0;
		w[in]+=sign*v[in];	
	}
}
void LOCAL::make_states(int charge=-1){
	states.clear();
	for(int i=0;i<1<<N;i++)
		if(charge==-1 || count_bits(i)==charge) states.push_back(i);
	rows=states.size();
}
Eigen::MatrixXd LOCAL::makeDense( function<void(double  *v, double *w)> matvec){
    Eigen::MatrixXd EigenDense=Eigen::Matrix<double,-1,-1>::Zero(rows,rows);
//	dense=new ART[n*n];
	double *v=new double[rows];
	double *w=new double[rows];
	for(int i=0;i<rows;i++){
		for(int j=0; j<rows; j++){
			if(i==j) v[j]=1;
			else v[j]=0;
			w[j]=0;
		}
		matvec(v,w);
		for(int j=0; j<rows; j++){
//			dense[i+j*n]=w[j];
			EigenDense(j,i)=w[j];
		}
	}
	delete [] v;
	delete [] w;
	return EigenDense;
}
int LOCAL::next(int i){
	if(i%N==N-1) return 0;
	else return i+1;
}

LOCAL::LOCAL(int tN){
	ifstream cin("params");
	cin>>N;
	cin>>h;
	cin>>alpha;
	cin>>ROD;

	int seed=0;
	ran.seed(seed);

	alphaz=vector<double>(N);
	randArray=vector< vector<double> > (ROD, vector<double>(N) );
	for(int is=0;is<ROD;is++)
		for(int i=0;i<N;i++) randArray[is][i]=(2*ran.rand()-1);

}
void LOCAL::time_evolve(){
	Eigen::MatrixXd H;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	Eigen::DiagonalMatrix< complex<double>,-1,-1> EHD;
	Eigen::MatrixXcd EH,U;
	vector<double> energies;
	int start_window,window;
	Eigen::VectorXcd tempvec;
	make_states(N/2);
	int n_trial_states=rows/10;
	vector<int> trial_states(n_trial_states);
	vector <complex<double> > outvec;
	for(int j=0;j<n_trial_states;j++) trial_states[j]=ran.randInt(rows-1);
	
	Wavefunction<complex<double> > w;
	w.init_wavefunction(N);
	vector<double> spectrum;
	double taus[]={0,0.1,0.15,0.2,0.3,0.5,0.8,1,1.5,2,3,5,8,10,15,20,30,50,80,100,150,200};
	int ntau=22;
	double tau;
	vector<double> EE(ntau,0);
	double frac=alpha;

	Eigen::VectorXd trunc_eigvals;
	Eigen::MatrixXd trunc_eigvecs;
	bool project_inner=true; //if true we project out states in the middle of the spectrum, if false we project out the top of the spectrum
	if(project_inner){
		start_window=rows/2*(1-frac);
		window=rows*frac;
	}else{
		start_window=(1-frac)*rows;
		window=rows-start_window;
	}
	cout<<rows<<" "<<start_window<<" "<<window<<endl;
	for(int is=0;is<ROD;is++){
		for(int i=0;i<N;i++) alphaz[i]=h*randArray[is][i];	
		H=makeDense(bind(&LOCAL::TFIM_conserved,this,placeholders::_1,placeholders::_2,1.) );
		es.compute(H);
		
		Eigen_To_Std(es.eigenvalues(),energies);

		for (int t=0;t<ntau;t++){
			tau=taus[t];

			trunc_eigvals=Eigen::VectorXd(rows-window);
			for(int i=0;i<start_window;i++) trunc_eigvals(i)=es.eigenvalues()(i);
			for(int i=start_window+window;i<rows;i++) trunc_eigvals(i-window)=es.eigenvalues()(i);
			EH=(  (trunc_eigvals.array()*tau*complex<double>(0,1)).exp()  );

			EHD=Eigen::DiagonalMatrix<complex<double>,-1,-1> (EH);

			trunc_eigvecs=Eigen::MatrixXd(rows,rows-window);
			for(int i=0;i<start_window;i++){
				for(int j=0;j<rows;j++) trunc_eigvecs(j,i)=es.eigenvectors().col(i)(j);
			}
			for(int i=start_window+window;i<rows;i++){
				for(int j=0;j<rows;j++) trunc_eigvecs(j,i-window)=es.eigenvectors().col(i)(j);
			}
			U=trunc_eigvecs*EHD*trunc_eigvecs.adjoint();				

			for(int j=0;j<n_trial_states;j++){
//				Eigen::VectorXd temp=es.eigenvectors().row(trial_states[j]);
//				temp=es.eigenvectors().middleCols(start_window,window)*temp.segment(start_window,window);
//				tempvec=U*temp;
				tempvec=U.col(trial_states[j]);
				tempvec=tempvec/tempvec.norm();
				Eigen_To_Std(tempvec,outvec);
				EE[t]+=w.von_neumann_entropy(outvec,states,w.rangeToBitstring(0,N/2),N/2);
			}
		}			
	}
	ofstream Eout("EE");
	for(int i=0;i<ntau;i++) Eout<<taus[i]<<" "<<EE[i]/(1.*ROD*n_trial_states)<<endl;
	Eout.close();
}
void LOCAL::run(){
	double hlist[]={h};
	int Nhs=1;
	double ipr_counter=0;
	Eigen::MatrixXd H,temp;	
	Eigen::VectorXd Sz;
	Eigen::MatrixXd M=Eigen::MatrixXd::Zero(Nhs,N), Nij=Eigen::MatrixXd::Zero(Nhs,N);
	Eigen::MatrixXd M2=Eigen::MatrixXd::Zero(Nhs,N), N2=Eigen::MatrixXd::Zero(Nhs,N);
//	double out;
	vector<int> window_counter(Nhs,0);
	vector<Eigen::MatrixXd> local_op(N,Eigen::MatrixXd(rows,rows)), rho(N,Eigen::MatrixXd(rows,rows));
	vector<double> energies;
	int start_window,window;
	Eigen::VectorXd tempM(N);
	
	//if the for loop over charge has more than one value, then you have to store one of these for each charge sector
//	make_states(N/2); ipr only!

	for(int is=0;is<ROD;is++){
		for(int ih=0;ih<Nhs;ih++){
			tempM=Eigen::VectorXd::Zero(N);
			for(int charge=0;charge<=N;charge++){
				make_states(charge);
				for(int i=0;i<N;i++) local_op[i]=makeDense(bind(&LOCAL::Sz,this,placeholders::_1,placeholders::_2,i) );
				for(int i=0;i<N;i++) rho[i]=makeDense(bind(&LOCAL::projUp,this,placeholders::_1,placeholders::_2,i) );

				for(int i=0;i<N;i++) alphaz[i]=hlist[ih]*randArray[is][i];	
				H=makeDense(bind(&LOCAL::TFIM_conserved,this,placeholders::_1,placeholders::_2,1.) );
	//			cout<<"time to make H"<<(clock()-t)/(1.*CLOCKS_PER_SEC)<<endl;
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);

				//setup energy window
				Eigen_To_Std(es.eigenvalues(),energies);
				start_window=(lower_bound(energies.begin(),energies.end(),lowest_energy)-energies.begin());
				window=(upper_bound(energies.begin(),energies.end(),highest_energy)-energies.begin())-start_window;
				
				//ipr window
//				start_window=rows/2;
//				window=10;
				if(window<=0) continue;
				window_counter[ih]+=window;
				tempM+=Mij(es,start_window,window,rho,local_op);
				//ipr_counter+=IPR(es,start_window,window,local_op);
			}
			for(int i=0;i<N;i++) {
				M(ih,i)+=tempM(i);
				M2(ih,i)+=pow(tempM(i),2);
			}
		}
	}

	for(int ih=0;ih<Nhs;ih++) cout<<window_counter[ih]/(1.*pow(2,N)*ROD)<<endl;
	//ipr printing
//	ofstream iout("ipr");
//	iout<<ipr_counter/(1.*ROD)<<endl;
//	iout.close();
	ofstream Mout("M");
	M=M/(1.*N*ROD*pow(2,N));
	M2=M2/(1.*N*N*ROD*pow(2,2*N));
	for(int i=0;i<N;i++){
		for(int ih=0;ih<Nhs;ih++){
			Mout<<M(ih,i)<<" "<<M2(ih,i)-pow(M(ih,i),2)<<" ";
//			Mout<<Nij(ih,i)<<" "<<N2(ih,i)-pow(Nij(ih,i),2)<<" ";
		}
		Mout<<endl;
	}

	Mout.close();
}
Eigen::VectorXd LOCAL::Mij(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es, int start_window, int window, const vector<Eigen::MatrixXd> &rho, const vector<Eigen::MatrixXd> &local_op){
				
	vector<Eigen::MatrixXd> O(N,Eigen::MatrixXd(rows,rows)), tO(N,Eigen::MatrixXd(rows,rows));
	Eigen::VectorXd tempM=Eigen::VectorXd::Zero(N);
	Eigen::VectorXd D=Eigen::VectorXd(window);

	for(int i=0;i<N;i++){
		//diagonal version
		for(int k=0;k<window;k++) D(k)=es.eigenvectors().col(k+start_window).transpose()*rho[i]*es.eigenvectors().col(k+start_window);
		tO[i]=Eigen::MatrixXd::Zero(window,window);
		tO[i].diagonal()=D;
		O[i]=es.eigenvectors().middleCols(start_window,window).transpose()*local_op[i]*es.eigenvectors().middleCols(start_window,window);
	}
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			tempM(supermod(j-i,N))+=(tO[i]*O[j]).trace();
		}
	}
	return tempM;
}
double LOCAL::IPR(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es, int start_window, int window, const vector<Eigen::MatrixXd> &local_op){
	double out=0, temp;
	for(int a=0;a<window;a++){
		for(int b=a;b<window;b++){
			temp=0;
			for(int i=0;i<N;i++){
				temp+=pow(es.eigenvectors().col(a+start_window).transpose()*local_op[i]*es.eigenvectors().col(b+start_window),4);
			}
			if(a==b) out+=temp*2;//double counting
			else out+=temp;
		}
	}
	return out;
}

int main(){
	LOCAL l(10);
	//l.run();
	l.time_evolve();
}

