#include "utils.h"
#include "wavefunction.h"
#include "MersenneTwister.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <functional>
#include <ctime>

class LOCAL{
public:
	void run(); 
	Eigen::VectorXd Mij(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es, int start_window, int window, const vector<Eigen::MatrixXd> &rho, const vector<Eigen::MatrixXd> &local_op);
	double IPR(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &es, int start_window, int window, const vector<Eigen::MatrixXd> &local_op);
	void time_evolve();
	Eigen::MatrixXd makeDense(function<void(double *v, double *w)> matvec );
	void TFIM_conserved(double *v, double *w, double Jz);
	void Sz(double *v, double *w, int);
	void projUp(double *v, double *w, int);
	LOCAL(int N);
	void make_states(int charge);
	int next(int);
private:
	MTRand ran;
	int N,rows,ROD;
	double alpha,h,lowest_energy,highest_energy;
	vector<int> states;
	vector<double> alphaz;
	vector< vector<double> > randArray;
};
