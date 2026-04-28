/*
 * Native complex Hermitian BPSDP support.
 *
 * This solver path keeps the dual variables real and stores primal/cone blocks
 * as dense complex matrices.  It is deliberately separate from BPSDPSolver so
 * the established real code path is not disturbed.
 */

#ifndef COMPLEX_BPSDP_SOLVER_H
#define COMPLEX_BPSDP_SOLVER_H

#include "sdp_solver.h"

#include <complex>
#include <functional>
#include <vector>

namespace libsdp {

typedef std::function<void(double*,std::complex<double>*,void*)> ComplexSDPAuCallbackFunction;
typedef std::function<void(std::complex<double>*,double*,void*)> ComplexSDPATuCallbackFunction;

/// Project Hermitian blocks U into x = U_+ / mu and z = -U_-.
void project_hermitian_psd_blocks(const std::complex<double> * u,
                                  double mu,
                                  const std::vector<int> & primal_block_dim,
                                  std::complex<double> * x,
                                  std::complex<double> * z);

class ComplexBPSDPSolver {

  public:

    ComplexBPSDPSolver(long int n_primal, long int n_dual, SDPOptions options);
    ~ComplexBPSDPSolver();

    void solve(std::complex<double> * x,
               double * b,
               std::complex<double> * c,
               std::vector<int> primal_block_dim,
               int maxiter,
               ComplexSDPAuCallbackFunction evaluate_Au,
               ComplexSDPATuCallbackFunction evaluate_ATu,
               SDPProgressMonitorFunction progress_monitor,
               int print_level,
               void * data);

    void evaluate_AATu(double * AATu, double * u);

    int iiter_total() { return iiter_total_; }
    int oiter_total() { return oiter_; }

    void set_mu(double mu) { mu_ = mu; }

    double get_mu() { return mu_; }
    double * get_y() { return y_; }
    std::complex<double> * get_z() { return z_; }

    bool is_converged() { return is_converged_; }

    void read_xyz(std::complex<double> * x);

  protected:

    SDPOptions options_;
    void * data_;

    ComplexSDPAuCallbackFunction evaluate_Au_;
    ComplexSDPATuCallbackFunction evaluate_ATu_;

    double primal_error_;
    double dual_error_;
    bool is_converged_;

    int oiter_;
    int iiter_total_;

    double mu_;

    double * y_;
    double * Au_;
    double * cg_rhs_;

    std::complex<double> * z_;
    std::complex<double> * ATu_;

    long int n_primal_;
    long int n_dual_;

    void update_xz(std::complex<double> * x,
                   std::complex<double> * c,
                   std::vector<int> primal_block_dim,
                   ComplexSDPATuCallbackFunction evaluate_ATu,
                   void * data);

    void write_xyz(std::complex<double> * x);
};

}

#endif
