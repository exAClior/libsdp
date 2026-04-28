/*
 * Native complex Hermitian BPSDP support.
 */

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>

#include <complex_bpsdp_solver.h>
#include <cg_solver.h>

#include "blas_helper.h"

namespace libsdp {

static void complex_axpy(size_t length,
                         double alpha,
                         const std::complex<double> * x,
                         std::complex<double> * y) {
    for (size_t i = 0; i < length; i++) {
        y[i] += alpha * x[i];
    }
}

static void complex_copy(size_t length,
                         const std::complex<double> * x,
                         std::complex<double> * y) {
    for (size_t i = 0; i < length; i++) {
        y[i] = x[i];
    }
}

static double complex_norm(size_t length, const std::complex<double> * x) {
    double nrm2 = 0.0;
    for (size_t i = 0; i < length; i++) {
        nrm2 += std::norm(x[i]);
    }
    return std::sqrt(nrm2);
}

static double complex_real_dot(size_t length,
                               const std::complex<double> * x,
                               const std::complex<double> * y) {
    double dot = 0.0;
    for (size_t i = 0; i < length; i++) {
        dot += std::real(std::conj(x[i]) * y[i]);
    }
    return dot;
}

static void copy_col_major_hermitian_to_row_major(const std::vector<std::complex<double>> & src,
                                                  long int n,
                                                  std::complex<double> * dst) {
    for (long int p = 0; p < n; p++) {
        dst[p * n + p] = std::complex<double>(std::real(src[p + p * n]), 0.0);
        for (long int q = p + 1; q < n; q++) {
            std::complex<double> v = 0.5 * (src[p + q * n] + std::conj(src[q + p * n]));
            dst[p * n + q] = v;
            dst[q * n + p] = std::conj(v);
        }
    }
}

void project_hermitian_psd_blocks(const std::complex<double> * u,
                                  double mu,
                                  const std::vector<int> & primal_block_dim,
                                  std::complex<double> * x,
                                  std::complex<double> * z) {

    char char_n = 'N';
    char char_c = 'C';
    std::complex<double> one(1.0, 0.0);
    std::complex<double> zero(0.0, 0.0);

#pragma omp parallel for schedule(dynamic)
    for (int iblock = 0; iblock < static_cast<int>(primal_block_dim.size()); iblock++) {
        if (primal_block_dim[iblock] == 0) continue;

        long int myoffset = 0;
        for (int j = 0; j < iblock; j++) {
            myoffset += static_cast<long int>(primal_block_dim[j]) * primal_block_dim[j];
        }

        long int block_dim = primal_block_dim[iblock];
        long int nn = block_dim * block_dim;

        std::vector<std::complex<double>> mat(nn);
        std::vector<std::complex<double>> scaled_vecs(nn);
        std::vector<std::complex<double>> eigvecs(nn);
        std::vector<std::complex<double>> out(nn);
        std::vector<double> eigval(block_dim);

        // LAPACK uses column-major storage.  The public libsdp vector layout is
        // row-major, so copy explicitly and hermitize while doing it.
        for (long int p = 0; p < block_dim; p++) {
            for (long int q = p; q < block_dim; q++) {
                std::complex<double> hpq = 0.5 * (u[myoffset + p * block_dim + q]
                                                + std::conj(u[myoffset + q * block_dim + p]));
                if (p == q) hpq = std::complex<double>(std::real(hpq), 0.0);
                mat[p + q * block_dim] = hpq;
                mat[q + p * block_dim] = std::conj(hpq);
            }
        }

        DiagonalizeHermitian(block_dim, mat.data(), eigval.data());

        long int mydim = 0;
        for (long int j = 0; j < block_dim; j++) {
            if (eigval[j] > 0.0) {
                for (long int q = 0; q < block_dim; q++) {
                    scaled_vecs[q + mydim * block_dim] = mat[q + j * block_dim] * (eigval[j] / mu);
                    eigvecs[q + mydim * block_dim]     = mat[q + j * block_dim];
                }
                mydim++;
            }
        }

        std::fill(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
        if (mydim > 0) {
            F_ZGEMM(&char_n, &char_c,
                    &block_dim, &block_dim, &mydim,
                    &one,
                    scaled_vecs.data(), &block_dim,
                    eigvecs.data(), &block_dim,
                    &zero,
                    out.data(), &block_dim);
        }
        copy_col_major_hermitian_to_row_major(out, block_dim, &x[myoffset]);

        mydim = 0;
        for (long int j = 0; j < block_dim; j++) {
            if (eigval[j] < 0.0) {
                for (long int q = 0; q < block_dim; q++) {
                    scaled_vecs[q + mydim * block_dim] = mat[q + j * block_dim] * (-eigval[j]);
                    eigvecs[q + mydim * block_dim]     = mat[q + j * block_dim];
                }
                mydim++;
            }
        }

        std::fill(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
        if (mydim > 0) {
            F_ZGEMM(&char_n, &char_c,
                    &block_dim, &block_dim, &mydim,
                    &one,
                    scaled_vecs.data(), &block_dim,
                    eigvecs.data(), &block_dim,
                    &zero,
                    out.data(), &block_dim);
        }
        copy_col_major_hermitian_to_row_major(out, block_dim, &z[myoffset]);
    }
}

static void evaluate_complex_cg_AATu(double * Ax, double * x, void * data) {
    ComplexBPSDPSolver * sdp = reinterpret_cast<ComplexBPSDPSolver*>(data);
    sdp->evaluate_AATu(Ax, x);
}

void ComplexBPSDPSolver::evaluate_AATu(double * AATu, double * u) {
    evaluate_ATu_(ATu_, u, data_);
    evaluate_Au_(AATu, ATu_, data_);
}

ComplexBPSDPSolver::ComplexBPSDPSolver(long int n_primal, long int n_dual, SDPOptions options) {
    options_      = options;
    n_primal_     = n_primal;
    n_dual_       = n_dual;
    mu_           = 0.1;
    primal_error_ = 0.0;
    dual_error_   = 0.0;
    oiter_        = 0;
    iiter_total_  = 0;
    is_converged_ = false;
    data_         = nullptr;

    y_      = new double[n_dual_]();
    Au_     = new double[n_dual_]();
    cg_rhs_ = new double[n_dual_]();
    z_      = new std::complex<double>[n_primal_]();
    ATu_    = new std::complex<double>[n_primal_]();
}

ComplexBPSDPSolver::~ComplexBPSDPSolver() {
    delete [] y_;
    delete [] Au_;
    delete [] cg_rhs_;
    delete [] z_;
    delete [] ATu_;
}

void ComplexBPSDPSolver::solve(std::complex<double> * x,
                               double * b,
                               std::complex<double> * c,
                               std::vector<int> primal_block_dim,
                               int maxiter,
                               ComplexSDPAuCallbackFunction evaluate_Au,
                               ComplexSDPATuCallbackFunction evaluate_ATu,
                               SDPProgressMonitorFunction progress_monitor,
                               int print_level,
                               void * data) {

    data_         = data;
    evaluate_Au_  = evaluate_Au;
    evaluate_ATu_ = evaluate_ATu;

    std::shared_ptr<CGSolver> cg(new CGSolver(n_dual_));
    cg->set_max_iter(options_.cg_maxiter);
    cg->set_convergence(options_.cg_convergence);

    double primal_dual_objective_gap = 0.0;
    int oiter_local = 0;

    if (options_.guess_type != "read") {
        mu_ = options_.penalty_parameter;
    }

    do {

        // Au_ = mu * (b - A x)
        evaluate_Au(Au_, x, data);
        C_DAXPY(n_dual_, -1.0, b, 1, Au_, 1);
        C_DSCAL(n_dual_, -mu_, Au_, 1);

        // cg_rhs_ = A(c - z) + mu * (b - A x)
        for (long int i = 0; i < n_primal_; i++) {
            ATu_[i] = c[i] - z_[i];
        }
        evaluate_Au(cg_rhs_, ATu_, data);
        C_DAXPY(n_dual_, 1.0, Au_, 1, cg_rhs_, 1);

        if (options_.dynamic_cg_convergence) {
            double cg_conv_i = options_.cg_convergence;
            if (oiter_ == 0) {
                cg_conv_i = 0.01;
            } else {
                cg_conv_i = (primal_error_ > dual_error_) ? 0.01 * dual_error_ : 0.01 * primal_error_;
            }
            if (cg_conv_i < options_.cg_convergence) {
                cg_conv_i = options_.cg_convergence;
            }
            cg->set_convergence(cg_conv_i);
        }

        cg->solve(Au_, y_, cg_rhs_, evaluate_complex_cg_AATu, (void*)this);
        int iiter = cg->total_iterations();
        iiter_total_ += iiter;

        update_xz(x, c, primal_block_dim, evaluate_ATu, data);

        evaluate_ATu(ATu_, y_, data);
        complex_axpy(n_primal_,  1.0, z_, ATu_);
        complex_axpy(n_primal_, -1.0, c,  ATu_);
        dual_error_ = complex_norm(n_primal_, ATu_);

        evaluate_Au(Au_, x, data);
        C_DAXPY(n_dual_, -1.0, b, 1, Au_, 1);
        primal_error_ = C_DNRM2(n_dual_, Au_, 1);

        double objective_primal = complex_real_dot(n_primal_, c, x);
        double objective_dual   = C_DDOT(n_dual_, b, 1, y_, 1);
        primal_dual_objective_gap = std::fabs(objective_primal - objective_dual);

        progress_monitor(print_level, oiter_, iiter, objective_primal, objective_dual,
                         mu_, primal_error_, dual_error_, data);

        oiter_++;
        oiter_local++;

        if (oiter_ % options_.mu_update_frequency == 0 && oiter_ > 0) {
            if (dual_error_ > 0.0) {
                mu_ = mu_ * primal_error_ / dual_error_;
            }
            write_xyz(x);
        }

        if (primal_error_ > options_.sdp_error_convergence
         || dual_error_ > options_.sdp_error_convergence
         || primal_dual_objective_gap > options_.sdp_objective_convergence) {
            is_converged_ = false;
        } else {
            is_converged_ = true;
        }

        if (oiter_local == maxiter) break;

        if (PyErr_CheckSignals() != 0) {
            throw pybind11::error_already_set();
        }

    } while (!is_converged_);

    write_xyz(x);
}

void ComplexBPSDPSolver::update_xz(std::complex<double> * x,
                                   std::complex<double> * c,
                                   std::vector<int> primal_block_dim,
                                   ComplexSDPATuCallbackFunction evaluate_ATu,
                                   void * data) {

    evaluate_ATu(ATu_, y_, data);
    complex_axpy(n_primal_, -1.0, c, ATu_);
    complex_axpy(n_primal_,  mu_, x, ATu_);

    project_hermitian_psd_blocks(ATu_, mu_, primal_block_dim, x, z_);
}

void ComplexBPSDPSolver::write_xyz(std::complex<double> * x) {
    if (options_.outfile == "") {
        return;
    }

    FILE * fp = fopen(options_.outfile.c_str(), "w");
    if (fp == NULL) {
        return;
    }

    fprintf(fp, "%li\n", n_primal_);
    for (long int i = 0; i < n_primal_; i++) {
        fprintf(fp, "%20.12le %20.12le\n", std::real(x[i]), std::imag(x[i]));
    }

    fprintf(fp, "%li\n", n_dual_);
    for (long int i = 0; i < n_dual_; i++) {
        fprintf(fp, "%20.12le\n", y_[i]);
    }

    fprintf(fp, "%li\n", n_primal_);
    for (long int i = 0; i < n_primal_; i++) {
        fprintf(fp, "%20.12le %20.12le\n", std::real(z_[i]), std::imag(z_[i]));
    }

    fprintf(fp, "%20.12le\n", mu_);
    fclose(fp);
}

void ComplexBPSDPSolver::read_xyz(std::complex<double> * x) {
    FILE * fp = fopen(options_.outfile.c_str(), "r");
    if (fp == NULL) {
        printf("\n");
        printf("    error: restart file does not exist: %s\n", options_.outfile.c_str());
        printf("\n");
        exit(1);
    }

    long int dim;
    double re, im;

    fscanf(fp, "%li", &dim);
    if (dim != n_primal_) {
        printf("\n");
        printf("    error: dimension mismatch when reading complex solution (x)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le %le", &re, &im);
        x[i] = std::complex<double>(re, im);
    }

    fscanf(fp, "%li", &dim);
    if (dim != 0 && dim != n_dual_) {
        printf("\n");
        printf("    error: dimension mismatch when reading complex solution (y)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le", &y_[i]);
    }

    fscanf(fp, "%li", &dim);
    if (dim != 0 && dim != n_primal_) {
        printf("\n");
        printf("    error: dimension mismatch when reading complex solution (z)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le %le", &re, &im);
        z_[i] = std::complex<double>(re, im);
    }

    fscanf(fp, "%le", &mu_);
    fclose(fp);
}

}
