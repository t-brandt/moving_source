import cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def fit_ramps(double [:, :] diffs,
              char [:, :] diffs2use,
              double [:, :] template,
              double [:] alpha_phnoise,
              double [:] beta_phnoise,
              double [:] alpha_readnoise,
              double [:] beta_readnoise,
              double [:, :] alpha_phnoise_template,
              double [:, :] beta_phnoise_template,
              double [:] sig,
              double [:] countrateguess,
              double [:] cguess_template,
              int nramps,
              int ndiffs):

    cdef extern from "math.h":
        double sqrt(double x) nogil
        double fabs(double x) nogil
        
    cdef int i, j
    
    cdef double *alpha = <double *> PyMem_Malloc(ndiffs * sizeof(double))
    cdef double *beta = <double *> PyMem_Malloc(ndiffs * sizeof(double))

    cdef double *d = <double *> PyMem_Malloc(ndiffs * sizeof(double))
    cdef double *g = <double *> PyMem_Malloc(ndiffs * sizeof(double))
    
    cdef double *phi = <double *> PyMem_Malloc((ndiffs + 1) * sizeof(double))
    cdef double *Phi = <double *> PyMem_Malloc(ndiffs * sizeof(double))
    cdef char *d2u = <char *> PyMem_Malloc(ndiffs * sizeof(char))

    countrate_np = np.empty(nramps)
    cdef double [:] countrate = countrate_np
    uncert_np = np.empty(nramps)
    cdef double [:] uncert = uncert_np
    countrate_sec_np = np.empty(nramps)
    cdef double [:] countrate_sec = countrate_sec_np
    uncert_sec_np = np.empty(nramps)
    cdef double [:] uncert_sec = uncert_sec_np
    chisq_np = np.empty(nramps)
    cdef double [:] chisq = chisq_np
    
    A_0_np = np.empty(nramps)
    cdef double [:] _A_0 = A_0_np
    A_a_np = np.empty(nramps)
    cdef double [:] _A_a = A_a_np
    A_aa_np = np.empty(nramps)
    cdef double [:] _A_aa = A_aa_np
    A_ab_np = np.empty(nramps)
    cdef double [:] _A_ab = A_ab_np
    A_b_np = np.empty(nramps)
    cdef double [:] _A_b = A_b_np
    A_bb_np = np.empty(nramps)
    cdef double [:] _A_bb = A_bb_np
    
    phi[ndiffs] = 1
    Phi[ndiffs - 1] = 0
    
    cdef double scale, iscale

    cdef double Theta_im1, theta_im1, theta_im2, theta_i, Theta_i, ThetaD_i
    cdef double Phi_ip1, phi_ip1, phi_ip2, phi_i, Phi_i
    cdef double theta_0, theta_1, Theta_0, Theta_1, ThetaD_0, ThetaD_1, theta_n
    cdef double ThetaG_0, ThetaG_1, ThetaG_i
    cdef double A_0, A_a, A_aa, A_b, A_bb, A_ab
    cdef double iT, dC, iC, ctrtguess, rnvar, sgn
    cdef double ctrtguess_template
    cdef char d2u_i, d2u_ip1
    cdef double totcts, a, b, c, q
    
    theta_0 = 1
    Theta_0 = -1
    ThetaD_0 = 0
    ThetaG_0 = 0
        
    for j in range(nramps):
        scale = countrateguess[j]*alpha_phnoise[0] + sig[j]*sig[j]*alpha_readnoise[0]
        scale += cguess_template[j]*alpha_phnoise_template[0, j]

        iscale = 1./scale
        
        ctrtguess = iscale*countrateguess[j]
        rnvar = iscale*sig[j]*sig[j]
        ctrtguess_template = iscale*cguess_template[j]

        d2u_ip1 = diffs2use[0, j]
        
        for i in range(ndiffs):
            d2u_i = d2u_ip1
            d2u[i] = d2u_i
            alpha[i] = ctrtguess*alpha_phnoise[i] + rnvar*alpha_readnoise[i]
            alpha[i] += ctrtguess_template*alpha_phnoise_template[i, j]
            
            d[i] = diffs[i, j]*d2u_i
            g[i] = template[i, j]*d2u_i
            
            if i < ndiffs - 1:
                d2u_ip1 = diffs2use[i + 1, j]
                beta[i] = d2u_i*d2u_ip1*(ctrtguess*beta_phnoise[i] +
                                         rnvar*beta_readnoise[i] +
                                         ctrtguess_template*beta_phnoise_template[i, j])
                
        phi[ndiffs - 1] = alpha[ndiffs - 1]
        phi_ip2 = phi[ndiffs]
        phi_ip1 = phi[ndiffs - 1]
        Phi_ip1 = Phi[ndiffs - 1]

        # sgn is negative for even indices
        sgn = 2*((ndiffs - 1)%2) - 1
        
        for i in range(ndiffs - 2, -1, -1):
            phi_i = alpha[i]*phi_ip1 - beta[i]**2*phi_ip2
            Phi_i = beta[i]*(Phi_ip1 + sgn*phi_ip2)
            Phi[i] = Phi_i
            phi[i] = phi_i
            phi_ip2 = phi_ip1
            phi_ip1 = phi_i
            Phi_ip1 = Phi_i
            sgn *= -1

        theta_1 = alpha[0]
        Theta_1 = -beta[0] + theta_1
        ThetaD_1 = -d[0]
        ThetaG_1 = -g[0]
        
        theta_im2 = theta_0
        theta_im1 = theta_1
        Theta_im1 = Theta_1

        ThetaD_i = ThetaD_1
        ThetaD_i = beta[0]*ThetaD_i + d[1]*theta_1

        ThetaG_i = ThetaG_1
        ThetaG_i = beta[0]*ThetaG_i + g[1]*theta_1

        A_0 = d[0]*phi[1]*(-2*ThetaD_0 + d[0]*theta_0)
        
        A_b = phi[1]*(-g[0]*ThetaD_0 - d[0]*ThetaG_0 + g[0]*d[0]*theta_0)
        A_bb = phi[1]*(-2*g[0]*ThetaG_0 + g[0]*g[0]*theta_0)
        
        dC = -(phi[1]*Theta_0 + theta_0*Phi[0])*d2u[0]
        
        A_ab = g[0]*dC
        A_a = d[0]*dC
        A_aa = dC

        A_0 = A_0 + d[1]*phi[2]*(2*beta[0]*ThetaD_1 + d[1]*theta_1)
        A_bb = A_bb + g[1]*phi[2]*(2*beta[0]*ThetaG_1 + g[1]*theta_1)
        A_b = A_b + phi[2]*(d[1]*beta[0]*ThetaG_1 + g[1]*beta[0]*ThetaD_1 + g[1]*d[1]*theta_1)
        
        dC = (phi[2]*Theta_1 + theta_1*Phi[1])*d2u[1]
        
        A_a = A_a + d[1]*dC
        A_aa = A_aa + dC
        A_ab = A_ab + g[1]*dC

        sgn = -1  # -1 for index 0, 1 for index 1, -1 for index 2
        for i in range(2, ndiffs):
            
            theta_i = alpha[i - 1]*theta_im1 - beta[i - 2]**2*theta_im2
            theta_im2 = theta_im1
            theta_im1 = theta_i
            
            Theta_i = Theta_im1*beta[i - 1] + sgn*theta_i
            Theta_im1 = Theta_i
            
            A_0 = A_0 + d[i]*phi[i + 1]*(2*sgn*beta[i - 1]*ThetaD_i + d[i]*theta_i)
            dC = sgn*(phi[i + 1]*Theta_i + theta_i*Phi[i])*d2u[i]
            A_a = A_a + d[i]*dC
            A_ab = A_ab + g[i]*dC
            A_aa = A_aa + dC

            A_b = A_b + phi[i + 1]*(sgn*beta[i - 1]*(g[i]*ThetaD_i +
                                                     d[i]*ThetaG_i)
                                    + d[i]*g[i]*theta_i)
            A_bb = A_bb + phi[i + 1]*(sgn*beta[i - 1]*(2*g[i]*ThetaG_i)
                                    + g[i]*g[i]*theta_i)
            
            ThetaD_i = beta[i - 1]*ThetaD_i + sgn*d[i]*theta_i
            ThetaG_i = beta[i - 1]*ThetaG_i + sgn*g[i]*theta_i
            sgn = -1*sgn
            
        theta_n = alpha[ndiffs - 1]*theta_im1 - beta[ndiffs - 2]**2*theta_im2
            
        iT = 1./theta_n*iscale

        A_0 = A_0*iT
        A_a = A_a*iT
        A_b = A_b*iT
        A_ab = A_ab*iT
        A_aa = A_aa*iT
        A_bb = A_bb*iT

        _A_0[j] = A_0
        _A_a[j] = A_a
        _A_aa[j] = A_aa
        _A_ab[j] = A_ab
        _A_b[j] = A_b
        _A_bb[j] = A_bb

        # Compute the square of the condition number of the A matrix
        a = A_aa*A_aa + A_ab*A_ab
        b = A_aa*A_ab + A_ab*A_bb
        c = A_ab*A_ab + A_bb*A_bb
        q = sqrt((a - c)*(a - c) + 4*b*b)

        # Square of the condition number
        #cond_sq = (a + c + q)/(a + c - q)
        
        # If the condition number is >1e6
        if a + c - q < 1e-12*(a + c + q):
            countrate[j] = A_a/A_aa
            # minus 1 on chi squared to avoid bias from a pixel near the
            # boundary of this test falling in or out
            chisq[j] = A_0 - A_a**2/A_aa
            uncert[j] = sqrt(1/A_aa)
            countrate_sec[j] = 0
            uncert_sec[j] = 1e100

        else:
        
            iC = 1/(A_aa*A_bb - A_ab*A_ab)
            
            countrate[j] = (A_a*A_bb - A_b*A_ab)*iC
            countrate_sec[j] = (A_b*A_aa - A_a*A_ab)*iC
            chisq[j] = A_aa*countrate[j]*countrate[j]
            chisq[j] += A_bb*countrate_sec[j]*countrate_sec[j]
            chisq[j] += A_0
            chisq[j] += 2*A_ab*countrate[j]*countrate_sec[j]
            chisq[j] -= 2*A_a*countrate[j] + 2*A_b*countrate_sec[j]
            
            uncert[j] = sqrt(A_bb*iC)
            uncert_sec[j] = sqrt(A_aa*iC)
        
    PyMem_Free(alpha)
    PyMem_Free(beta)
    PyMem_Free(d)
    PyMem_Free(g)
    PyMem_Free(phi)
    PyMem_Free(Phi)
    PyMem_Free(d2u)

    output = {"countrate": countrate_np,
              "uncert": uncert_np,
              "countrate_sec": countrate_sec_np,
              "uncert_sec": uncert_sec_np,
              "chisq": chisq_np,
              "A_0": A_0_np,
              "A_a": A_a_np,
              "A_aa": A_aa_np,
              "A_ab": A_ab_np,
              "A_b": A_b_np,
              "A_bb": A_bb_np
              }

    return output
