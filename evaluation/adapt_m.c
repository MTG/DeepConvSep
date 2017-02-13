/* This file implements the adaptation loops described in [Dau et al., 1997]
 * as a C function, to be incorporated into MATLAB as a MEX function.
 *
 * Remark: The adaptation loops described in [Dau et al., 1997] differ from 
 * that described in [Dau et al., 1996] by limiting the output of each loop
 * as proposed by [Muenkner, 1993]. 
 *
 * filename : adapt_m.c
 * copyright: Universitaet Oldenburg, c/o Prof. B. Kollmeier
 * authors  : rh et al. 
 * date     : 2003
 */


/*-----------------------------------------------------------------------------
 *   Copyright (C) 2003   Medizinische Physik
 *                        c/o Prof. B. Kollmeier                            
 *                        Universitaet Oldenburg, Germany
 *                        http://medi.uni-oldenburg.de
 *   
 *   Permission to use, copy, and distribute this software/file and its
 *   documentation for any purpose without permission by Prof. B. Kollmeier
 *   is not granted.
 *   
 *   Permission to use this software for academic purposes is generally
 *   granted.
 *
 *   Permission to modify the software is granted, but not the right to
 *   distribute the modified code.
 *
 *   This software is provided "as is" without expressed or implied warranty.
 *
 *   AUTHORS:                                                                    
 *
 *   Rainer Huber         (rainer.huber@uni-oldenburg.de)                
 *   Michael Kleinschmidt                 
 *   Martin Hansen                         
 *   Juergen Tchorz                           
 *   Andreas Krause
 *
 *   REFERENCES:
 *
 *   Dau, T. , Püschel, D. and Kohlrausch, A. (1996): "A quantitative model of the
 *     `effective' signal processing in the auditory system: I. Model structure," 
 *     J. Acoust. Soc. Am. 99, p. 3615-3622.
 *
 *   Dau, T., Kollmeier, B. and Kohlrausch, A. (1997): "Modeling auditory processing
 *     of amplitude modulation: I. Modulation Detection and masking with narrowband
 *     carriers," J. Acoust. Soc. Am. 102(5), p. 2892-2905.
 *
 *   Hansen, M. and Kollmeier, B. (2000): "Objective modelling of speech quality
 *     with a psychoacoustically validated auditory model," J. Audio Eng. Soc. 48", 
 *     p. 395-409. 
 *
 *   Münkner, S. (1993): "A psychoacoustical model for the perception of
 *     non-stationary sounds," in: Schick, A., ed., Contributions to Psychological
 *     Acoustics, vol. 6, p. 121-134
 *
 *   Tchorz, J. and Kollmeier, B. (1999): " A model of auditory perception as front
 *     end for automatic speech recognition.", J. Acoust. Soc. Am. 106(4),
 *     p. 2040-2050.   
 *---------------------------------------------------------------------------*/

#include "adapt.h"
#include <math.h>
#include "mex.h"


/* Input Arguments */
#define IN        prhs[0]
#define FSAMPLE   prhs[1]
#define LP_CUTOFF prhs[2]
#define LIMIT     prhs[3]
#define T1        prhs[4]
#define T2        prhs[5]
#define T3        prhs[6]
#define T4        prhs[7]
#define T5        prhs[8]


/* Output Arguments */
#define OUT       plhs[0]

/* Default values */
/* Nrgl time constants */

#define TAU0 0.005  
#define TAU1 0.05
#define TAU2 0.129
#define TAU3 0.253
#define TAU4 0.5 

#define TRUE 1
#define FALSE 0
#define OUT_LOWPASS_CUTOFF 7.957747   
#define DBRANGE 100

#define QLIMIT true
#define max(A, B)   ((A) > (B) ? (A) : (B))
#define min(A, B)   ((A) < (B) ? (A) : (B))

#define MUENK_LIMIT 10    



/* structure containing factors, thresholds and other constants for ADLs */
struct adaptive_loop_stage_constants_structure 
{
     /* 1st resonance filter lowpass cutoff freq. 1 kHz */
     DATA dmpin ;        /* weighting factor previously processed sample */
     DATA addin ;        /* weighting factor currently processed sample */
     
     /* for each adaptive stage loop  */
     DATA dmp[NSTAGE] ;  /* weighting factor previously processed sample */
     DATA add[NSTAGE] ;  /* weighting factor currently processed sample */
     DATA thr[NSTAGE] ;  /* lower bound for divisor  */
     
     /* 2nd resonance filter lowpass cutoff freq. 8 Hz */
     DATA dmpout ;       /* weighting factor previously processed sample */
     DATA addout ;       /* weighting factor currently processed sample*/
     
     /* internal lower bound for input & scaling of output */
     DATA minin ;        /* set the input to the loops to this minimal value */
     DATA lbound ;       /* for final scaling to model units: offset */
     DATA dbfac ;        /* for final scaling to model units: factor */
} Adlc;                       




/************************** NRGLINIT ***********************************
   Description: Initilizes the constants-structure 'Adlc'
                for the adaptive loop stages and (allocates and
                initializes an array of loop_varible_structures no longer)

   Input:
         adl:           an array of loop_varible_structures
         smplfrq:       sampling frequency
         LowpassCutoff: output lowpass cutoff frequency
         tau:           time constants for five lowpass filters
            
   Output:
            none
   Return:
            none
 *********************************************************************/
void NrglInit (adl_struc *adl,float smplfrq, float LowpassCutoff,float *tau)
{
     float dbrange ;
     short i, j ;
     float tauout;
     
     /* brain lowpass at output */
     if (LowpassCutoff) { 
        tauout = 0.5 / (M_PI * LowpassCutoff); 
        Adlc.dmpout = (float) exp ((double) (-1.0 / (smplfrq * tauout))) ;
        Adlc.addout = 1.0 - Adlc.dmpout ;
     }
     
     dbrange = DBRANGE ;  /* ampl 1.0 becomes this dB SPL value [0,100 dB] */
    
    /* -------------------------------------------------- */
    /* initialize the adaptive loops constants-structure  */
    /* -------------------------------------------------- */
    Adlc.minin = pow (10.0, -dbrange / 20.) ;

    /* calculate the (constant) LP filter coefficients
       and the lower bound of the divisor of the five loops */
    for (Adlc.lbound=Adlc.minin, i=0; i<NSTAGE; i++) {
        Adlc.dmp[i] = exp (-1.0 / (smplfrq * tau[i])) ;
        Adlc.add[i] = 1.0 - Adlc.dmp[i] ;
     
        Adlc.thr[i] = i ? sqrt (Adlc.thr[i-1]) : sqrt (Adlc.minin) ;
        Adlc.lbound = sqrt (Adlc.lbound) ;
    }
    
    Adlc.dbfac = dbrange / (1.0 - Adlc.lbound) ;


    /* -------------------------------------------------- */
    /* initialize the adaptive loops variables-structure  */
    /* -------------------------------------------------- */
     
    /* set input(lowpass) to zero */
    adl->in = 0.0 ;
      
    /* initially, set the divisor of each stage to the value of
       its state of rest.  This state of rest is reached when the
       input is between 0 dB and absolute threshold for a
       sufficient long time.
       This inital value is also the lower bound for the divisor.
       The divisor is at the same time also the previous
       sample of the corresponding lowpass filter  */
    for (j=0; j<NSTAGE; j++)
        adl->stage[j] = Adlc.thr[j];
      
     /* set stage output values, maximum stage output values 
        and maximum divisor values to zero (for histogram
        statistics only)   */
    for (j=0; j<NSTAGE; j++) {
         adl->stageout[j] = 0;
         adl->stageoutmax[j] = 0;
         adl->stagemax[j] = 0;
    }   

    /* set output (lowpass) to its initial value
       it depends on the order of processing in the loops,
       whether adl[i].out has to be set to 0.0 or to Adlc.lbound :
         
       if scaling precedes LP-filtering, it has to be zero, but
       if LP-filtering precedes scaling, it has to be Adlc.lbound
    */

               /* either */
    /* adl[i].out = Adlc.lbound ;  */
               /* or */
    adl->out = 0.0; 
      
} /* end NrglInit */




/************************** NRGL *************************************
  Description: Adaptive loops to simulate forward masking.                    
  The dynamic input/output range is 100 dB or [1e-5 - 1] in absolute numbers. 
  Lbound is the 32th root of 1E-5 ~= 0.7.  In case of a stationary input      
  five stages map the input values of the interval [1e-5, 1] to the           
  intervall [0.7, 1]. Before passing these [output] values to the final output
  low pass filter stage, they are rescaled: first an offset of -0.7 is added, 
  shifting the range to [0, 0.3] and then multiplied with 100/0.3 to achieve  
  the desired final output range  [0, 100] in model units.

  By default qbrainlp equals 1, meaning the output of the adaptive loop stage  
  is low pass filtered with a cutoff frequency of 8 Hz (modelling the brain   
  lowpass).                                                                   
  In feature mode this cutoff frequency can be changed or the brain lowpass   
  deactivated completely.                                                     
  In modulation filterbank mode qbrainlp is always zero. The brain lowpass     
  is replaced with the modulation filterbank code in file mfb.c.              

  Input:
         *adl:       array of structures with adaptive loop variables
                     for this one channel,
         *data_in:   Databuffer with input samples 
                     with the new output
         framelen:   frame length, i.e. number of input samples in databuffer
         q_out_lp:   flag: set q_out_lp!=0  for the output low pass "brainlp"
         qlimit:     flag: set qlimit!=0 for limiting of the divisors
         out_limit:  limit of the output of each loop according to Muenkner '93
      
  Output:
         *adl:       array of structures with newly overwritten adaptive
                     loop variables
         *data_out:  Databuffer containing the output samples
  Return:
         none
 ********************************************************************** */
void Nrgl(adl_struc *adl, double *data_in, double *data_out, int framelen, int q_out_lp, int qlimit, float out_limit) 
{
    float tmp ;
    int i, j ;
    
    float maxvalue = (1 - Adlc.minin)*out_limit - 1;
    float factor = maxvalue*2;
    float expfac = -2/maxvalue;
    float offset = maxvalue - 1;
  
    
    
    for (i=0; i<framelen; i++) {     /* take every sample */     
        tmp = data_in[i] ;
        tmp = tmp > Adlc.minin ? tmp : Adlc.minin ;
      
        for (j=0; j<NSTAGE; j++) {
    
            /* Division of input to loop j by the divisor */
            tmp = tmp / adl->stage[j] ;
            
            /* Münkner's modification: limit output to out_limit times of the loop's static output */
            if (tmp > 1)
                tmp = factor/(1+exp(expfac*(tmp-1)))-offset;
            
            /* lowpass filter the divisor of stage j */
            adl->stage[j] = Adlc.dmp[j] * adl->stage[j] + Adlc.add[j] * tmp;
      
            if (qlimit) {
            /* limit divisor to a lower bound */
                if (adl->stage[j] < Adlc.thr[j])      
                    adl->stage[j] = Adlc.thr[j] ;
            }

        } /* end stage loop */
      
        /* rescale working buffer to get
        "Model Units" in the range of [0 ... 100] for a
        stationary input in the range of [0 ... 1.0].    */ 
        tmp = (tmp - Adlc.lbound) * Adlc.dbfac ; 
        
        /* if 0, skip resonance LP 8 Hz (brain lowpass) */
        if (q_out_lp) { 
            /* output resonance LP Filter 8 Hz */
            tmp = Adlc.dmpout * adl->out + Adlc.addout * tmp ; 
            adl->out = tmp ;
        }

        data_out[i] = tmp ;
      
    } /* end sample loop */
  
}   /* end Nrgl() */


void usage()
{
    mexPrintf("\n");
    mexPrintf(" Usage: OUT = ADAPT_M(IN, FSAMPLE, [LP_CUTOFF], [LIMIT], [TAU1],[TAU2],[TAU3],[TAU4],[TAU5])\n");                                                           
    mexPrintf("\n");                                                                      
    mexPrintf(" Feedback loops to simulate effects of adaptation.\n");                                                                          
    mexPrintf("\n");                                                                      
    mexPrintf(" Parameters:\n");                                                                
    mexPrintf("   OUT:         output data array\n");            
    mexPrintf("   IN:          input data array\n");            
    mexPrintf("   FSAMPLE:     sampling frequency\n");
    mexPrintf("   [LP_CUTOFF]: cut-off frequency of output (brain) lowpass filter\n");
    mexPrintf("                (default = 7.96Hz (-> time constant = 20ms), set = 0 to shortcut LP-filter)]\n"); 
    mexPrintf("   [LIMIT]:     limit of the output of each adaptation loop (s. Muenkner, 1993)\n");
    mexPrintf("                (default value = 10) \n");
    mexPrintf("   [TAUN]:      time constant for loop N in ms\n");
    mexPrintf("                (default values = 5, 50, 129, 253, 500) \n");
    mexPrintf("\n"); 
    mexPrintf(" copyright: Universitaet Oldenburg, c/o Prof. B. Kollmeier, 2003\n");
    mexPrintf("\n"); 
}   /* end usage()  */



/*
 * The Matlab extension interface function. It checks for proper arguments,
 * extracts the needed data from the input arguments and creates the output
 * arguments.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *in, *out;
    int i, m, n, len;
    float fsample;
    float LowpassCutoff = OUT_LOWPASS_CUTOFF;
    float MuenkLimit = MUENK_LIMIT;
    int qlimit = QLIMIT;
    int q_out_lp = true;
    adl_struc *adl;
    int exit = false;
    float tau[5];
    
    adl = mxCalloc(1, sizeof(adl_struc));
        
    /* Check for proper number of arguments */
    if ((nrhs < 2) || (nrhs > 9) || (nlhs != 1) || ((nrhs < 9) && (nrhs > 4))){
        usage();
        exit = true;   
    } 
    
    if (!exit) {
        /* Check the dimensions and type of IN */
        m = mxGetM(IN);
        n = mxGetN(IN);
        if (!mxIsNumeric(IN) || mxIsComplex(IN) || mxIsSparse(IN)  || !mxIsDouble(IN) || (min(m,n) != 1)) 
            mexErrMsgTxt("ADAPT requires that IN be a N x 1 real vector.");
    
        /* Assign pointers and values to the various parameters*/
        in = mxGetPr(IN);
        fsample = mxGetScalar(FSAMPLE);
        if (nrhs >= 3) {
            LowpassCutoff = mxGetScalar(LP_CUTOFF);
            if (!LowpassCutoff) q_out_lp = false;  
        }
        if (nrhs >= 4) {
            MuenkLimit = mxGetScalar(LIMIT);
        }
        if (nrhs == 9) {
            tau[0] = mxGetScalar(T1)/1000;
            tau[1] = mxGetScalar(T2)/1000;
            tau[2] = mxGetScalar(T3)/1000;
            tau[3] = mxGetScalar(T4)/1000;
            tau[4] = mxGetScalar(T5)/1000;
        } else{
            tau[0] = TAU0;
            tau[1] = TAU1;
            tau[2] = TAU2;
            tau[3] = TAU3;
            tau[4] = TAU4;              
        }
        
        /* Create a matrix (vector) for the return argument */
        len = max(m,n);
        OUT = mxCreateDoubleMatrix(1, len, mxREAL);
        out = mxGetPr(OUT);
    
        /* actual computations */
        NrglInit(adl,fsample, LowpassCutoff, tau);
        Nrgl(adl, in, out, len, q_out_lp, qlimit, MuenkLimit); 
    }

}    /* end mexFunction() */




























