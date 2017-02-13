/* 
 * This file implements the haircell model described in [Dau et al., 1996]
 * as a C function, to be incorporated into MATLAB as a MEX function.
 *
 * Only the real values are used for calculation, the imaginary parts are
 * discarded.
 *
 * filename : haircell.c
 * copyright: Universitaet Oldenburg, c/o Prof. B. Kollmeier
 * authors  : rh et al. 
 * date     : 1998
 * update   : 08/2003
 */


/*---------------------------------------------------------------------------------
 *   Copyright (C) 1998-2003   Medizinische Physik
 *                             c/o Prof. B. Kollmeier                                                  
 *                             Universitaet Oldenburg, Germany
 *                             http://medi.uni-oldenburg.de
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
 *   Dau, T. , P�chel, D. and Kohlrausch, A. (1996): "A quantitative model of the
 *     `effective' signal processing in the auditory system: {I}. Model structure", 
 *     J. Acoust. Soc. Am. 99, p. 3615-3622.
 *
 *   Hansen, M. and Kollmeier, B. (2000): "Objective modelling of speech quality
 *     with a psychoacoustically validated auditory model", J. Audio Eng. Soc. 48", 
 *     p. 395-409. 
 *
 *   Tchorz, J. and Kollmeier, B. (1999): " A model of auditory perception as front
 *     end for automatic speech recognition.", J. Acoust. Soc. Am. 106(4),
 *     p. 2040-2050.   
 *---------------------------------------------------------------------------------*/


#define _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#include "mex.h"


#define MY_MAX(x,y) ((x) > (y) ? (x) : (y))

/* Input Arguments */
#define	IN			prhs[0]
#define FSAMPLE  	prhs[1]
#define THRESHOLD  	prhs[2]

/* Output Arguments */
#define	OUT			plhs[0]	

#define	min(A, B)	((A) < (B) ? (A) : (B))
#define LOWPASS_CUTOFF 1000
#define THRESHOLD_DEF	0.0   




/********************************* HAIRCELL *************************************** 
   Description: Haircell model: 
                1) halfway-rectification
                2) 1kHz first order lowpass filter

   Input:
         in:        Databuffer containing the input samples
         len:       number of input samples in databuffer
         fsample:   sampling frequency
         threshold: lower bound at halfway rectification  
            
   Output:
          Databuffer containing the output samples
   Return:
          none
 **********************************************************************************/
void Haircell (double *in, double *out, unsigned int len, float fsample, float threshold)
{
 
  /*  Calculate time constant from cut-off frequency for 1000Hz lowpass */
  double tau = 0.5 / (M_PI * LOWPASS_CUTOFF) ; 
  
  double fcoeff = exp((double) (-1.0 / (fsample * tau))) ;
  double fgain = 1.0 - fcoeff;
  double last = threshold;
  unsigned int i;

  for (i=0; i<len; i++)      /* for each sample */
  {
      /* Halfway-rectification */
      if (*in < threshold)
	  	  *out = threshold ;
      else *out = *in;
      
      /* Input resonance LP filter 1 kHz */
      *out = fcoeff * last + fgain * *out ; 
      last = *out++ ;
      in++;
 }	

}   /* end Haircell() */



void usage()
{
	mexPrintf("\n");
	mexPrintf(" Usage: OUT = HAIRCELL(IN, FSAMPLE, [THRESHOLD])\n");                                                           
	mexPrintf("\n");                                                                      
	mexPrintf(" calculate haircell model for sample in inputbuffer:\n");                   
	mexPrintf(" 1) halfway-rectification\n");                                              
	mexPrintf(" 2) 1kHz  first order lowpass filter\n");                                   
	mexPrintf("\n");                                                                      
	mexPrintf(" Only the real values are used for calculation, the imaginary parts are\n");
	mexPrintf(" discarded.\n");                                                            
	mexPrintf("\n");                                                                      
	mexPrintf(" Parameters:\n");                                                                
	mexPrintf("   OUT:         output data array\n");            
	mexPrintf("   IN:          input data array\n");            
	mexPrintf("   FSAMPLE:     sampling frequency\n");
	mexPrintf("   [THRESHOLD:  threshold for halfway-rectification; default = 0.0]\n");
	mexPrintf("\n"); 
	mexPrintf(" copyright: Universitaet Oldenburg, c/o Prof. B. Kollmeier, 1998-2003\n");
	mexPrintf(" last change: Jul 2003");
	mexPrintf("\n"); 
}    /* end usage() */



/*
 * The Matlab extension interface function. It checks for proper arguments,
 * extracts the needed data from the input arguments and creates the output
 * arguments.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *in, *out;
  	unsigned int m, n, len;
  	float threshold = THRESHOLD_DEF, fsample;
  	int exit = false;
	
  	/* Check for proper number of arguments */
  	if ((nrhs < 2) || (nrhs > 3) || (nlhs != 1)) {
    	usage();
    	exit = true;   
  	} 
  	
  	if(!exit) {
  		/* Check the dimensions and type of IN */
	  	m = mxGetM(IN);
  		n = mxGetN(IN);
	  	if (!mxIsNumeric(IN) || mxIsComplex(IN) || mxIsSparse(IN)  || !mxIsDouble(IN) || (min(m,n) != 1))
   			mexErrMsgTxt("HAIRCELL requires that IN be a N x 1 real vector.");
  	
   		/* Assign pointers and values to the various parameters*/
  		in = mxGetPr(IN);
  		fsample = mxGetScalar(FSAMPLE);
  		if (nrhs == 3) threshold = mxGetScalar(THRESHOLD);
  	
		/* Create a matrix (vector) for the return argument */
  		len = MY_MAX(m,n);
  		OUT = mxCreateDoubleMatrix(1, len, mxREAL);
  		out = mxGetPr(OUT);
  	
		/* Do the actual computations in a subroutine */
  		Haircell(in, out, len, fsample, threshold);
  	}
  	
}   /* end mexFunction() */

