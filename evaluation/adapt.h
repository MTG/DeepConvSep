/* 
 * This file implements the adaptation loops described in [Dau et al., 1996]
 * as a C function, to be incorporated into MATLAB as a MEX function.
 *
 * filename : adapt.c
 * copyright: Universitaet Oldenburg, c/o Prof. B. Kollmeier
 * authors  : rh et al. 
 * date     : 1998
 * update   : 08/2003
 */


/*-----------------------------------------------------------------------------
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
 *   Dau, T. , Püschel, D. and Kohlrausch, A. (1996): "A quantitative model of the
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
 *---------------------------------------------------------------------------*/


#include<stdio.h>

#ifndef _NRGL_H
#define _NRGL_H	1

#define NSTAGE 5

#ifndef DATA
#define DATA float
#endif


/* structure saving intermediate values for ADLs */
/* adaptive_loop_stage_variables_structure */
typedef struct  
{
     /* delay variable of first input filter lowpass 1 kHz*/
     DATA in ;
     /* output of each adaptive loop stage after division */
     DATA stageout[NSTAGE] ;
     /* maximum output of each adaptive loop stage after division for statistics */
     DATA stageoutmax[NSTAGE] ;
     /* delay variables of 0th to 4th lowpass filter in adaptive loops */
     DATA stage[NSTAGE] ;
     /* maximum delay variables of 0th to 4th lowpass filter in adaptive loops for statistics */
     DATA stagemax[NSTAGE] ;
     /* delay variable of final output filter lowpass 8 Hz*/
     DATA out ;

     /* all filters are 1st order recursive low pass filter */
} adl_struc;              


#endif /* _NRGL_H */








