function [MER,perm]=bss_eval_mix(Ae,A)

% BSS_EVAL_MIX Ordering and measurement of the quality of an estimated
% (possibly frequency-dependent) mixing matrix
%
% [MER,perm]=bss_eval_mix(Ae,A)
%
% Inputs:
% Ae: either a nchan x nsrc estimated mixing matrix (for instantaneous
% mixtures) or a nchan x nsrc x nbin estimated frequency-dependent mixing
% matrix (for convolutive mixtures)
% A: the true nchan x nsrc or nchan x nsrc x nbin mixing matrix
%
% Outputs:
% MER: nsrc x 1 vector of Mixing Error Ratios (SNR-like criterion averaged
% over frequency and expressed in decibels, allowing arbitrary scaling for
% each source in each frequency bin)
% perm: nsrc x 1 vector containing the best ordering of estimated sources
% in the maximum MER sense (estimated source number perm(j) corresponds to
% true source number j)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2008 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Emmanuel Vincent, Shoko Araki and Pau Bofill, "The 2008 Signal Separation
% Evaluation Campaign: A community-based approach to large-scale
% evaluation," In Proc. Int. Conf. on Independent Component Analysis and
% Signal Separation (ICA), pp. 734-741, 2009.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors %%%
if nargin<2, error('Not enough input arguments.'); end
[nchan,nsrc,nbin]=size(Ae);
[nchan2,nsrc2,nbin2]=size(A);
if ~((nchan2==nchan)&&(nsrc2==nsrc)&&(nbin2==nbin)), error('The estimated and true mixing matrix must have the same size.'); end

%%% Performance criterion %%%
% Computation of the criterion for all possible pair matches
MER=zeros(nsrc,nsrc,nbin);
for f=1:nbin,
    for jest=1:nsrc,
        for jtrue=1:nsrc,
            Aproj=A(:,jtrue,f)'*Ae(:,jest,f)/sum(abs(A(:,jtrue,f)).^2)*A(:,jtrue,f);
            MER(jest,jtrue,f)=10*log10(sum(abs(Aproj).^2)/sum(abs(Ae(:,jest,f)-Aproj).^2));
        end
    end
end
MER=mean(MER,3);
% Selection of the best ordering
perm=perms(1:nsrc);
nperm=size(perm,1);
meanMER=zeros(nperm,1);
for p=1:nperm,
    meanMER(p)=mean(MER((0:nsrc-1)*nsrc+perm(p,:)));
end
[meanMER,popt]=max(meanMER);
perm=perm(popt,:).';
MER=MER((0:nsrc-1).'*nsrc+perm);

return;
