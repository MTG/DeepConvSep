function [SDR,ISR,SIR,SAR,perm]=bss_eval_images(ie,i)

% BSS_EVAL_IMAGES Ordering and measurement of the separation quality for
% estimated source spatial image signals in terms of true source, spatial
% (or filtering) distortion, interference and artifacts.
%
% [SDR,ISR,SIR,SAR,perm]=bss_eval_images(ie,i)
%
% Inputs:
% ie: nsrc x nsampl x nchan matrix containing estimated source images
% i: nsrc x nsampl x nchan matrix containing true source images
%
% Outputs:
% SDR: nsrc x 1 vector of Signal to Distortion Ratios
% ISR: nsrc x 1 vector of source Image to Spatial distortion Ratios
% SIR: nsrc x 1 vector of Source to Interference Ratios
% SAR: nsrc x 1 vector of Sources to Artifacts Ratios
% perm: nsrc x 1 vector containing the best ordering of estimated source
% images in the mean SIR sense (estimated source image number perm(j)
% corresponds to true source image number j)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2007-2008 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Emmanuel Vincent, Hiroshi Sawada, Pau Bofill, Shoji Makino and Justinian
% P. Rosca, "First stereo audio source separation evaluation campaign:
% data, algorithms and results," In Proc. Int. Conf. on Independent
% Component Analysis and Blind Source Separation (ICA), pp. 552-559, 2007.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors %%%
if nargin<2, error('Not enough input arguments.'); end
[nsrc,nsampl,nchan]=size(ie);
[nsrc2,nsampl2,nchan2]=size(i);
if nsrc2~=nsrc, error('The number of estimated source images and reference source images must be equal.'); end
if nsampl2~=nsampl, error('The estimated source images and reference source images must have the same duration.'); end
if nchan2~=nchan, error('The estimated source images and reference source images must have the same number of channels.'); end

%%% Performance criteria %%%
% Computation of the criteria for all possible pair matches
SDR=zeros(nsrc,nsrc);
ISR=zeros(nsrc,nsrc);
SIR=zeros(nsrc,nsrc);
SAR=zeros(nsrc,nsrc);
for jest=1:nsrc,
    for jtrue=1:nsrc,
        [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(reshape(ie(jest,:,:),nsampl,nchan).',i,jtrue,512);
        [SDR(jest,jtrue),ISR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_image_crit(s_true,e_spat,e_interf,e_artif);
    end
end
% Selection of the best ordering
perm=perms(1:nsrc);
nperm=size(perm,1);
meanSIR=zeros(nperm,1);
for p=1:nperm,
    meanSIR(p)=mean(SIR((0:nsrc-1)*nsrc+perm(p,:)));
end
[meanSIR,popt]=max(meanSIR);
perm=perm(popt,:).';
SDR=SDR((0:nsrc-1).'*nsrc+perm);
ISR=ISR((0:nsrc-1).'*nsrc+perm);
SIR=SIR((0:nsrc-1).'*nsrc+perm);
SAR=SAR((0:nsrc-1).'*nsrc+perm);

return;



function [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)

% BSS_DECOMP_MTIFILT Decomposition of an estimated source image into four
% components representing respectively the true source image, spatial (or
% filtering) distortion, interference and artifacts, derived from the true
% source images using multichannel time-invariant filters.
%
% [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)
%
% Inputs:
% se: nchan x nsampl matrix containing the estimated source image (one row per channel)
% s: nsrc x nsampl x nchan matrix containing the true source images
% j: source index corresponding to the estimated source image in s
% flen: length of the multichannel time-invariant filters in samples
%
% Outputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchan2,nsampl2]=size(se);
[nsrc,nsampl,nchan]=size(s);
if nchan2~=nchan, error('The number of channels of the true source images and the estimated source image must be equal.'); end
if nsampl2~=nsampl, error('The duration of the true source images and the estimated source image must be equal.'); end

%%% Decomposition %%%
% True source image
s_true=[reshape(s(j,:,:),nsampl,nchan).',zeros(nchan,flen-1)];
% Spatial (or filtering) distortion
e_spat=project(se,s(j,:,:),flen)-s_true;
% Interference
e_interf=project(se,s,flen)-s_true-e_spat;
% Artifacts
e_artif=[se,zeros(nchan,flen-1)]-s_true-e_spat-e_interf;

return;



function sproj=project(se,s,flen)

% SPROJ Least-squares projection of each channel of se on the subspace
% spanned by delayed versions of the channels of s, with delays between 0
% and flen-1

[nsrc,nsampl,nchan]=size(s);
s=reshape(permute(s,[3 1 2]),nchan*nsrc,nsampl);

%%% Computing coefficients of least squares problem via FFT %%%
% Zero padding and FFT of input data
s=[s,zeros(nchan*nsrc,flen-1)];
se=[se,zeros(nchan,flen-1)];
fftlen=2^nextpow2(nsampl+flen-1);
sf=fft(s,fftlen,2);
sef=fft(se,fftlen,2);
% Inner products between delayed versions of s
G=zeros(nchan*nsrc*flen);
for k1=0:nchan*nsrc-1,
    for k2=0:k1,
        ssf=sf(k1+1,:).*conj(sf(k2+1,:));
        ssf=real(ifft(ssf));
        ss=toeplitz(ssf([1 fftlen:-1:fftlen-flen+2]),ssf(1:flen));
        G(k1*flen+1:k1*flen+flen,k2*flen+1:k2*flen+flen)=ss;
        G(k2*flen+1:k2*flen+flen,k1*flen+1:k1*flen+flen)=ss.';
    end
end
% Inner products between se and delayed versions of s
D=zeros(nchan*nsrc*flen,nchan);
for k=0:nchan*nsrc-1,
    for i=1:nchan,
        ssef=sf(k+1,:).*conj(sef(i,:));
        ssef=real(ifft(ssef,[],2));
        D(k*flen+1:k*flen+flen,i)=ssef(:,[1 fftlen:-1:fftlen-flen+2]).';
    end
end

%%% Computing projection %%%
% Distortion filters
C=G\D;
C=reshape(C,flen,nchan*nsrc,nchan);
% Filtering
sproj=zeros(nchan,nsampl+flen-1);
for k=1:nchan*nsrc,
    for i=1:nchan,
        sproj(i,:)=sproj(i,:)+fftfilt(C(:,k,i).',s(k,:));
    end
end

return;



function [SDR,ISR,SIR,SAR]=bss_image_crit(s_true,e_spat,e_interf,e_artif)

% BSS_IMAGE_CRIT Measurement of the separation quality for a given source
% image in terms of true source, spatial (or filtering) distortion,
% interference and artifacts.
%
% [SDR,ISR,SIR,SAR]=bss_image_crit(s_true,e_spat,e_interf,e_artif)
%
% Inputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component
%
% Outputs:
% SDR: Signal to Distortion Ratio
% ISR: source Image to Spatial distortion Ratio
% SIR: Source to Interference Ratio
% SAR: Sources to Artifacts Ratio

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchant,nsamplt]=size(s_true);
[nchans,nsampls]=size(e_spat);
[nchani,nsampli]=size(e_interf);
[nchana,nsampla]=size(e_artif);
if ~((nchant==nchans)&&(nchant==nchani)&&(nchant==nchana)), error('All the components must have the same number of channels.'); end
if ~((nsamplt==nsampls)&&(nsamplt==nsampli)&&(nsamplt==nsampla)), error('All the components must have the same duration.'); end

%%% Energy ratios %%%
% SDR
SDR=10*log10(sum(sum(s_true.^2))/sum(sum((e_spat+e_interf+e_artif).^2)));
% ISR
ISR=10*log10(sum(sum(s_true.^2))/sum(sum(e_spat.^2)));
% SIR
SIR=10*log10(sum(sum((s_true+e_spat).^2))/sum(sum(e_interf.^2)));
% SAR
SAR=10*log10(sum(sum((s_true+e_spat+e_interf).^2))/sum(sum(e_artif.^2)));

return;