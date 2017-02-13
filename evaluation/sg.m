function [yo,fo,to] = sg(x,nfft,Fs,win,noverlap)
% S = sg(B,NFFT,Fs,win,NOVERLAP)
% win must be created like win = hanning(Nsamples)
% All parameters like SPECGRAM

nx = length(x);
nwind = length(win);
if nx < nwind    % zero-pad x if it has length less than the win length
    x(end+1:nwind)=0;  nx=nwind;
end
x = x(:); % make a column vector for ease later
win = win(:); % be consistent with data set

ncol = fix((nx-noverlap)/(nwind-noverlap));
colindex = 1 + (0:(ncol-1))*(nwind-noverlap);
rowindex = (1:nwind)';
if length(x)<(nwind+colindex(ncol)-1)
    x(nwind+colindex(ncol)-1) = 0;   % zero-pad x
end
y = zeros(nwind,ncol);
y(:) = x(rowindex(:,ones(1,ncol))+colindex(ones(nwind,1),:)-1);
y = win(:,ones(1,ncol)).*y;

% y2=zeros(nfft,ncol);
% chunks = 1:100:ncol;
% if chunks(end)<ncol,
%     chunks(end+1)=ncol;
% end;
% for k=1:length(chunks)-1,
%     pos_ini = 1*(k==1) + (k~=1)*(chunks(k)+1);        
%     pos_fin = chunks(k+1);
%     y2(:,pos_ini:pos_fin) = fft(y(:,pos_ini:pos_fin),nfft);
% end;
% 
% y=y2;
y = fft(y,nfft);

if ~any(any(imag(x)))    % x purely real
    if rem(nfft,2),    % nfft odd
        select = [1:(nfft+1)/2];
    else
        select = [1:nfft/2+1];
    end
    y = y(select,:);
else
    select = 1:nfft;
end
f = (select - 1)'*Fs/nfft;
t = (colindex-1)'/Fs;
if nargout == 1,
    yo = y;
elseif nargout == 2,
    yo = y;
    fo = f;
elseif nargout == 3,
    yo = y;
    fo = f;
    to = t;
end


