function xpad = cf_overlap_add(F_x,SY_WINDOW,OVERLAP);

% Overlap-add reconstruction
%
% Usage: [x,xpad] = overlap_add(F_x,SY_WINDOW,OVERLAP)
%
% Input:
%   - F_x: W x n_frames matrix,
%   - SY_WINDOW: window of size W,
%   - OVERLAP: number of samples overlap
%
% Output:
%   - xpad is the reconstructed, zero-padded signal.

% Author: Cedric Fevotte
% cedric.fevotte@mist-technologies.com


SY_WINDOW=SY_WINDOW(:).'; % Produces a row signal;

W=length(SY_WINDOW);
[junk n_frames] = size(F_x);

if junk ~= W
  disp('Size error between F_x and W');
end

Tpad = OVERLAP + n_frames*(W-OVERLAP); % Length of zero-padded signal
xpad = zeros(1,Tpad);

frames_index = 1 + [0:(n_frames-1)]*(W-OVERLAP); % Index of beginnings of frames

%% Frame 1
xpad(frames_index(1):frames_index(1)+W-1) = (F_x(:,1).').*SY_WINDOW;

%% Frame 1 < n =< n_frames
for n=2:n_frames
  xpad(frames_index(n):frames_index(n)+W-1) = xpad(frames_index(n):frames_index(n)+W-1) + (F_x(:,n).').*SY_WINDOW;
end