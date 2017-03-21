
function Bach10_eval_only_original(dataset_folder,base_estimates_directory,output_dir,id)

%% ADDED CODE

%we need to obtain imethod_name, song_index, and subset_index from id
%we commented the subsequent for loops for these variables
maxDir=10; 
imethod_name = floor((id-1)/(maxDir))+1;
song_index = mod((id-1),maxDir)+1;

%change the working folder add subfolders to path
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
addpath(genpath(fullfile('..','Toolboxes','bss_eval')));


%%
%here provide the name of the directory that contains all results from your
%different techniques as subfolders
%base_estimates_directory = 'link/to/your/results/rootfolder';

%here, provide the root folder for the DSD100 dataset.
%dataset_folder = 'DSD100';


warning('off','all')

%ADDED CODE - other is saved as others in our separation script
sources_names_files = {'bassoon','clarinet','saxophone','violin'};
sources_names = {'bassoon','clarinet','saxophone','violin'};
test_cases = {'gt','fast','slow','original'};


methods_folders = dir(base_estimates_directory);
isub = [methods_folders(:).isdir]; %# returns logical vector
methods_folders = {methods_folders(isub).name}';
methods_folders(ismember(methods_folders,{'.','..'})) = [];

%comment this line if you don't want to use parallel toolbox. This may
%happen for memory usage reasons. Also, modify this line if you don't want
%to use the maximum number of cores, which is the default behaviour.
%parpool('local');

%for imethod_name = 1:length(methods_folders )
    %for each method (subfolder of the results root folder provided)
    method_name = methods_folders{imethod_name};
    fprintf('\n computing results for method %s\n', method_name)
    result = struct;

    %loop over dev and test subsets
    %for subset_index = 1:2
        sources_folder = fullfile(dataset_folder,'Source separation');
        estimates_folder = fullfile(base_estimates_directory,method_name); %ADDED CODE
        songs_list = dir(sources_folder); %ADDED CODE
        results_set=cell(maxDir,1);

        %loop over songs
        %parfor song_index = 1:50
            song_name = songs_list(song_index+2).name;           
            disp([num2str(song_index),'/',num2str(maxDir),' ',song_name])
            
        for t=1:numel(test_cases)
            ff = strcat(song_name,'_',test_cases(t),'_',sources_names_files{1},'.wav');
        if exist(fullfile(estimates_folder,ff{1}),'file') == 2
        %if ~isequal(exist( fullfile(output_dir,method_name,[song_name '_results.mat']), 'file'),2) %ADDED CODE
            %load the sources references
            sources_data = [];
            for source_index = 1:4
                ff = strcat(song_name, '_', test_cases(t), '_', sources_names{source_index},'.wav');
                source_file = fullfile(sources_folder,song_name,ff{1}); %ADDED CODE
                [source_data,source_sampling] = wavread(source_file);
                [sources_samples,sources_channels] = size(source_data);
                %source_data = repmat(source_data,[1,1]);
                sources_data = cat(2,sources_data,source_data);
            end
      

            %get estimated sotest_casesurces
            sources_estimates_data = zeros(sources_samples,4);
            for estimate_index = 1:4
                ff = strcat(song_name ,'_', test_cases(t), '_', sources_names_files{estimate_index}, '.wav');
                estimate_file = fullfile(estimates_folder,ff{1}); %ADDED CODE
                if exist(estimate_file,'file') == 2
                    estimate_data = wavread(estimate_file);
                    [estimate_samples,estimate_channels] = size(estimate_data);
                    %if mono estimate: duplicate it to get stereo
                    %estimate_data = repmat(estimate_data,[1,3-estimate_channels]);

                    estimates_samples = min(size(sources_estimates_data,1),estimate_samples);
                    sources_estimates_data = sources_estimates_data(1:estimates_samples,:);
                    sources_estimates_data(:,estimate_index) = estimate_data(1:estimates_samples,:);
                end
            end
            tic       
            %estimate quality for sources
            %[SDR,ISR,SIR,SAR] = bss_eval(sources_estimates_data,sources_data,30*source_sampling,15*source_sampling);
            [SDR, SIR, SAR] = bss_eval_sources(sources_estimates_data',sources_data');
            toc      

            %build the result structure for this song
            results_set{song_index} = struct;
            results_set{song_index}.name = song_name;
            for source_index = 1:4
                results_set{song_index}.(sources_names{source_index}).sdr = SDR(source_index,:);
                %results_set{song_index}.(sources_names{source_index}).isr = ISR(source_index,:);
                results_set{song_index}.(sources_names{source_index}).sir = SIR(source_index,:);
                results_set{song_index}.(sources_names{source_index}).sar = SAR(source_index,:);
            end

        %end
        % now gather all the results for the subset
        %for song_index=1:50
        if ~isequal(exist(output_dir, 'dir'),7) 
            mkdir(output_dir)
        end
        if ~isequal(exist(fullfile(output_dir,method_name), 'dir'),7) 
            mkdir(fullfile(output_dir,method_name))
        end
            song_name = songs_list(song_index+2).name;
            ff = strcat(song_name,'_',test_cases(t),'_results.mat');
            results_file = fullfile(output_dir,method_name,ff{1});
            results = results_set{song_index};
            save(results_file,'results')
            result.song_index.results =results_set{song_index};
        %end
        %end
        end
        end
    %end
    %save the results for the method - This runs in parallel so comment
    %this part.
    %result_file = fullfile(base_estimates_directory,[method_name,'.mat']);
    %save(result_file,'result')
    %warning('on','all')
%end


%Now, bsseval code
function [SDR,ISR,SIR,SAR,Gj,G] = bss_eval(ie,i,win,ove,Gj,G)

[nsampl,~,nsrc] = size(ie);
nwin = floor((nsampl-win+1+ove)/ove);
SDR = zeros(nsrc,nwin);
ISR = zeros(nsrc,nwin);
SIR = zeros(nsrc,nwin);
SAR = zeros(nsrc,nwin);
if (nargin == 4) || isempty(Gj) || isempty(G)
    Gj = cell(nsrc,nwin);
    G = cell(nwin,1);
end
for k = 1:nwin
    K = (k-1)*ove+1:(k-1)*ove+win;
    [SDR(:,k),ISR(:,k),SIR(:,k),SAR(:,k),Gj_k,G_k] = bss_eval_images(ie(K,:,:),i(K,:,:),Gj(:,k),G{k});
    Gj(:,k)=Gj_k;
    G{k}=G_k;
end

function [SDR,ISR,SIR,SAR,Gj,G] = bss_eval_images(ie,i,Gj,G)
nsrc = size(ie,3);
if nargin == 2
    %if the G matrix is not given, initialize it to empty
    Gj = cell(nsrc,1);
    G = [];
end
SDR = zeros(nsrc,1);
ISR = zeros(nsrc,1);
SIR = zeros(nsrc,1);
SAR = zeros(nsrc,1);
for j = 1:nsrc
    [s_true,e_spat,e_interf,e_artif,Gj_temp,G] = bss_decomp_mtifilt(ie(:,:,j),i,j,512,Gj{j},G);
    Gj{j}=Gj_temp;
    [SDR(j,1),ISR(j,1),SIR(j,1),SAR(j,1)] = bss_image_crit(s_true,e_spat,e_interf,e_artif);
end

function [s_true,e_spat,e_interf,e_artif,Gj,G] = bss_decomp_mtifilt(se,s,j,flen,Gj,G)

nchan = size(se,2);
s_true = [s(:,:,j);zeros(flen-1,nchan)];
[e_spat,Gj] = project(se,s(:,:,j),flen,Gj);
e_spat=e_spat-s_true;
[e_interf,G] = project(se,s,flen,G);
e_interf=e_interf-s_true-e_spat;
e_artif = [se;zeros(flen-1,nchan)]-s_true-e_spat-e_interf;

function [sproj,G] = project(se,s,flen,G)
warning('off','all')
[nsampl,nchan,nsrc] = size(s);
s = reshape(s,[nsampl,nchan*nsrc]);
s = [s;zeros(flen-1,nchan*nsrc)];
se = [se;zeros(flen-1,nchan)];
fftlen = 2^nextpow2(nsampl+flen-1);
sf = fft(s',fftlen,2);
sef = fft(se',fftlen,2);
if isempty(G)
    G = zeros(nchan*nsrc*flen);
    for k1 = 0:nchan*nsrc-1
        for k2 = 0:k1
            ssf = sf(k1+1,:).*conj(sf(k2+1,:));
            ssf = real(ifft(ssf));
            ss = toeplitz(ssf([1,fftlen:-1:fftlen-flen+2]),ssf(1:flen));
            G(k1*flen+1:k1*flen+flen,k2*flen+1:k2*flen+flen) = ss;
            G(k2*flen+1:k2*flen+flen,k1*flen+1:k1*flen+flen) = ss';
        end
    end
end
D = zeros(nchan*nsrc*flen,nchan);
for k = 0:nchan*nsrc-1
    for i = 1:nchan
        ssef = sf(k+1,:).*conj(sef(i,:));
        ssef = real(ifft(ssef,[],2));
        D(k*flen+1:k*flen+flen,i) = ssef(:,[1,fftlen:-1:fftlen-flen+2])';
    end
end

C = G\D;
C = reshape(C,flen,nchan*nsrc,nchan);
sproj = zeros(nsampl+flen-1,nchan);
for k = 1:nchan*nsrc
    for i = 1:nchan
        sproj(:,i) = sproj(:,i)+fftfilt(C(:,k,i),s(:,k));
    end
end

function [SDR,ISR,SIR,SAR] = bss_image_crit(s_true,e_spat,e_interf,e_artif)

s_true = s_true(:);
e_spat = e_spat(:);
e_interf = e_interf(:);
e_artif = e_artif(:);
SDR = 10*log10(sum(s_true.^2)/sum((e_spat+e_interf+e_artif).^2));
ISR = 10*log10(sum(s_true.^2)/sum(e_spat.^2));
SIR = 10*log10(sum((s_true+e_spat).^2)/sum(e_interf.^2));
SAR = 10*log10(sum((s_true+e_spat+e_interf).^2)/sum(e_artif.^2));
