function evaluate_SS_iKala(id,ruta,method)

warning off;

%% LIBRARY ADDING
%change the working folder add subfolders to path
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
addpath(genpath(fullfile('bss_eval')));

%% parse id to variables
%methods = {'fft_2048'};
outname = {'voice','music'};
maxDir=252;
testid = floor((id-1)/maxDir)+1;
dirid = mod((id-1),maxDir)+1;

%% GET ALL DIRS to do get all files in dir and divide with MaxInst
nameFolds = dir(fullfile(ruta,'Wavfile','*.wav'));
filename=nameFolds(dirid).name;

ruta_original=fullfile(ruta,'Wavfile');
ruta_wav=fullfile(ruta,'output',method);
outpath=fullfile(ruta,'measures',['test_' method]);
if ~isequal(exist(fullfile(ruta,'measures'), 'dir'),7) 
    mkdir(fullfile(ruta,'measures'))
end
if ~isequal(exist(outpath, 'dir'),7) 
    mkdir(outpath)
end


estimated_voice=fullfile(ruta_wav,strrep(filename,'.wav',['-' outname{1} '.wav']));
estimated_music=fullfile(ruta_wav,strrep(filename,'.wav',['-' outname{2} '.wav']));
mfile = fullfile(outpath,strrep(filename,'.wav','.mat'));
if isequal(exist(estimated_voice, 'file'),2)
    if ~isequal(exist(mfile, 'file'),2) %only run if the file does not exist    
        
        [estimatedVoice,fs1] = wavread(estimated_voice);
        [estimatedKaraoke,fs2] = wavread(estimated_music);
        
        source_file = fullfile(ruta_original,filename);
        [source,fs3] = wavread(source_file);
   
        trueVoice = source(:,2);
        trueKaraoke = source(:,1);
        trueMixed = (trueVoice + trueKaraoke)/2;
        clear source;
        
        if length(estimatedVoice)>length(trueVoice)
            estimatedVoice=estimatedVoice(1:length(trueVoice));
        end
        if length(estimatedKaraoke)>length(trueKaraoke)
            estimatedKaraoke=estimatedKaraoke(1:length(trueKaraoke));
        end
        
        [SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / norm(estimatedVoice + estimatedKaraoke), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
	    [NSDR, NSIR, NSAR] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
        NSDR = SDR - NSDR;
        NSIR = SIR - NSIR;
        NSAR = SAR - NSAR;        
        
        save(mfile,'NSDR','NSIR','NSAR','SDR','SIR','SAR');
        
    end
end
