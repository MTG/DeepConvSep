%---------------------------------------------------
%Needed variables: 1. Path to results folder
%                  2. Output path
%                  3. Output filename (with extension)
%---------------------------------------------------

%Path to results folder
rp = 'results_test/'

%Output path
op = 'results_dB/'

%Output filename
filename = 'results_dB_test.txt'

s = dir(rp);

file_list = {s.name}
file_list = string(file_list(4:end))

sdr_v = [];
isr_v = [];
sir_v = [];
sar_v = [];

for n = 1:length(file_list)
    
    mat = matfile(char(strcat(rp,file_list(n))));
    
    disp(mat.results)
    
    r = mat.results;
    
    sdr_dB = r.vocals.sdr;
    isr_dB = r.vocals.isr;
    sir_dB = r.vocals.sir;
    sar_dB = r.vocals.sar;
    
    disp(r.vocals)
    
    sdr = power(10,sdr_dB/10)
    isr = power(10,isr_dB/10)
    sir = power(10,sir_dB/10)
    sar = power(10,sar_dB/10)
    
    sdr_v = [sdr_v sdr];
    isr_v = [isr_v isr];
    sir_v = [sir_v sir];
    sar_v = [sar_v sar];
    
end

sdr_mean = nanmean(sdr_v)
isr_mean = nanmean(isr_v)
sir_mean = nanmean(sir_v)
sar_mean = nanmean(sar_v)

sdr_mean_dB = 10*log10(sdr_mean)
isr_mean_dB = 10*log10(isr_mean)
sir_mean_dB = 10*log10(sir_mean)
sar_mean_dB = 10*log10(sar_mean)

fid = fopen([op filename],'w');
fprintf(fid,'SDR, ISR, SIR, SAR\n');
fclose(fid);

dlmwrite([op filename],[sdr_mean_dB isr_mean_dB sir_mean_dB sar_mean_dB],'-append');