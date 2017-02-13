load('DataTree.mat'); %240 test cases
K=[1,5];

cases = zeros(3,80,4); %3x80
SDR_c = zeros(3,2); %mean and variance for the 3 cases: BEAT, F0, MIDI
SAR_c = zeros(3,2); %mean and variance for the 3 cases: BEAT, F0, MIDI
SIR_c = zeros(3,2); %mean and variance for the 3 cases: BEAT, F0, MIDI
ISR_c = zeros(3,2); %mean and variance for the 3 cases: BEAT, F0, MIDI
SDR_c_n = zeros(3,2);
for j=1:3
 idx = 1;
 for i=1:10
     for k=1:2
         for s=1:4
             cases(j,idx,1)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SDR;
             cases(j,idx,2)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SAR;
             cases(j,idx,3)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SIR; 
             cases(j,idx,4)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.ISR; 
             idx = idx + 1;
         end
     end
 end
 SDR_c(j,1) = mean(cases(j,:,1));
 SDR_c(j,2) = std(cases(j,:,1));
 SAR_c(j,1) = mean(cases(j,:,2));
 SAR_c(j,2) = std(cases(j,:,2));
 SIR_c(j,1) = mean(cases(j,:,3));
 SIR_c(j,2) = std(cases(j,:,3));
 ISR_c(j,1) = mean(cases(j,:,4));
 ISR_c(j,2) = std(cases(j,:,4));
 if j > 1
    SDR_c_n(j,1) = mean(cases(j,:,1)-cases(1,:,1));
    SDR_c_n(j,2) = std(cases(j,:,1)-cases(1,:,1));
 end
end

% clrCerulean = [0.8, 1.0, 0.8];
% clrOrangeRed = [0.2, 0.2, 1.0];
% clrOliveGreen = [0.5, 0.0, 0.0];
clrCerulean = [1.0, 1.0, 0.8];
clrOrangeRed = [1.0, 0.50, 0.30];
clrOliveGreen = [0.0, 0.18, 0.18];
figure
hBar=bar([1,2,3],SDR_c(:,1),'BarWidth',0.4);hold on
errorbar([1,2,3],SDR_c(:,1),SDR_c(:,2),'kx','LineWidth',2);
hBarChildren = get(hBar, 'Children');
myBarColors = [clrCerulean; clrOrangeRed; clrOliveGreen];
index = [1 2 3];
set(hBarChildren, 'CData', index);
colormap(myBarColors);
%errorbar([1,2,3],SDR_c_n(:,1),SDR_c_n(:,2),'bx' )
ylim([0,10])
xlim([0.5,3.5])
% display a colorbar
cb_ax = colorbar;
% label it appropriately
set(cb_ax, 'YTick', [1:3]*3/4+5/8, 'YTickLabels', {'BEAT', 'IMGF0', 'MIDI'});


% kcases_SDR = zeros(2,10,12); 
% kcases_SAR = zeros(2,10,12); 
% kcases_SIR = zeros(2,10,12); 
% kcases_ISR = zeros(2,10,12); 
% SDR_k = zeros(2,4,30); %mean and variance for the 1 and 5 filters
% 
% for k=1:2
%  for s=1:4
%      idx = 1;
%      for j=1:3
%         for i=1:10   
%              kcases_SDR(k,s,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SDR;
%              kcases_SAR(k,s,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SAR;
%              kcases_SIR(k,s,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SIR; 
%              kcases_ISR(k,s,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.ISR; 
%              idx = idx + 1;
%          end
%      end
%      SDR_k(k,s,1) = mean(kcases_SDR(k,s,:));
%      SDR_k(k,s,2) = std(kcases_SDR(k,s,:));
%  end
% end
% 
% figure
% for s=1:4
%     errorbar([s,s+0.1],SDR_k(:,s,1),SDR_k(:,s,2),'bx' );hold on
% end
% ylim([0,10])


icases_SDR = zeros(4,3,20); 
icases_SAR = zeros(4,3,20); 
icases_SIR = zeros(4,3,20); 
icases_ISR = zeros(4,3,20); 
SDR_i = zeros(4,3,2); %mean and variance for the 4 instruments

for s=1:4
 for j=1:3
    idx = 1;
    for k=1:2
         for i=1:10
             icases_SDR(s,j,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SDR;
             icases_SAR(s,j,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SAR;
             icases_SIR(s,j,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.SIR; 
             icases_SIR(s,j,idx)=DataTree{i}.branch{K(k)}.branch{j}.branch{s}.ISR; 
             idx = idx + 1;
         end
    end
    SDR_i(s,j,1) = mean(icases_SDR(s,j,:));
    SDR_i(s,j,2) = std(icases_SDR(s,j,:));
 end
end

figure
clrCerulean = [1.0, 1.0, 0.8];
clrOrangeRed = [1.0, 0.50, 0.30];
clrOliveGreen = [0.0, 0.18, 0.18];
for s=1:4
    hBar=bar([s,s+0.2,s+0.4],SDR_i(s,:,1),'BarWidth',0.9);hold on
    errorbar([s,s+0.2,s+0.4],SDR_i(s,:,1),SDR_i(s,:,2),'kx','LineWidth',2);hold on
    hBarChildren = get(hBar, 'Children');
    myBarColors = [clrCerulean; clrOrangeRed; clrOliveGreen];
    index = [1 2 3];
    set(hBarChildren, 'CData', index);
    colormap(myBarColors);
end
% display a colorbar
cb_ax = colorbar;
% label it appropriately
set(cb_ax, 'YTick', [1:3]*3/4+5/8, 'YTickLabels', {'BEAT', 'IMGF0', 'MIDI'});
ylim([0,10])
xlim([0.7,4.8])
