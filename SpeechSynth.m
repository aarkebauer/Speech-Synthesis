% Speech Synthesis System for Final Project
% EENG 450
% Andrew Arkebauer
% 12/3/2016

%{
This program works in four primary functions. The first (the 'SpeechSynth'
function), is simply a menu which gives the user the choice to record
speech, analyze speech, or synthesize speech. When the user analyzes and
then synthesizes speech, the following are performed three times: once to
generate the first harmonic (Fp), once for the one half harmonic (0.5*Fp)
and once for the second harmonic (2*Fp).


The second function, SpeechAcq, allows the user to record a 2-second
segment of speech which is then saved as a file, 'speechSample.txt'.


The third function, SpeechAnalysis, splits this file up into 40 400-element
segments. It reads parameters specified in 'speechParams.txt' for AR system
order, maximum and minimum frequency of allowed pole locations, energy
threshold, and pitch period threshold. Using the getC and getRTheta
functions, it determines the pole locations (r and theta values) for an
AR filter corresponding to each of the 40 segments, filtering out those
above the specified maximum frequency and below the minimum. These values
are then "transmitted" to a receiver along with necessary PP, Energy, and
amplitude values, which generates the AR filter coefficients for the given
pole locations and the pulse train input. If the segment has energy lower
than the defined threshold, the speech synthesized is just a string of 400
zeros. If the energy is sufficient, the function then calculates the
pitch period of the appropriate segment using cepstral analysis and calls
the 'synth' function on the result in order to generate a speech segment.
THis synthesizes the speech segment by running a unit pulse sequence of
period equal to the pitch period/(harmonic number) through an AR filter
with the specified coefficients. If the pitch period of the segment is
below the threshold (unvoiced), this function passes IID through the AR
filter. The resulting segments are concatenated to form a 2-second segment of
speech that mimics the user input.

In order to smooth the output waveform, I varied the amplitude of the
synthesized segments in order to prevent amplitude discontinuities between
segments. These make the output sound very choppy and thus unrealistic. I
did so by keeping track of the amplitude of the last 50 samples in each of
the previous segments. I then divided the first 250 samples of the segment
into 5 "sections" of 50 samples each. I multiplied each section by a factor
that varied linearly and resulted in the first such section being the same
amplitude of the end of the previous segment, and the fifth section being
the same amplitude as the maximum amplitude of samples 250-300 in the
current segment. This improved the output speech segment noticably.

I experimented with varying the pitch period within segments in a similar
manner, but the results sounded very poor in comparison. As such, I
neglected to vary pitch period within and between segments.

Finally, the number of bytes "transmitted" (r, theta, PP, Energy, and 
amplitude values) to produce the synthesized speech is output to the user
for comparison to the number of bytes necessary to stream the speech.


The final function, SpeechSynthesis, simply outputs the sound corresponding
to the generated speech segment and plots the resulting waveforms.
%}

function SpeechSynth
clear; format compact;
global nseg; nseg=400;
global ham; ham = 0.54 - 0.46 * cos(2*pi*(0:nseg-1)/(nseg-1));
global freq; freq = 8000;
global nBytes; nBytes = 0;
exit = 0;

speechAnalyzed = 0; % generate error message if you don't analyze speech before synthesizing

fileName = 'speechSample.txt';

while exit == 0
    choice = menu('Choose to Load a Speech Sample or Acquire a New One, then Analyze, then Synthesize','Load Sample "We Were..."','Load Sample "Luke I am..."','Load Sample "Heave the Line..."','Speech Acquisition','Speech Analysis','Speech Synthesis', 'Exit');
    switch choice
        case 1
            fileName = 'speechSampleWeWere.txt'; % speechAnalysis will load this pre-made speech sample
        case 2
            fileName = 'speechSampleLuke.txt';   % speechAnalysis will load this pre-made speech sample
        case 3
            fileName = 'speechSampleHeave.txt';  % speechAnalysis will load this pre-made speech sample
        case 4
            speechAcq;
            fileName = 'speechSample.txt';       % speechAnalysis will load the newly acquired speech sample
        case 5
            % analyze speech to generate a segment for each harmonic (Fp, 0.5*Fp, 2*Fp)
            harmonics = [1 .5 2];
            for i=1:3
                nBytes = 0;
                [original, out(i,:)] = speechAnalysis(harmonics(i), fileName);
            end
            out = [original'; out];
            speechAnalyzed = 1;
        case 6
            if speechAnalyzed
                % play and plot speech segment for each harmonic (Fp, 0.5*Fp, 2*Fp)
                for i=1:4
                    speechSynthesis(out(i,:), i);
                    pause(2)
                end
            else
                disp('Must Analyze Speech First!')
            end
        case 7
            exit = 1;
    end % switch case
end % while
end % main



% Allow user to record a 2-second sample of speech, save for later analysis
% and synthesis
function speechAcq
global freq
% Acquire and save microphone speech
recObj = audiorecorder(freq,8,1);   % sets mic as input
                                    % Fs=8000Hz, 8bits, one channel
input('Press Enter to record audio')
disp('Start speaking now')          % prompt speaker
recordblocking(recObj, 2.05);          % records from mic for 2 sec (first 0.05 seconds deleted to remove transient, so record for 2.05 to make final sample 2 seconds long - won't start speaking in first 0.05 seconds anyway)
disp('End of recording');           % indicate end
signal = getaudiodata(recObj)';     % write data in real-valued array

signal = signal(401:end); % delete transient at beginning


plot(signal);                       % plot the two-sec waveform
axis off; grid off
soundsc(signal)                     % plays a vector using default values (8000,8)


fid = fopen('speechSample.txt','w');    % open file for writing
for i = 1:length(signal)
    fprintf(fid,'%6.2f\n',signal(i)); % write using LF separator
end
fclose(fid);                        % close file
end



% Analyze speech file generated by user in 40 400-element segments.
% Generate coefficients and pitch period for each, then synthesize
% corresponding speech using the 'synth' function. The output 'out' is a
% 16000-element speech waveform which is played for the user by
% 'speechSynthesis'. The input 'harmonic' determines which harmonic of the
% speech to generate (i.e. Fp, 0.5*Fp, or 2*Fp).
function [original, out] = speechAnalysis(harmonic, fileName)
global ham nseg freq nBytes

%% read parameters from speechParams.txt

fid = fopen('speechParams.txt','r');
params = fscanf(fid,'%f\n');
fclose(fid);
if harmonic == 1
    disp('parameter values read from SpeechParams.txt')
    nOrder = params(1)
    E_thresh = params(2)
    P_thresh = params(3)
    fmin = params(4)
    fmax = params(5)
else
    nOrder = params(1);
    E_thresh = params(2);
    P_thresh = params(3);
    fmin = params(4);
    fmax = params(5);
end

%% read speech data from speechSample.txt or a pre-made file, whatever the user selects

fid = fopen(fileName,'r');
data = fscanf(fid,'%f\n'); % read in phoneme data file (e.g. ah.txt, etc.)
fclose(fid);
original = data; % store original spoken speech sample as 'original'


%% split signal (from speechSample.txt) up into 40 400-element segments

dataSplit = zeros((2*freq)/nseg,nseg);
    
for ii=1:length(dataSplit(:,1)) % for each of 40 segments
    dataSplit(ii,:) = data((1 + nseg*(ii-1)):(nseg*(ii-1) + nseg)); % rows of dataSplit are segments
end % for


%% segment-by-segment analysis and synthesis
% initializations
PP = zeros(length(dataSplit(:,1)),1); % column vector to store 40 pitch periods (one for each segment)
averageAmp = zeros(length(dataSplit(:,1)),1); % column vector to store 40 average amplitudes (one for each segment)

E_sigHam = zeros(length(dataSplit(:,1)),1); % column vector to store 40 segment energies

for ii=1:length(dataSplit(:,1)) % for ii=1:(number of 400-element segments in the recording)
    
    segment = dataSplit(ii,:); % segment is the 400-element segment being analyzed
    
    maxSegAmp = max(abs(segment)); % find average amplitude of entire signal for scaling of individual segments
    
    segment = segment - mean(segment);
    sigHam = (segment.*ham); % get rid of edge effects in the segment by applying Hamming window
    
    % keep track of the ampitude of the last segment, for use in Amplitude
    % Smoothing (below)
    if ii > 1
        lastSeg = dataSplit(ii-1,:);
        prevAmp = max(abs(lastSeg(nseg-200:nseg-1)));
    end
    
    % determine energy of the Hamming-windowed segment
    E_sigHam(ii) = sum(sigHam.^2);
    
    if E_sigHam(ii) > E_thresh % output will be all zeros if insufficient energy, otherwise will be synthesized speech segment

        PP(ii) = cepstrum(segment); % determine pitch period from the cepstrum
        averageAmp(ii) = mean(abs(segment)); % record average amplitude for later smoothing
        
        C = getC(segment, nOrder); % generate inverse filter so that can get AR filter coefficients
        % calculate pole locations in order to get AR filter coefficients of segment - filter out poles with frequencies > fmax or < fmin
        [r, theta] = getRTheta(C,fmin,fmax);
        nBytes = nBytes + length(r) + length(theta); % record number of transmitted bytes
        
        % also would have to transmit PP and E_sigHam in order to reconstruct
        % each of the 40 segments accurately, and six amplitude values per
        % segment in order to appropriately smooth the amplitudes between
        % segments (shown below)
        nBytes = nBytes + 8;
        
        %%% "transmission" of r, theta values occurs here
        
        AC = getCoeffs(r, theta); % get AR filter coefficients of segment
        
        % synthesize speech as either unvoiced (pitch period less than the
        % threshold) or voiced
        if PP(ii) < P_thresh
            segment = synth(PP(ii),AC,harmonic,1); % unvoiced
        else
            segment = synth(PP(ii),AC,harmonic,0); % voiced
        end
        
        % make sure each segment is 0-mean
        segment = segment - mean(segment);
        
        
        %% Amplitude smoothing
        segment = segment*(maxSegAmp/max(abs(segment)));
        n = 50;
        d = 5;
        
%         split beginning of 400-sample segment into d sections of n
%         samples; vary amplitude in each linearly between the amplitude of
%         the last 100 samples of the previous segment and the maximum
%         amplitude of the current segment
        for i=1:d
            prevSecAmp = max(abs(segment(n*(i-1) + 1:i*n)));
            segment(n*(i-1) + 1:i*n) = segment(n*(i-1) + 1:i*n) * (((maxSegAmp - prevAmp)/(d-1))*(i-1) + prevAmp)/prevSecAmp;
        end
        
        %% put segments back together
        if ii == 1
            out = segment;
        else
            out = [out segment];
        end
        
    else
        % have to transmit a byte to signal that the segment is all zeros
        nBytes = nBytes + 1;
        
        % synthesized segment is all zeros
        if ii == 1
            out = zeros(1,nseg);
        else
            out = [out zeros(1,nseg)];
        end
    end % if E_sigHam > E_thresh
    
    
end % for - analysis of each segment in the recording
disp(['Number of bytes transmitted for ', num2str(harmonic), '*Fp: ', num2str(nBytes)])

end



% speech synthesis simply plays the audio segment generated in
% speechAnalysis for each of the harmonics, and plots the appropriate
% segment
function speechSynthesis(out, i)
    %% Play sound
    soundsc(out)
    
    %% Plot
    figure(1)
    subplot(4,1,i)
    plot(out)
    switch i
        case 1
            title('Original')
        case 2
            title('Synthesized with Fp')
        case 3
            title('Synthesized with 0.5*Fp')
        case 4
            title('Synthesized with 2*Fp')
    end
    grid off; axis off
end



% synthesize phoneme using pitch period (PP), AR filter coefficients (as),
% the harmonic desired (harmonic) and voiced or unvoiced (unvoiced = 0 or
% 1) - if unvoiced, pass IID through AR filter; if voiced, pass string of
% impulses separated by PP zeros
function out = synth(PP,as,harmonic,unvoiced)
global nseg;

PP = floor(PP/harmonic);
h = [1 zeros(1,PP-1)];
numExcitations = ceil(nseg/PP); % use 'ceil', because output waveform appears to be "missing" an impulse at the end of each 400-sample segment

in = repmat(h,1,numExcitations); % 1 by numExcitations matrix having value h (h = 1 followed by (pitch period) - 1 zeros)
length(in);
in = [in zeros(1,nseg-length(in))];
in = in(1:400); % since changed the numExcitations to 'ceil', make sure segment is still 400 segments long

% for unvoiced case pass IID through AR
if unvoiced
    in = randn(1,nseg);
end

% for voiced, pass through AR filter
y_prev = in;


for i=1:length(as) % for each second-order system (in cascade)
    y = AR(as(:,i)',y_prev); % pass through second-order AR filter with appropriate coefficients
    y_prev = y;
end % for
out = y;

end % synth



% determine pitch period (PP) of data
function PP = cepstrum(data)
global ham;
rp = data.*ham;
% % % size(rp)
rp = [rp zeros(1,1000-length(rp))];
fftRes = 20*log10(abs(fft(rp)));
Cs = abs(ifft(fftRes));
% Gather first peak
[~,firstInt] = max(Cs(1:81));
nCs = Cs(1:161);
nCs(1:firstInt+5) = 0;
[~,secondInt] = max(nCs);
PP = abs(secondInt - firstInt);
end % cepstrum



% generate inverse filter coefficients from data
function C = getC(data, order)
global ham;
rp = (data.*ham)';
% Get autocorrelation by inverse power spectrum
RX = ifft(abs(fft(rp)).^2/length(rp));
R = toeplitz(RX(1:order));
g = RX(2:order+1);
h = (inv(R)*g); % linear prediction
C = [1 -h']; % inverse filter of linear prediction
end % getC



% determine pole locations of speech segment coefficients, used to generate
% the coefficients of AR filter for speech synthesis
function [r, theta] = getRTheta(C,fmin,fmax)
global freq

root = roots(C); % roots of C yield poles of system (sorted in frequency order)

theta_prev = 0;
AC =[];
% for i=1:length(C)-1
for i=1:length(root)
    rt = root(i);
    r = abs(rt);
    if (r>1)
        r=1/r;
    end
    
    theta = atan2(imag(rt),real(rt));
    
    if abs(abs(theta) - abs(theta_prev)) < 0.05
        continue % skip all remaining instructions in for loop, go to next iteration of for loop
    end
    
    % filter out poles at frequencies > fmax or < fmin
    if ((freq/2)/pi)*theta > fmax || ((freq/2)/pi)*theta < fmin
        continue
    end
    
    r_store(i) = r;
    theta_store(i) = theta;
    
%     theta_prev = theta;
end % for


r = r_store(r_store ~= 0)';
theta = theta_store(r_store ~= 0)';



end % getRTheta



% generate coefficients for AR filter using transmitted pole locations
function AC = getCoeffs(r,theta)
for i=1:length(r)
    if i==1
        AC(1,i) = 2*r(i)*cos(theta(i));
        AC(2,i) = -(r(i)*r(i));
    else
        AC = [AC [2*r(i)*cos(theta(i));-(r(i)*r(i))]];
    end
end
end



% second order AR filter
function y = AR(a,x)
na = length(a);
mem = zeros(1,na);
y = zeros(1,length(x));
for i=1:length(x)
    y(i) = a*mem' + x(i);
    mem(2:na) = mem(1:na-1);
    mem(1) = y(i);
end % for
end % AR