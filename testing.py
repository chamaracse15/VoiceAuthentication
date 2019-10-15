from typing import Dict, Any
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
FREQRANGE = 4000
expectedVoiceRange=2000

WAVE_OUTPUT_FILENAME = "sample.wav"
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listeninig... . . . . .")
frames = []
data = np.fromstring(stream.read(RATE*RECORD_SECONDS), dtype=np.int16)  # read data 2 sseconds instead of chunk
peak = np.average(np.abs(data)) * 2
frames.append(data)

datas = data * np.hanning(len(data)) # smooth the FFT by windowing data
fft = abs(np.fft.fft(datas).real)
fft = fft[:int(len(fft)/2)]

freq = np.fft.fftfreq(RATE * RECORD_SECONDS, 1.0 / RATE)  # check2 seconds frequency to find the maximum

outFreq = (freq.tolist())[2:1+expectedVoiceRange*2:2]
outAmp =(fft.tolist())[2:1+ expectedVoiceRange*2:2]

print("Finished Listening...")
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

extracter = []
counter = 0
extractAmp=[]
for i in range(100, expectedVoiceRange):
    if outAmp[i] > 1000000:
        tempList=[i,outAmp[i]]
        extracter.append(tempList)
extracted = sorted(extracter, key=itemgetter(1), reverse=True)
if (len(extracter)>20):
    extracter=extracter[:20]
mean = 0
maxFrequency = extracter[0]
varianceFreq = 0

size = len(extracter)
for j in range(0, size):
    freqVal = extracter[j][0]
    mean += freqVal
    varianceFreq += freqVal ** 2
if size == 0:
    varianceFreq = 0
    mean = 0
else:
    mean = mean / size
    varianceFreq = (varianceFreq / size) - mean ** 2

statData = open('statdata.txt', 'r')
previous_data = (statData.read()).split('\n')
statData.close()
open('statdata.txt', 'w').close()

if len(previous_data)<8:
    previous_data=[0,0,0,0,mean,0,varianceFreq,0]

#print(previous_data)
previous_data_size = int(previous_data[0])
previous_mean = float(previous_data[1])
previous_variance = int(float(previous_data[2]))
mean_max = int(float(previous_data[3]))
mean_min = int(float(previous_data[4]))
variance_max = int(float(previous_data[5]))
variance_min = int(float(previous_data[6]))

passLevel=0
#if (mean_max>mean>mean_min) and (variance_max>varianceFreq>variance_min):
 #   passLevel=1
if mean_max>mean:
    passLevel+=1
    print('max_freq passed')
elif mean_min<mean or mean_min==0:
    passLevel+=1
    print('low freq passed')
if variance_max>varianceFreq:
    passLevel+=1
    print('var max passed')
elif varianceFreq > variance_min or variance_min==0:
    passLevel+=1
    print('var low passed')

(rate,sig) = wav.read("sample.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])



print (passLevel)