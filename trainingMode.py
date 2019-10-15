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

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
FREQRANGE = 4000
expectedVoiceRange=2000

for k in range(1,100):
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

    statData = open('statdata.txt', 'r')
    previous_data = (statData.read()).split('\n')
    statData.close()

    if len(previous_data)==0:
        previous_data==[0,0,0,0,4000,0,10000000000,0]
    previous_data_size = int(previous_data[0])
    previous_mean = float(previous_data[1])
    previous_variance = int(float(previous_data[2]))
    mean_max = int(float(previous_data[3]))
    mean_min = int(float(previous_data[4]))
    variance_max = int(float(previous_data[5]))
    variance_min = int(float(previous_data[6]))
    trainingSampleNumber = int(previous_data[7]) + 1
    print('sample is = ' + str(trainingSampleNumber - 1))

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    extracter = []
    counter = 0
    for i in range(100, expectedVoiceRange):
        if outAmp[i] > 1000000:
            tempList=[i,outAmp[i]]
            extracter.append(tempList)
    extracter =sorted(extracter, key=itemgetter(1), reverse=True)
    if (len(extracter)>20):
        extracter=extracter[:20]
    if (len(extracter)) !=0:
        print("fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuck")
        mean = 0
        maxFrequency = 0
        varianceFreq = 0
        size = len(extracter)
        for j in range(0, size):
            freqVal = extracter[j][0]
            mean += freqVal
            varianceFreq += freqVal ** 2
            if freqVal > maxFrequency:
                maxFrequency = freqVal
        if size == 0:
            varianceFreq = 0
            mean = 0
        else:
            mean = mean / size
            varianceFreq = (varianceFreq / size) - mean ** 2

        if mean_max<mean:
            mean_max=mean
        elif mean_min>mean  or mean_min==0:
            mean_min=mean
        if variance_max<varianceFreq:
            variance_max=varianceFreq
        elif varianceFreq < variance_min or variance_min==0:
            variance_min= varianceFreq

        current_freq_size = previous_data_size + size
        current_mean = (mean * size + previous_mean* previous_data_size) / (size + previous_data_size)
        current_variance = (previous_data_size * (previous_variance + ((previous_mean - current_mean) ** 2) + size * (
                    varianceFreq + (mean - current_mean) ** 2))) / (size + previous_data_size)
        print('extracter is ' + str(len(extracter)))
        statData = open('statdata.txt', 'w')
        statData.write(str(size + previous_data_size))
        statData.write('\n')
        statData.write(str(mean))
        statData.write('\n')
        statData.write(str(varianceFreq))
        statData.write('\n')
        statData.write(str(mean_max))
        statData.write('\n')
        statData.write(str(mean_min))
        statData.write('\n')
        statData.write(str(variance_max))
        statData.write('\n')
        statData.write(str(variance_min))
        statData.write('\n')
        statData.write((str(trainingSampleNumber)))
        statData.write('\n')
        statData.close()

        freqPeak = extracter[0][0]
        print(freqPeak)
        print(len(extracter))
    else:
        print('nothing recorded form this voice cut ........... !! ')
        print("====================================================")
    #
    # print ("length of the ex =" +str(len(extracter)/2))
    # print('the number freq stored       = ' + str(size))
    # print('the mean                     = ' + str(mean))
    # print('the maximum frequency        = ' + str(maxFrequency))
    # print('the variance of the data set = ' + str(varianceFreq))
    #
    # print('total readings               = ' + str(current_freq_size))
    # print('the new mean                 = '+ str(current_mean))
    # print('the new variance             = ' +str(current_variance))
    #
    # print("the minimum f mean           = "+str(mean_min))
    # print("the maximum f mean           = " + str(mean_max))
    # print ("the min variance            = " + str (variance_min))
    # print ('the maximum variance        = ' + str(variance_max))



    # (rate, sig) = wav.read("sample.wav")
    # mfcc_feat = mfcc(sig, rate)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig, rate)
    # neededCoeff=fbank_feat[1:3, :]
    # print('the length ' + str(len(neededCoeff)))
    # t = (str(neededCoeff[0]).split(' '))[1:-1]
    #
    # for i in range (len(t)):
    #     k=t(i)
    #     #t(i)=float(k)
    # print(t)