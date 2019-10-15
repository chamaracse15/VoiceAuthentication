from typing import Dict, Any
import wave
import numpy as np
from operator import itemgetter
import scipy.io.wavfile as wav
import pyaudio


tempLis=[]
typesIn=['mode Switch : ','ready to get input : ']
for i in range(2):
    tempLis.append(int(input(typesIn[i])))



mode_switch=tempLis[0]
readytoGetInput=tempLis[1]
#gettingInputs=1

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
FREQRANGE = 4000
expectedVoiceRange=2000

def testMode(): # this is the testing mode.
    #L2 is off in the testing mode
    if readytoGetInput == 1:


        # L1 is also made on to indicate recording

        # record the audio input stream for 2 seconds

        WAVE_OUTPUT_FILENAME = "sample.wav"
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print("recording....")
        #recording starts here
        frames = []
        data = np.fromstring(stream.read(RATE * RECORD_SECONDS),
                             dtype=np.int16)  # read data 2 sseconds instead of chunk
        peak = np.average(np.abs(data)) * 2
        frames.append(data)

        datas = data * np.hanning(len(data))  # smooth the FFT by windowing data
        fft = abs(np.fft.fft(datas).real)
        fft = fft[:int(len(fft) / 2)]

        freq = np.fft.fftfreq(RATE * RECORD_SECONDS, 1.0 / RATE)  # check2 seconds frequency to find the maximum

        outFreq = (freq.tolist())[2:1 + expectedVoiceRange * 2:2]
        outAmp = (fft.tolist())[2:1 + expectedVoiceRange * 2:2]

        #recording is terminated
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
        extractAmp = []
        for i in range(100, expectedVoiceRange):
            if outAmp[i] > 1000000:
                tempList = [i, outAmp[i]]
                extracter.append(tempList)
        extracted = sorted(extracter, key=itemgetter(1), reverse=True)
        if (len(extracter) > 20):
            extracter = extracter[:20]
        if (len(extracter)) !=0:
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

            if len(previous_data) < 8:
                previous_data = [0, 0, 0, 0, 0, 0, 0, 0]

            # print(previous_data)
            previous_data_size = int(previous_data[0])
            previous_mean = float(previous_data[1])
            previous_variance = int(float(previous_data[2]))
            mean_max = int(float(previous_data[3]))
            mean_min = int(float(previous_data[4]))
            variance_max = int(float(previous_data[5]))
            variance_min =int(float(previous_data[6]))
            passLevel = 0
            threshold =10 # total pass score
            # if (mean_max>mean>mean_min) and (variance_max>varianceFreq>variance_min):
            #   passLevel=1
            if mean_max - mean<10:
                passLevel += 2
                print("mean_max pssed")

            elif mean-mean_min<10:
                passLevel += 2
                print("mean_min passed")

            if variance_max > varianceFreq:
                passLevel += 3
                print("var_max passed")
            if variance_min<varianceFreq:
                passLevel+=3
                print("var_max passed")
            print("pass level = "+str(passLevel))

            # L1 is off
            # L3 is also off
            # to say sound is recorded blink L6 for 2 seconds

            # checking values with pre recorded data
            if passLevel>=threshold-3:
                print ("passed")
              #  GPIO.setup(2, GPIO.HIGH)
                #now he has the access to the system
                #blink L4 to denote Access granted
            else:
                print("failed")

                #access denied
                # blink L5 to denote access denied

def trainMode(): # this is to select the mode. if the switch is on then we get the training mode
    #training mode is to be on and L2 is on
    if readytoGetInput==1:
       # GPIO.setup(4, GPIO.HIGH)
        # push button input is taken
        #on the L3 led
        #L1 is also made on to indicate recording

        #record the audio input stream for 2 seconds

        #L1 is off
        #L3 is also off
        #store the values to calculate the ratios

        WAVE_OUTPUT_FILENAME = "sample.wav"
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print('recording..training data sample.... ')
        frames = []
        data = np.fromstring(stream.read(RATE * RECORD_SECONDS),
                             dtype=np.int16)  # read data 2 sseconds instead of chunk
        peak = np.average(np.abs(data)) * 2
        frames.append(data)

        datas = data * np.hanning(len(data))  # smooth the FFT by windowing data
        fft = abs(np.fft.fft(datas).real)
        fft = fft[:int(len(fft) / 2)]

        freq = np.fft.fftfreq(RATE * RECORD_SECONDS, 1.0 / RATE)  # check2 seconds frequency to find the maximum

        outFreq = (freq.tolist())[2:1 + expectedVoiceRange * 2:2]
        outAmp = (fft.tolist())[2:1 + expectedVoiceRange * 2:2]

        #recording terminated
        stream.stop_stream()
        stream.close()
        audio.terminate()

        statData = open('statdata.txt', 'r')
        previous_data = (statData.read()).split('\n')
        statData.close()

        if len(previous_data) == 0:
            previous_data == [0, 0, 0, 0, 0, 0, 0, 0]
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
                tempList = [i, outAmp[i]]
                extracter.append(tempList)
        extracter = sorted(extracter, key=itemgetter(1), reverse=True)
        if (len(extracter) > 20):
            extracter = extracter[:20]
        if (len(extracter)) != 0:

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

            if mean_max < mean:
                mean_max = mean
            elif mean_min > mean or mean_min==1:
                mean_min = mean
            if variance_max < varianceFreq:
                variance_max = varianceFreq
            elif abs(current_variance-varianceFreq)<variance_max-varianceFreq :
                variance_min = varianceFreq

            current_freq_size = previous_data_size + size
            current_mean = (mean * size + previous_mean * previous_data_size) / (size + previous_data_size)
            current_variance = (previous_data_size * (
                        previous_variance + ((previous_mean - current_mean) ** 2) + size * (
                        varianceFreq + (mean - current_mean) ** 2))) / (size + previous_data_size)

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
            print('sample recorded')
            print(str(variance_min),str(mean_min))


if mode_switch==1:
    for num in range(25):
        trainMode()
else:
    testMode()
