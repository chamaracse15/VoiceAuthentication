import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 4096
RECORD_SECONDS = 20


p=pyaudio.PyAudio()

WAVE_OUTPUT_FILENAME = "strokee.wav"

audio = pyaudio.PyAudio()


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# start to listning to input

print("Listninig...")
frames = []

for i in range(0,10):
    data = np.fromstring(stream.read(RATE*2), dtype=np.int16)  # read data 2 sseconds instead of chunk
    peak = np.average(np.abs(data)) * 2
    frames.append(data)

    datas = data * np.hanning(len(data)) # smooth the FFT by windowing data
    fft = abs(np.fft.fft(datas).real)
    fft = fft[:int(len(fft)/2)]
    freq = np.fft.fftfreq(RATE*2,1.0/RATE)  #check2 seconds frequency to find the maximum
    #freq = freq[:int(len(freq)/2)]
    freqPeak = freq[np.where(fft==np.max(fft))[0][0]]+1

    print("peak frequency: %d Hz"%freqPeak)


    plt.plot(freq,fft)
    # plt.axis([0,4000,None,None])
    plt.axis([0,4000,None, None])
    plt.show()
    plt.close()




print("Finished Listning...")

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()

#stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()