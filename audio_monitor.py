import pyaudio
import wave
import struct
import numpy as np
import librosa
from matplotlib import pyplot as plt
import IPython.display as ipd
import os
import time


# Set chunk size of 1024 samples per data frame
chunk = 1024
data_format = pyaudio.paInt16
channels = 1
rate = 4096
record_seconds = 10


def record_audio(filename):
    # >>>> recoding a certain length audio >>>>
    p = pyaudio.PyAudio()

    stream = p.open(format=data_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    # >>>> recoding a certain length audio >>>>
    print("* recording")

    frames = []

    # for i in range(0, int(rate / chunk * record_seconds)):
    for i in range(0, 1):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")
    # <<<< recording a certain length audio <<<<

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(data_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def detect_record_audio(filename):
    # >>>> detect and record audio >>>>
    p = pyaudio.PyAudio()

    stream = p.open(format=data_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print(p.get_default_input_device_info())

    flag = False
    pause_num = 0
    # >>>> detect and record audio >>>>
    print("* recording")

    frames = list()
    sentence = list()
    chunk_flags = list()

    # calculate the background sound level
    # the first chunk is abandoned as circuit bias
    background = 0.
    for i in range(20):
        data = stream.read(chunk)
        data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
        data_amplitude = librosa.db_to_amplitude(data_int)
        ave_amplitude = np.average(data_amplitude)
        if 4 < i < 19:
            background += ave_amplitude
        print(ave_amplitude)
    print(background)
    ave_background_level = background/14
    print('the background average level is: %f' % ave_background_level)

    while True:
        try:
            data = stream.read(chunk)
            data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
            # ===================================================================================
            # ===================================================================================
            # struct converts data into numpy for computation, tobytes converts back to raw info #
            # print(data)
            # data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
            # data_bytes = np.ndarray.tobytes(data_int)
            # print(data_bytes)
            # ===================================================================================
            # ===================================================================================
            data_amplitude = librosa.db_to_amplitude(data_int)
            # print_plot_play('test', data_amplitude, rate)
            ave_amplitude = np.average(data_amplitude)

            if flag is False:
                if ave_amplitude > 2*ave_background_level:
                    flag = True
                    pause_num = 0
                    sentence.append(data)
                    chunk_flags.append(1)
            else:
                sentence.append(data)
                if ave_amplitude > 2*ave_background_level:
                    chunk_flags.append(1)
                    pause_num = 0
                else:
                    chunk_flags.append(0)
                    pause_num += 1
                    if pause_num >= 8:
                        # refine the voice by clipping the silent chunks
                        # chunk_flags = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        start = chunk_flags.index(1, 0, len(chunk_flags))
                        chunk_flags.reverse()
                        end = chunk_flags.index(1, 0, len(chunk_flags))
                        frames.append(sentence[start:-end+1])
                        # replace this append with feeding to external perception of hearing
                        sentence = list()
                        chunk_flags = list()
                        flag = False
                        pause_num = 0
        except KeyboardInterrupt:
            break
    print(np.shape(frames))
    print("* done recording")
    # <<<< detect and record audio <<<<

    stream.stop_stream()
    stream.close()
    p.terminate()

    for i in range(len(frames)):
        file = filename + 'color' + str(i+1) + '.wav'
        wf = wave.open(file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(data_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames[i]))
        wf.close()


def load_audio(filename):
    # Open the sound file
    # wf = wave.open(filename, 'r')
    wf = wave.open(filename, 'rb')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                 channels=wf.getnchannels(),
    #                 rate=wf.getframerate(),
    #                 output=True)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    frames = []
    # Play the sound by writing the audio data to the stream
    while True:
        print('reading...')
        # Read data in chunks
        data = wf.readframes(chunk)
        if len(data) == 0:
            break
        # data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
        # data_amplitude = librosa.db_to_amplitude(data_int)
        # ave_amplitude = np.max(data_amplitude)
        # print(ave_amplitude)
        frames.append(data)
    # Close and terminate the stream
    stream.close()
    p.terminate()

    return frames


def play_audio(filename=None, sound_stream=None):

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    if filename is not None:
        # Open the sound file
        wf = wave.open(filename, 'rb')

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        # format=8; channels=1; rate=4096, output=True

        # Play the sound by writing the audio data to the stream
        while True:
            # Read data in chunks
            data = wf.readframes(chunk)
            if len(data) == 0:
                break
            data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
            data_amplitude = librosa.db_to_amplitude(data_int)
            ave_amplitude = np.average(data_amplitude)
            max_amplitude = np.max(data_amplitude)
            # print(ave_amplitude)
            stream.write(data)

    elif sound_stream is not None:
        stream = p.open(format=8, channels=1, rate=4096, output=True)
        for i in range(len(sound_stream)):
            # Read data in chunks
            data = sound_stream[i]
            data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
            data_amplitude = librosa.db_to_amplitude(data_int)
            ave_amplitude = np.average(data_amplitude)
            max_amplitude = np.max(data_amplitude)
            print(ave_amplitude)
            stream.write(data)
    else:
        print('please give at least one source!')
        return
    # Close and terminate the stream
    time.sleep(1)
    stream.close()
    p.terminate()


def clip_audio_files():
    # delete the silent chunks before and after speech in audio files
    source = 'clipped_voices/'
    target = 'clipped_voices/'
    source_files = os.listdir(target)

    # sort and rename the files
    # source_files.sort(key=sort_keys)
    # for i in range(len(source_files)):
    #     os.rename(source+source_files[i], source+'voice'+str(i)+'.wav')

    for i in range(len(source_files)):
        print('%dth file:' % i)
        print('read source audio file...')
        source = source + source_files[i]
        target = target + source_files[i]
        clip_audio(source, target)


def clip_audio(source, target):
    print('read source audio file...')
    wf = wave.open(source, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    frames = []
    chunk_flags = []
    while True:
        data = wf.readframes(chunk)
        if len(data) == 0:
            break
        stream.write(data)
        # play chunk

        # set chunk flags as if its max_amplitude > 1500000
        data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
        data_amplitude = librosa.db_to_amplitude(data_int)
        ave_amplitude = np.average(data_amplitude)
        max_amplitude = np.max(data_amplitude)
        frames.append(data)
        if max_amplitude > 2000000:
            chunk_flags.append(1)
        else:
            chunk_flags.append(0)
    start = chunk_flags.index(1, 0, len(chunk_flags))
    chunk_flags.reverse()
    end = chunk_flags.index(1, 0, len(chunk_flags))
    frames = frames[start:end + 1]
    wf.close()

    print('write to target audio file...')
    wf = wave.open(target, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(data_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    stream.close()
    p.terminate()


def print_plot_play(filename, x, Fs):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player

    Notebook: C1/B_PythonAudio.ipynb

    Args:
        x: Input signal
        Fs: Sampling rate of x
        text: Text to print
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % ('WAV file: ', Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.title(filename)
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    # plt.ylim([-1, 1])
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    # ipd.display(ipd.Audio(data=x, rate=Fs))


def sort_keys(name):
    return int(name[5:-4])


def detector(x):
    # the input x is composed of a number of chunks, of [chunk_num, chunk_size]
    # it is divided from other sentence with 8 or more silent chunks
    # each word in x should be divided by 3 chunks of silence
    # the original sound track in hexadecimal format is converted to integers
    chunk_flags = list()
    word_ends = list()
    words = list()
    # end[i]+3 : end[i+1] is a word
    max_amplitude = 0.
    sound_stream = x.copy()

    for i in range(len(x)):
        data_int = np.array(struct.unpack(str(len(x[i])) + 'B', x[i]), dtype='b')
        # convert strings of hexadecimals into integers
        x[i] = data_int
        data_amplitude = librosa.db_to_amplitude(data_int)
        ave_amplitude = np.average(data_amplitude)
        max_amplitude = np.max(data_amplitude)
        if ave_amplitude > 33000:
            chunk_flags.append(1)
        else:
            chunk_flags.append(0)

    word_ends.append(chunk_flags.index(1, 0, len(chunk_flags)) - 4)
    # first virtual word end is determined by the start of words
    for i in range(len(chunk_flags)):
        if i < len(chunk_flags) - 3 and chunk_flags[i] == 1 and \
                chunk_flags[i + 1] + chunk_flags[i + 2] + chunk_flags[i + 3] == 0:
            word_ends.append(i)

    reverse_chunks = chunk_flags.copy()
    reverse_chunks.reverse()
    last_end = len(chunk_flags) - reverse_chunks.index(1, 0, len(reverse_chunks)) - 1
    if last_end > word_ends[-1]:
        word_ends.append(last_end)

    for i in range(len(word_ends)-1):
        start = word_ends[i] + 4
        end = word_ends[i+1] + 1
        words.append(sound_stream[start:end])

    return words, max_amplitude


if __name__ == '__main__':
    filename = 'recorded_voices/voices/'

    #     filename = 'clipped_voices/' + files[i]
    #     x, Fs = librosa.load(filename, sr=None)
    #     print_plot_play(filename, x, Fs)
    #     record_audio('recorded_voices/silence_chunk.wav')
    # play_audio(filename='recorded_voices/silence_chunk.wav', sound_stream=None)  # play from file
    # play_audio(filename='recorded_voices/voices/phone1.wav', sound_stream=None)  # play from file
    # time.sleep(3)
    # streams = load_audio(filename='recorded_voices/voices/phone1.wav')  # play from bytes
    # frames = []
    # for frame in streams:
    #     data_int = np.array(struct.unpack(str(len(frame)) + 'B', frame), dtype='b')
    #     frames.append(data_int)
    # streams_rec = []
    # for frame in frames:
    #     frame_rec = np.ndarray.tobytes(frame)
    #     streams_rec.append(frame_rec)

    # words, amplitude = detector(streams)
    # for i in range(len(words)):
    #     play_audio(filename=None, sound_stream=words[i])
    # detect_record_audio(filename)

    # clip audio
    # clip_audio_files()

    # detect_record_audio(filename)

    # play all the audios
    files = os.listdir('recorded_voices/speeches/')
    files.sort()
    for i in range(len(files)):
        print(files[i])
        file = 'recorded_voices/speeches/' + files[i]
        play_audio(file)
        a = input('press any key to continue')