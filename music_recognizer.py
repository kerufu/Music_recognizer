import queue
import sys
from operator import itemgetter
import operator
import threading
import hashlib

from scipy.io import wavfile
import matplotlib.mlab as mlab
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, binary_erosion
import pyaudio
from numpy import *

# should be recorded by the same microphone
# example from https://www.youtube.com/watch?v=xQzS3JnZQZM, 0~38s
music_list = [
    "./music_database/1.wav",
    "./music_database/2.wav"
]

WINDOW = 2048  
OVERLAP = 0.5
MIN_MAX = 3
CONNECTIVITY = 1
NEIG_NUMBER = 10
REGION_T = 200
REGION_F = 200
HASH_KEEP = 24

CHUNK_SIZE = 4000
DEVICE_ID = 0
CHANNELS = 1
SAMPLE_WIDTH = pyaudio.paInt16
SAMPLE_RATE = 16000

NUM_CHUNKS_IN_WAV = 4

def process(rate,data):
    spectrum = mlab.specgram(data,NFFT=WINDOW,Fs=rate,window=mlab.window_hanning,noverlap=int(WINDOW * OVERLAP))
    
    spec_data = asarray(spectrum[0])
    
    struct = generate_binary_structure(2, CONNECTIVITY)
    neighborhood = iterate_structure(struct,NEIG_NUMBER)
    local_max = maximum_filter(spec_data, footprint=neighborhood) == spec_data
    background = (spec_data==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background # because previously the eroded background is also true in the peaks;    
    
    the_peaks = spec_data[detected_peaks]
    
    p_row, p_col = where(detected_peaks)
    
    peaks = vstack((p_row,p_col,the_peaks))
    
    real_peaks = peaks[:,peaks[2,:]>MIN_MAX]

    f_index = real_peaks[0,:]
    t_index = real_peaks[1,:]
    
    points = zip(f_index,t_index)
    points = sorted(list(points), key=itemgetter(1))
    points = asarray(points).T
    
    points_leng = points.shape[1]
    feature_points = list()
    for i in range(points_leng):
        for j in range(1,NEIG_NUMBER):
            if (i+j) < points_leng and (points[1, (i+j)]-points[1,i]) < REGION_T and abs((points[0,(i+j)] - points[0,i])) < REGION_F:
                f1 = points[0,i]
                f2 = points[0,(i+j)]
                t = points[1,i]
                t_diff = points[1,(i+j)]-points[1,i]
                
                hass = hashlib.sha1(("%s|%s|%s" % (str(f1), str(f2), str(t_diff))).encode("utf-8"))            
                
                this_hash = [hass.hexdigest()[0:HASH_KEEP],t]
                feature_points.append(this_hash)
                
    return feature_points

class MusicRecognizer:
    def __init__(self):
        self.music_processed = []
        for m in music_list:
            rate, data = wavfile.read(m)
            data = data[:,0]
            self.music_processed.append(asarray(process(rate, data)).T)

    def callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def get_new_wav(self):
        # get newest chunks in bus
        while self.queue.qsize() > NUM_CHUNKS_IN_WAV:
            _ = self.queue.get()
        self.new_sample = b""
        for _ in range(NUM_CHUNKS_IN_WAV):
            self.new_sample += self.queue.get()
        
    def matching(self, music_index):
        mp = self.music_processed[music_index]
        match = {-1: -1}
        for i in range(mp.shape[1]):
            for j in range(self.test_leng):
                if mp[0,i] == self.input_feature_point[0,j]:
                    key = abs(int(float(mp[1,i]) - float(self.input_feature_point[1,j])))
                    if key in match.keys():
                        match[key] += 1
                    else:
                        match[key] = 1
        
        self.matching_result.append((music_index, max(match.values())))

    def run(self):
        self.queue = queue.SimpleQueue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            input=True,
            output=False,
            start=True,
            format=SAMPLE_WIDTH,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=DEVICE_ID,
            stream_callback=self.callback
        )

        self.get_new_wav()
        wav = frombuffer(self.new_sample, dtype=int16)

        while True:

            get_new_wav_thread = threading.Thread(target=self.get_new_wav)
            get_new_wav_thread.start()

            self.input_feature_point=process(SAMPLE_RATE, wav)
            self.input_feature_point=asarray(self.input_feature_point).T
            self.test_leng=self.input_feature_point.shape[1]
            matching_threads = []
            self.matching_result = []
            for i in range(len(self.music_processed)):
                matching_threads.append(threading.Thread(target=self.matching, args=(i,)))
                matching_threads[-1].start()
            for i in range(len(matching_threads)):
                matching_threads[i].join()
            print(self.matching_result)
            get_new_wav_thread.join()
            wav = frombuffer(self.new_sample, dtype=int16)


mr = MusicRecognizer()
mr.run()