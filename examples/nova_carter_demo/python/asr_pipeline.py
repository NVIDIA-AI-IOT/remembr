# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import numpy as np
import time
import pyaudio
import socketio
# import eventlet
import asyncio
import uvicorn
import starlette
# from aiohttp import web
import socketio
import threading
# from multiprocessing import Process, Queue, Event
from threading import Thread, Event
from queue import Queue
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from whisper_trt.vad import load_vad
from whisper_trt import load_trt_model, set_cache_dir


def find_respeaker_audio_device_index():

    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    for i in range(num_devices):

        device_info = p.get_device_info_by_host_api_device_index(0, i)
        
        if "respeaker" in device_info.get("name").lower():

            device_index = i

    return device_index


@contextmanager
def get_respeaker_audio_stream(
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 6,
        bitwidth: int = 2
    ):

    if device_index is None:
        device_index = find_respeaker_audio_device_index()

    if device_index is None:
        raise RuntimeError("Could not find Respeaker device.")
    
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=sample_rate,
        format=p.get_format_from_width(bitwidth),
        channels=channels,
        input=True,
        input_device_index=device_index
    )

    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def audio_numpy_from_bytes(audio_bytes: bytes):
    audio = np.fromstring(audio_bytes, dtype=np.int16)
    return audio


def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, 
                      num_channels: int = 6):
    return audio_numpy[channel_index::num_channels]


def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768


@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None


@dataclass
class AudioSegment:
    chunks: AudioChunk


class Microphone(Thread):

    def __init__(self, 
                 output_queue: Queue, 
                 chunk_size: int = 1536, 
                 device_index: int | None = None,
                 use_channel: int = 0, 
                 num_channels: int = 6,
                 sample_rate: int = 16000,
                 bitwidth: int = 2):
        super().__init__()
        self.output_queue = output_queue
        self.chunk_size = chunk_size
        self.use_channel = use_channel
        self.num_channels = num_channels
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.bitwidth = bitwidth

    def run(self):
        with get_respeaker_audio_stream(sample_rate=self.sample_rate, 
                                        device_index=self.device_index, 
                                        channels=self.num_channels, bitwidth=self.bitwidth) as stream:
            while True:
                audio_raw = stream.read(self.chunk_size)
                audio_numpy = audio_numpy_from_bytes(audio_raw)
                audio_numpy = np.stack([audio_numpy_slice_channel(audio_numpy, i, self.num_channels) for i in range(self.num_channels)])
                audio_numpy_normalized = audio_numpy_normalize(audio_numpy)

                audio = AudioChunk(
                    audio_raw=audio_raw,
                    audio_numpy=audio_numpy,
                    audio_numpy_normalized=audio_numpy_normalized
                )

                self.output_queue.put(audio)


class VAD(Thread):

    def __init__(self,
            input_queue: Queue, 
            output_queue: Queue,
            sample_rate: int = 16000,
            use_channel: int = 0,
            speech_threshold: float = 0.5,
            max_filter_window: int = 1,
            ready_flag = None,
            speech_start_flag = None,
            speech_end_flag = None,
            vad_start_callback = None,
            vad_end_callback = None
            ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.use_channel = use_channel
        self.speech_threshold = speech_threshold
        self.max_filter_window = max_filter_window
        self.ready_flag = ready_flag
        self.speech_start_flag = speech_start_flag
        self.speech_end_flag = speech_end_flag
        self.vad_start_callback = vad_start_callback
        self.vad_end_callback = vad_end_callback

    def run(self):

        vad = load_vad()
        
        # warmup run
        vad(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)
        

        max_filter_window = deque(maxlen=self.max_filter_window)

        speech_chunks = []

        prev_is_voice = False

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            

            audio_chunk = self.input_queue.get()

            voice_prob = float(vad(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0])

            chunk = AudioChunk(
                audio_raw=audio_chunk.audio_raw,
                audio_numpy=audio_chunk.audio_numpy,
                audio_numpy_normalized=audio_chunk.audio_numpy_normalized,
                voice_prob=voice_prob
            )

            max_filter_window.append(chunk)

            is_voice = any(c.voice_prob > self.speech_threshold for c in max_filter_window)
            
            if is_voice > prev_is_voice:
                speech_chunks = [chunk for chunk in max_filter_window]
                # start voice
                speech_chunks.append(chunk)

                if self.vad_start_callback:
                    self.vad_start_callback()

            elif is_voice < prev_is_voice:
                # end voice
                segment = AudioSegment(chunks=speech_chunks)
                self.output_queue.put(segment)
                
                if self.vad_end_callback:
                    self.vad_end_callback()

            elif is_voice:
                # continue voice
                speech_chunks.append(chunk)

            prev_is_voice = is_voice


class ASR(Thread):

    def __init__(self, 
            model: str, 
            backend: str, 
            input_queue, 
            use_channel: int = 0, 
            ready_flag = None, 
            model_path: str = None, 
            asr_callback = None
        ):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.backend = backend
        self.model_path = model_path
        self.asr_callback = asr_callback

    def run(self):
        
        if self.backend == "whisper_trt":
            from whisper_trt import load_trt_model
            model = load_trt_model(self.model, path=self.model_path)
        elif self.backend == "whisper":
            from whisper import load_model
            model = load_model(self.model)
        elif self.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            class FasterWhisperWrapper:
                def __init__(self, model):
                    self.model = model
                def transcribe(self, audio):
                    segs, info = self.model.transcribe(audio)
                    text = "".join([seg.text for seg in segs])
                    return {"text": text}
                
            model = FasterWhisperWrapper(WhisperModel(self.model))

        # warmup
        model.transcribe(np.zeros(1536, dtype=np.float32))

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:

            speech_segment = self.input_queue.get()

            audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])

            text = model.transcribe(audio)['text']


            if self.asr_callback is not None:
                self.asr_callback(text)


class ASRPipeline:

    def __init__(self,
            model: str = "small.en",
            vad_window: int = 5,
            backend: str = "whisper_trt",
            cache_dir: Optional[str] = None,
            vad_start_callback = None,
            vad_end_callback = None,
            asr_callback = None,
            mic_device_index: Optional[int] = None,
            mic_channel_for_asr: int = 0,
            mic_num_channels: int = 6,
            mic_sample_rate: int = 16000,
            mic_bitwidth: int = 2
        ):
        
        self.audio_chunks = Queue()
        self.speech_segments = Queue()
        self.speech_outputs = Queue()

        self.vad_ready = Event()
        self.asr_ready = Event()

        if cache_dir is not None:
            set_cache_dir(cache_dir)

        self.mic = Microphone(
            self.audio_chunks,
            device_index=mic_device_index,
            use_channel=mic_channel_for_asr,
            num_channels=mic_num_channels,
            sample_rate=mic_sample_rate,
            bitwidth=mic_bitwidth
        )

        self.vad = VAD(
            self.audio_chunks, 
            self.speech_segments, 
            max_filter_window=vad_window, 
            ready_flag=self.vad_ready, 
            vad_start_callback=vad_start_callback,
            vad_end_callback=vad_end_callback
        )

        self.asr = ASR(
            model, 
            backend, 
            self.speech_segments, 
            ready_flag=self.asr_ready,
            asr_callback=asr_callback
        )

    def start(self):

        self.vad.start()
        self.asr.start()
        
        self.vad_ready.wait()
        self.asr_ready.wait()

        self.mic.start()

    def join(self):

        self.mic.join()
        self.vad.join()
        self.asr.join()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="small.en")
    parser.add_argument("--backend", type=str, default="whisper_trt")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--vad_window", type=int, default=5)
    args = parser.parse_args()

    def handle_vad_start():
        print("vad start")

    def handle_vad_end():
        print("vad end")

    def handle_asr(text):
        print("asr done: " + text)

    pipeline = ASRPipeline(
        model=args.model,
        backend=args.backend,
        cache_dir=args.cache_dir,
        vad_window=args.vad_window,
        vad_start_callback=handle_vad_start,
        vad_end_callback=handle_vad_end,
        asr_callback=handle_asr
    )

    pipeline.start()
    pipeline.join()