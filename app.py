from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pyaudio
import wave
import whisper
import warnings

app = Flask(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the model on CPU
model = whisper.load_model("small", device="cpu")

def record_audio(duration, filename):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print("Recording...")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording")

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(file_path):
    result = model.transcribe(file_path, language="japanese")
    return result["text"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    # Record audio and save it as 'audio.wav'
    record_audio(5, 'static/audio.wav')
    return redirect(url_for('transcribe'))

@app.route('/transcribe')
def transcribe():
    # Transcribe the recorded audio
    transcription = transcribe_audio('static/audio.wav')
    return render_template('transcription.html', transcription=transcription)

if __name__ == '__main__':
    app.run(debug=True)
