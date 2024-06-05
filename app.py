from flask import Flask, request, render_template, send_from_directory
from scipy.io.wavfile import write
import numpy as np
import torch
from diffusers import AudioLDMPipeline

app = Flask(__name__)

model_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    negative_prompt = "low quality, average quality"
    
    audio = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=10, audio_length_in_s=10, generator=generator).audios[0]
    
    output_file = 'output/music.wav'
    # audio.save(output_file)  # Save the generated audio to a file
    sample_rate = 16000
    write(output_file, sample_rate, np.int16(audio * 32767))

    return render_template('result.html', prompt=prompt, filename=output_file)

@app.route('/output/<path:filename>')
def play(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(debug=True)
