import numpy as np
import gradio as gr
import glob
from tortoise import api,utils

voices = [[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/*'))]
qualities = [
	["High Quality", "high_quality"],
	["Standard", "standard"],
	["Fast", "fast"],
	["Ultra Fast", "ultra_fast"],
]


def generate_voice(voice, preset, speaktxt):
	sr = 24000
	srcwavs = glob.glob(voice + "/*.wav")
	reference_clips = [utils.audio.load_audio(p, sr) for p in srcwavs]
	tts = api.TextToSpeech()
	pcm_audio = tts.tts_with_preset(speaktxt, voice_samples=reference_clips, preset=preset)
	return sr, pcm_audio.detach().cpu().numpy()

demo = gr.Interface(
    fn=generate_voice,
    inputs=[
        gr.Dropdown(voices, type="value", label="Source Voice", value=voices[0][1]),
	gr.Radio(qualities, type="value", value=qualities[1][1], label="Generation Quality"),
        gr.Textbox(value="Hello world!", lines=3, label="Text to speak"),
    ],
    outputs="audio",
)
if __name__ == "__main__":
    demo.launch()
