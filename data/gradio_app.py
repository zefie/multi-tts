print("Loading Numpy")
import numpy as np
print("Loading GRadio")
import gradio as gr
print("Loading torch, torchaudio, and librosa...")
import torchaudio, torch, librosa
print("Preloading Coqui...")
from TTS.api import TTS
from TTS.utils import audio
print("Preloading Bark...")
from bark import SAMPLE_RATE, generate_audio, preload_models
print("Preloading Tortoise...")
from tortoise import api,utils
print("Preloading Mars5...")
from mars5.inference import Mars5TTS, InferenceConfig as config_class
print("Loading miscellanous modules...")
import glob, os, argparse, unicodedata, json, random, psutil, requests, re, time, builtins
import scipy.io.wavfile as wav
from string import ascii_letters, digits, punctuation

version = 20240811

device = "cuda" if torch.cuda.is_available() else "cpu"
advanced_opts={}

tts_engines = [
	['Camb.ai Mars5','mars5'],
	['Coqui','coqui'],
	['Suno Bark','bark'],
	['TorToiSe','tortoise']
]


qualities = [
	["High Quality", "high_quality"],
	["Standard", "standard"],
	["Fast", "fast"],
	["Ultra Fast", "ultra_fast"],
]

coqui_voice_models = [
        "tts_models/multilingual/multi-dataset/xtts_v1.1",
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/multilingual/multi-dataset/your_tts",
        "tts_models/en/jenny/jenny",
        "tts_models/en/ek1/tacotron2",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/en/ljspeech/vits"
    ]

bark_voice_models = [item.replace("./bark/assets/prompts/","").replace(".npz","") for item in sorted(glob.glob('./bark/assets/**/*.npz', recursive=True))]

theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="purple",
    neutral_hue="zinc",
)

def getSampleRate(model):
    if model == coqui_voice_models[2]:
        return 16000
    elif model == coqui_voice_models[0] or model == coqui_voice_models[1]:
        return 24000
    elif model == coqui_voice_models[3]:
        return 48000
    else:
        return 22050

def strip_unicode(string):
    """Remove non-ASCII characters from a string."""
    ascii_chars = set(ascii_letters + digits + punctuation) - {' ', '\t', '\n'}
    return ''.join([c for c in string if (ord(c) <= 127 and ord(c) not in (0xc0, 0xc1)) or c in ascii_chars])

def generate_tts(engine, model, voice, speaktxt):
	global advanced_opts
	sr = 22050
	bit_depth = -16
	channels = 1
	audio = np.array([False])
	if engine == "coqui":
		sr = getSampleRate(model)
		speaktxt = strip_unicode(speaktxt)
		os.environ["TTS_HOME"] = "./coqui/"
		tts = TTS().to(device)
		if ',' in model:
			TTS.tts.configs.xtts_config.X
			tts_path = "./coqui/tts/"+model.split(",")[0]+"/"+model.split(",")[1]
			config_path = "./coqui/tts/"+model.split(",")[0]+"/config.json"
			TTS.load_tts_model_by_path(tts, tts_path, config_path)
		else:
			TTS.load_tts_model_by_name(tts, model)

		if tts.is_multi_speaker or tts.is_multi_lingual:
			wavs = glob.glob(voice + "/*.wav")
			if len(wavs) > 0:
				if tts.is_multi_lingual:
					# multilingual, so send language and speaker
					ttsgen = tts.tts(text=speaktxt, speaker_wav=wavs, language="en")
				else:
					# not multilingual, just send speaker
					ttsgen = tts.tts(text=speaktxt, speaker_wav=wavs)
		else:
			# no speaker
			ttsgen = tts.tts(text=speaktxt, sample_rate=sr, channels=channels, bit_depth=bit_depth)
	if engine == "bark":
		sr = SAMPLE_RATE;
		ttsgen = generate_audio(speaktxt, history_prompt="bark/assets/prompts/"+model+".npz")
	if engine == "tortoise":
		sr = 22050
		reference_clips = [utils.audio.load_audio(p, sr) for p in glob.glob(voice + "/*.wav")]
		use_deepspeed = False
		kv_cache = False
		half = False
		if 'use_deepspeed' in advanced_opts:
			use_deepspeed = bool(advanced_opts['use_deepspeed'])
		if 'kv_cache' in advanced_opts:
			kv_cache = bool(advanced_opts['kv_cache'])
		if 'half' in advanced_opts:
			half = bool(advanced_opts['half'])
		tts = api.TextToSpeech(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
		pcm_audio = tts.tts_with_preset(speaktxt, voice_samples=reference_clips, preset=model)
		audio = pcm_audio.detach().cpu().numpy()
	if engine == "mars5":
		mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
		wav, sr = librosa.load(voice, mono=True)
		wav = torch.from_numpy(wav)
		if sr != mars5.sr:
			wav = torchaudio.functional.resample(wav, sr, mars5.sr)
		cfg = config_class(deep_clone=advanced_opts['deep_clone'], rep_penalty_window=advanced_opts['rep_penalty_window'], top_k=advanced_opts['top_k'], top_p=advanced_opts['top_p'], temperature=advanced_opts['temperature'], freq_penalty=advanced_opts['freq_penalty'])
		if len(advanced_opts['transcription']) == 0:
			cfg.deep_clone = False
		ar_codes, output_audio = mars5.tts(speaktxt, wav, advanced_opts['transcription'], cfg=cfg)
		audio = output_audio.detach().cpu().numpy()

	if not audio.any():
		audio = ttsgen / np.max(np.abs(ttsgen))

	return sr, audio

with gr.Blocks(title="zefie's Multi-TTS v"+str(version), theme=theme) as demo:
	def updateVoicesVisibility(tts, model):
		if (tts == 'coqui' and (model == coqui_voice_models[0] or model == coqui_voice_models[1] or model == coqui_voice_models[2])) or tts == 'tortoise' or tts == 'mars5':
			voices = getVoices(tts)
			voice = voices[0][1]
			return gr.Dropdown(choices=voices, visible=True, value=voice)
		else:
			for item in qualities:
				if model == item[1]:
					return gr.Dropdown(choices=qualities, visible=True, value=model)

			return gr.Dropdown(choices=[['','']], visible=False, value="")

	def updateModels(value):
		if value == "bark":
			return gr.Dropdown(choices=bark_voice_models, value=bark_voice_models[0], label="TTS Model")
		if value == "coqui":
			return gr.Dropdown(choices=coqui_voice_models, value=coqui_voice_models[0], label="TTS Model")
		if value == "tortoise":
			return gr.Dropdown(choices=qualities, value=qualities[1][1], label="Quality")
		if value == "mars5":
			return gr.Dropdown(choices=['Camb-ai/mars5-tts'], value='Camb-ai/mars5-tts', label="TTS Model")

	def updateAdvancedVisiblity(value):
		if value == "tortoise":
			updateAdvancedOpts(value, tortoise_opt_comp.value)
			return gr.Group(visible=True), gr.Group(visible=False)
		elif value == "mars5":
			updateAdvancedOpts(value, transcription.value, mars5_bool.value, temperature.value, top_k.value, top_p.value, rep_penalty_window.value, freq_penalty.value, presence_penalty.value, max_prompt_dur.value)
			return gr.Group(visible=False), gr.Group(visible=True)
		else:
			return gr.Group(visible=False), gr.Group(visible=False)

	def getVoices(value):
		if value == "mars5":
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/**/*.wav', recursive=True))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/**/*.wav', recursive=True))])
		elif value == "coqui" or value == "tortoise":
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/*'))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/*'))])
		return wavs

	def updateAdvancedOpts(tts, val1, val2 = None, val3 = None, val4 = None, val5 = None, val6 = None, val7 = None, val8 = None, val9 = None, val10 = None):
		global advanced_opts
		if tts == "tortoise":
			use_deepspeed, kv_cache, half = [False, False, False]
			if 'use_deepspeed' in val1:
				use_deepspeed = True
			if 'kv_cache' in val1:
				kv_cache = True
			if 'half' in val1:
				half = True

			advanced_opts = {
				'use_deepspeed': use_deepspeed,
				'kv_cache': kv_cache,
				'half': half
			}
		elif tts == "mars5":
			advanced_opts = {
				'transcription': val1,
				'deep_clone': ('deep_clone' in val2),
				'use_kv_cache': ('use_kv_cache' in val2),
				'temperature': val3,
				'top_k': val4,
				'top_p': val5,
				'rep_penalty_window': val6,
				'freq_penalty': val7,
				'presence_penalty': val8,
				'max_prompt_dur': val9
			}
		else:
			advanced_opts = {}

	def voiceChanged(tts, voice):
		m5_bool_value = mars5_bool.value.copy()
		out = ''
		if tts == "mars5":
			text = voice.replace(".wav",".txt")
			if os.path.isfile(text):
				f = open(text,"r");
				out = f.read();
				f.close()
				if 'deep_clone' not in mars5_bool.value:
					m5_bool_value.append('deep_clone')
		else:
			if 'deep_clone' in mars5_bool.value:
				m5_bool_value.remove('deep_clone')

		return gr.Textbox(value=out), gr.CheckboxGroup(value=m5_bool_value)

	voices = getVoices("coqui")
	tts_select = gr.Radio(tts_engines, type="value", value="coqui", label="TTS Engine")
	voice = voices[0][1]
	voice_select = gr.Dropdown(choices=voices, value=voice, type="value", visible=True, label="Voice Cloning", info="Place your custom voices in /home/app/srcwav/Desired Name/File1.wav, etc")
	model_select = gr.Dropdown(coqui_voice_models, type="value", value=coqui_voice_models[0], label="TTS Model")
	audioout = gr.Audio(show_download_button=True, label="Generated Audio")
	speak_text = gr.Textbox(value="Welcome to the multi text to speech generator", label="Text to speak", lines=3)
	demo2 = gr.Interface(
		generate_tts,
		[
			tts_select,
			model_select,
			voice_select,
			speak_text,
		],
		audioout,
		title="zefie's Multi-TTS v"+str(version),
		theme="soft",
		allow_flagging="never",
		submit_btn="Generate",
		show_progress='full',
	)
	with gr.Group(visible=False) as tortoise_opts:
		tortoise_opt_comp = gr.CheckboxGroup([["Use Deepspeed","use_deepspeed"],["Use KV Cache","kv_cache"],["fp16 (half)","half"]], label="Tortoise Advanced Options", value=['use_deepspeed','kv_cache'])

	with gr.Group(visible=False) as mars5_opts:
		mars5_bool = gr.CheckboxGroup([["Deep Clone (requires transcription)","deep_clone"],["Use KV Cache","use_kv_cache"]],label="Camb.ai Mars5 Advanced Options", value=['use_kv_cache'])
		transcription = gr.Textbox("", lines=4, placeholder="Type your transcription here, or provide a .txt file of the same name next to the .wav", label="Voice Cloning Transcription (Optional, but recommended)", info="You can place a .txt of the same name next to a .wav to autoload its transcription.")
		temperature = gr.Slider(value=0.7, minimum=0, maximum=3, label="Temperature", info="high temperatures (T>1) favour less probable outputs while low temperatures reduce randomness")
		with gr.Row():
			top_k = gr.Slider(value=200,minimum=0,maximum=1000, label="top_k", info="used for sampling, keeps tokens with the highest probabilities until a certain number (top_k) is reached")
			rep_penalty_window = gr.Slider(value=80,minimum=0,maximum=200,label="rep_penalty_window",info="how far in the past to consider when penalizing repetitions. default equates to 5s")
		with gr.Row():
			top_p = gr.Slider(value=0.2,minimum=0,maximum=1, label="top_p",info="used for sampling, keep the top tokens with cumulative probability >= top_p")
			freq_penalty = gr.Slider(value=3,minimum=0,maximum=100,label="freq_penalty",info="increasing it would penalize the model more for reptitions")
		with gr.Row():
			max_prompt_dur = gr.Slider(value=12,minimum=1,maximum=30,label="max_prompt_dur",info="maximum length prompt is allowed, in seconds")
			presence_penalty = gr.Slider(value=0.4,minimum=0,maximum=1,label="presence_penalty",info="increasing it would increase token diversity")

	tts_select.change(updateModels,tts_select,model_select)
	tts_select.change(updateAdvancedVisiblity,tts_select,[tortoise_opts,mars5_opts])
	transcription.blur(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	mars5_bool.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	temperature.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	top_k.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	top_p.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	rep_penalty_window.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	freq_penalty.change(updateAdvancedOpts,[tts_select,transcription,mars5_bool,temperature,top_k,top_p,rep_penalty_window,freq_penalty,presence_penalty,max_prompt_dur])
	tortoise_opt_comp.change(updateAdvancedOpts,[tts_select,tortoise_opt_comp])
	voice_select.change(voiceChanged, [tts_select, voice_select], [transcription, mars5_bool])
	model_select.change(voiceChanged, [tts_select, voice_select], [transcription, mars5_bool])
	model_select.change(updateVoicesVisibility,[tts_select,model_select],voice_select)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

