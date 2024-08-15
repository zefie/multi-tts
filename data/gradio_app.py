print("Loading Numpy")
import numpy as np
print("Loading GRadio")
import gradio as gr
print("Loading torch, torchaudio, and librosa...")
import torch, torchaudio, librosa
print("Loading miscellanous modules...")
from TTS.utils.manage import ModelManager
import glob, os, argparse, unicodedata, json, random, psutil, requests, re, time, builtins, sys, argparse, gc, io
import scipy.io.wavfile as wav
from string import ascii_letters, digits, punctuation

loaded_tts = { 'voice': None }

version = 20240814

paths = [
	'/root/.cache/coqui',
	'/root/.cache/coqui/tts',
	'/root/.cache/coqui/vocoder',
	'/root/.cache/coqui/speaker_encoder',
]

for dir in paths:
	if not os.path.exists(dir):
		os.mkdir(dir)

# Get device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class Logger:
	logdata = ""
	def __init__(self, terminal):
		self.terminal = terminal

	def read(self):
		return self.logdata

	def write(self, message):
		self.terminal.write(message)
		self.logdata += message

	def flush(self):
		self.terminal.flush()

	def clear(self):
		self.logdata = ""

	def terminal(self):
		return self.terminal

	def isatty(self):
		return False


sys.stdin = io.StringIO(f"n\n")

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    newline="",
    line_buffering=True,
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace",
    newline="",
    line_buffering=True,
)

sys.stdout = Logger(sys.stdout)

# save stdout to a var for restoring with unmute()
stdout = sys.stdout
stderr = sys.stderr

def mute(full = False):
	# Mutes any stdout text
	f = open(os.devnull, 'w')
	sys.stdout = f
	if full:
		# also mute stderr
		sys.stderr = f

def unmute():
    # Unmutes stdout and stderr
    sys.stdout = stdout
    sys.stderr = stderr

advanced_opts={}

tts_engines = [
	['Camb.ai Mars5','mars5'],
	['Coqui','coqui'],
	['Parler', 'parler'],
	['Suno Bark','bark'],
	['TorToiSe','tortoise']
]


# Instead of models, offer presets for TorToiSe
tortoise_presets = {
	'ultra_fast': {'label': 'Ultra Fast', 'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
	'fast': {'label': 'Fast', 'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
	'standard': {'label': 'Standard', 'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
	'high_quality': {'label': 'High Quality', 'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
	'zefie_hq': {'label': "zefie's High Quality", 'num_autoregressive_samples': 512, 'diffusion_iterations': 400},
}

# Query Coqui for it's TTS model list, mute to prevent dumping the list to console
mute()
manager = ModelManager()
coqui_voice_models = []
for model in manager.list_models():
	if model[:10] == "tts_models":
		coqui_voice_models.append(model)
unmute()

css_style = """
.bark_console {
	font: 1.3rem Inconsolata, monospace;
	white-space: pre;
	padding: 5px;
	border: 2px dashed purple;
	border-radius: 3px;
	max-height: 500px;
	overflow-y: scroll;
	font-size: 90%;
	overflow-x: hidden;
}
"""

# This seems broken, and we already have Bark and Tortoise
coqui_voice_models.remove('tts_models/multilingual/multi-dataset/bark')
coqui_voice_models.remove('tts_models/en/multi-dataset/tortoise-v2')

# Scan bark /home/app/bark/assets/* for .npz files. You could add your own to /home/app/bark/custom/yourvoice.npz
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

def unload_engines(keep):
	global loaded_tts
	keys = []
	for engine in loaded_tts.keys():
		keys.append(engine)
	for engine in keys:
		if keep != engine:
			print('Unloading '+engine+'...')
			del loaded_tts[engine]
	gc.collect()
	if device == 'cuda':
		torch.cuda.empty_cache()

def optionSelected(option, options):
	for opt in options:
		if option in opt:
			return True
	return False


def generate_tts(engine, model, voice, speaktxt, progress=gr.Progress()):
	global advanced_opts, loaded_tts
	print("Advanced Options:", advanced_opts)
	sr = 22050
	bit_depth = -16
	channels = 1
	audio = np.array([False])
	progress(0, "Preparing...")
	print("Preparing...")
	if engine == "coqui":
		sr = getSampleRate(model)
		sys.stdin.seek(0)
		if sys.stdin.read() != f"y\n" and 'xtts' in model:
			raise gr.Error("Please accept the Coqui Public Model License (CPML) before using XTTS models!")
			return sr, None
		sys.stdin.seek(0)
		speaktxt = strip_unicode(speaktxt)
		os.environ["TTS_HOME"] = "./coqui/"
		if engine not in loaded_tts:
			progress(0.15, "Loading Coqui...")
			print("Loading Coqui...")
			from TTS.api import TTS as tts_api
			unload_engines(engine)
			loaded_tts[engine] = {}
			loaded_tts[engine]['api'] = tts_api
			loaded_tts[engine]['engine'] = loaded_tts[engine]['api']().to(device)
			loaded_tts[engine]['model'] = None
			del tts_api
		progress(0.25,"Loaded Coqui")
		print("Loaded Coqui")

		if loaded_tts[engine]['model'] != model:
			progress(0.30,"Loading model...")
			print("Loading model...")
			if ',' in model:
				loaded_tts[engine]['api'].tts.configs.xtts_config.X
				tts_path = "./coqui/tts/"+model.split(",")[0]+"/"+model.split(",")[1]
				config_path = "./coqui/tts/"+model.split(",")[0]+"/config.json"
				loaded_tts[engine]['api'].load_tts_model_by_path(loaded_tts[engine]['engine'], tts_path, config_path)
			else:
				loaded_tts[engine]['api'].load_tts_model_by_name(loaded_tts[engine]['engine'], model)
			loaded_tts[engine]['model'] = model

		progress(0.50,"Generating...")
		print("Generating...")
		if loaded_tts[engine]['engine'].is_multi_speaker or loaded_tts[engine]['engine'].is_multi_lingual:
			wavs = glob.glob(voice + "/*.wav")
			if len(wavs) > 0:
				if loaded_tts[engine]['engine'].is_multi_lingual:
					# multilingual, so send language and speaker
					ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, speaker_wav=wavs, language="en")
				else:
					# not multilingual, just send speaker
					ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, speaker_wav=wavs)
		else:
			# no speaker
			ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, sample_rate=sr, channels=channels, bit_depth=bit_depth)
	if engine == "bark":
		if engine not in loaded_tts:
			progress(0.15, "Loading Bark...")
			print("Loading Bark...")
			import bark
			unload_engines(engine)
			loaded_tts[engine] = bark
		progress(0.25, "Loaded Bark")
		print("Loaded Bark")
		sr = loaded_tts[engine].SAMPLE_RATE;
		progress(0.50, "Generating...")
		print("Generating...")
		ttsgen = loaded_tts[engine].generate_audio(speaktxt, history_prompt="bark/assets/prompts/"+model+".npz")
	if engine == "tortoise":
		sr = 24000
		if engine not in loaded_tts:
			progress(0.15,"Loading TorToiSe...")
			print("Loading TorToiSe...")
			unload_engines(engine)
			from tortoise import utils, api
			loaded_tts[engine] = {}
			loaded_tts[engine]['api'] = api
			loaded_tts[engine]['utils'] = utils
			del api, utils
		progress(0.25, "Loaded TorToiSe")
		print("Loaded TorToiSe")
		loaded_tts[engine]['model'] = None
		progress(0.50, "Generating...")
		print("Generating...")
		tts = loaded_tts[engine]['api'].TextToSpeech(use_deepspeed=advanced_opts['use_deepspeed'], kv_cache=advanced_opts['kv_cache'], half=advanced_opts['half'], device=device)
		pcm_audio = tts.tts(speaktxt, voice_samples=reference_clips, temperature=advanced_opts['temperature'], num_autoregressive_samples=advanced_opts['num_autoregressive_samples'], diffusion_iterations=advanced_opts['diffusion_iterations'], cond_free=advanced_opts['cond_free'], diffusion_temperature=advanced_opts['diffusion_temperature'])
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = pcm_audio.detach().cpu().numpy()
	if engine == "mars5":
		if engine not in loaded_tts:
			progress(0.15, "Loading Camb.ai Mars5...")
			print("Loading Camb.ai Mars5...")
			unload_engines(engine)
			from mars5.inference import Mars5TTS, InferenceConfig as config_class
			loaded_tts[engine] = {}
			loaded_tts[engine]['Mars5TTS'] = Mars5TTS
			loaded_tts[engine]['config_class'] = config_class
			loaded_tts[engine]['model'] = None
			del Mars5TTS, config_class
		progress(0.25, "Loaded Camb.ai Mars5")
		print("Loaded Camb.ai Mars5")

		if loaded_tts[engine]['model'] != model:
			progress(0.30, "Loading model...")
			print("Loading model...")
			loaded_tts[engine]['api'], loaded_tts[engine]['config_class'] = torch.hub.load(model, 'mars5_english', trust_repo=True)
			loaded_tts[engine]['model'] = model

		progress(0.30, "Loading voice...")
		print("Loading voice...")
		wav, sr = librosa.load(voice, mono=True)
		wav = torch.from_numpy(wav)
		if sr != loaded_tts[engine]['api'].sr:
			print("Resampling voice...")
			wav = torchaudio.functional.resample(wav, sr, loaded_tts[engine]['api'].sr)
		sr = loaded_tts[engine]['api'].sr
		cfg = loaded_tts[engine]['config_class'](
			deep_clone=advanced_opts['deep_clone'],
			use_kv_cache=advanced_opts['use_kv_cache'],
			temperature=advanced_opts['temperature'],
			top_k=advanced_opts['top_k'],
			top_p=advanced_opts['top_p'],
			rep_penalty_window=advanced_opts['rep_penalty_window'],
			freq_penalty=advanced_opts['freq_penalty'],
			presence_penalty=advanced_opts['presence_penalty'],
			max_prompt_dur=advanced_opts['max_prompt_dur'])
		if len(advanced_opts['transcription']) == 0:
			cfg.deep_clone = False
			if 'deep_clone' in mars5_bool.value:
				mars5_bool.value.remove('deep_clone')
		progress(0.50, "Generating...")
		print("Generating...")
		ar_codes, output_audio = loaded_tts[engine]['api'].tts(speaktxt, wav, advanced_opts['transcription'], cfg=cfg)
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = output_audio.detach().cpu().numpy()
	if engine == "parler":
		if engine not in loaded_tts:
			progress(0.15, "Loading Parler...")
			print("Loading Parler...")
			unload_engines(engine)
			from parler_tts import ParlerTTSForConditionalGeneration
			from transformers import AutoTokenizer
			loaded_tts[engine] = {}
			loaded_tts[engine]['ParlerTTSForConditionalGeneration'] = ParlerTTSForConditionalGeneration
			loaded_tts[engine]['AutoTokenizer'] = AutoTokenizer
			loaded_tts[engine]['model'] = None
			del ParlerTTSForConditionalGeneration, AutoTokenizer
		progress(0.25, "Loaded Parler")
		print("Loaded Parler")
		if loaded_tts[engine]['model'] != model:
			progress(0.30, "Loading model...")
			print("Loading model...")
			loaded_tts[engine]['api'] = loaded_tts[engine]['ParlerTTSForConditionalGeneration'].from_pretrained(model, attn_implementation=advanced_opts['attn_implementation']).to(device)
			loaded_tts[engine]['model'] = model
		sr = loaded_tts[engine]['api'].config.sampling_rate
		if advanced_opts['compile_mode']:
			progress(0.40, "Compiling Deepspeed...")
			print("Compiling Deepspeed...")
			loaded_tts[engine]['api'].generation_config.cache_implementation = "static"
			loaded_tts[engine]['api'].forward = torch.compile(pmodel.forward, mode=advanced_opts['compile_mode'])
			progress(0.45, "Compliation Done.")
			print("Compliation Done.")
		tokenizer = loaded_tts[engine]['AutoTokenizer'].from_pretrained(model)
		description = tokenizer(advanced_opts['description'], return_tensors="pt").to(device)
		prompt = tokenizer(speaktxt, return_tensors="pt").to(device)
		if advanced_opts['inc_attn_mask']:
			model_kwargs = {"input_ids": description.input_ids, "prompt_input_ids": prompt.input_ids, "attention_mask": description.attention_mask, "prompt_attention_mask": prompt.attention_mask}
		else:
			model_kwargs = {"input_ids": description.input_ids, "prompt_input_ids": prompt.input_ids}
		progress(0.50, "Generating...")
		print("Generating...")
		generation = loaded_tts[engine]['api'].generate(**model_kwargs)
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = generation.cpu().numpy().squeeze()

	if not audio.any():
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = ttsgen / np.max(np.abs(ttsgen))

	progress(100, "Done.")
	print("Done.")
	return sr, audio

def updateVoicesVisibility(engine, model, current_voice):
	if (engine == 'coqui' and 'multilingual' in model) or engine == 'tortoise' or engine == 'mars5':
		voices = getVoices(engine)
		if current_voice != '':
			loaded_tts['voice'] = current_voice

		if  voice_select.choices == voices:
			return gr.Dropdown(visible=True) # no update, just vis
		elif optionSelected(loaded_tts['voice'],voices):
			return gr.Dropdown(choices=voices, visible=True, value=loaded_tts['voice']) # update and restore selected
		else:
			return gr.Dropdown(choices=voices, visible=True, value=voices[0][1]) # update and default selected (#1)
	return gr.Dropdown(choices=[['','']], visible=False, value="") # Empty and hidden

def updateModels(value):
	if value == "bark":
		return gr.Dropdown(choices=bark_voice_models, value=bark_voice_models[0], label="TTS Model")
	if value == "coqui":
		return gr.Dropdown(choices=coqui_voice_models, value=coqui_voice_models[0], label="TTS Model")
	if value == "tortoise":
		tortoise_pr = [[tortoise_presets[item]['label'],item] for item in tortoise_presets.keys()]
		return gr.Dropdown(choices=tortoise_pr, value='standard', label="Preset")
	if value == "mars5":
		return gr.Dropdown(choices=['Camb-ai/mars5-tts'], value='Camb-ai/mars5-tts', label="TTS Model")
	if value == "parler":
		return gr.Dropdown(choices=['parler-tts/parler-tts-mini-v1', 'parler-tts/parler-tts-large-v1'], value='parler-tts/parler-tts-large-v1', label="TTS Model")

def updateOpts(engine):
	if engine == "tortoise":
		updateAdvancedOpts(engine, tortoise_opts_comp.value, tortoise_temperature.value, tortoise_diffusion_temperature.value, tortoise_num_autoregressive_samples.value, tortoise_diffusion_iterations.value)
	elif engine == "mars5":
		updateAdvancedOpts(engine, mars5_transcription.value, mars5_bool.value, mars5_temperature.value, mars5_top_k.value, mars5_top_p.value, mars5_rep_penalty_window.value, mars5_freq_penalty.value, mars5_presence_penalty.value, mars5_max_prompt_dur.value)
	elif engine == "parler":
		updateAdvancedOpts(engine, parler_options.value, parler_description.value, parler_attn_implementation.value, parler_temperature.value)
	else:
		updateAdvancedOpts(engine)

def updateAdvancedVisiblity(engine):
	if engine == "coqui":
		return {
			coqui_opts: gr.Group(visible=True),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	if engine == "tortoise":
		return {
			coqui_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=True),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "mars5":
		return {
			coqui_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=True),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "parler":
		return {
			coqui_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=True)
		}
	else:
		return {
			coqui_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	updateOpts(engine)

def updateAdvancedOpts(engine, *args):
	# wtf...
	global advanced_opts
	if engine == "tortoise":
		use_deepspeed, kv_cache, half, cond_free = [False, False, False, False]
		if 'use_deepspeed' in args[0]:
			use_deepspeed = True
		if 'kv_cache' in args[0]:
			kv_cache = True
		if 'half' in args[0]:
			half = True
		if 'cond_free' in args[0]:
			cond_free = True

		advanced_opts = {
			'use_deepspeed': use_deepspeed,
			'kv_cache': kv_cache,
			'half': half,
			'cond_free': cond_free,
			'temperature': args[1],
			'diffusion_temperature': args[2],
			'num_autoregressive_samples': args[3],
			'diffusion_iterations': args[4]
		}
	elif engine == "mars5":
		advanced_opts = {
			'transcription': args[0],
			'deep_clone': ('deep_clone' in args[1]),
			'use_kv_cache': ('use_kv_cache' in args[1]),
			'temperature': args[2],
			'top_k': args[3],
			'top_p': args[4],
			'rep_penalty_window': args[5],
			'freq_penalty': args[6],
			'presence_penalty': args[7],
			'max_prompt_dur': args[8]
		}
	elif engine == "parler":
		compile_mode, inc_attn_mask = [False,False]
		if 'compile_mode' in args[0]:
			compile_mode = True
		if 'inc_attn_mask' in args[0]:
			inc_attn_mask = True
			advanced_opts = {
			'description': args[1],
			'attn_implementation': args[2],
			'compile_mode': 'default' if compile_mode else False,
			'inc_attn_mask': inc_attn_mask,
			'temperature': args[3]
		}
	else:
		advanced_opts = {}

def voiceChanged(engine, voice):
	global loaded_tts
	# Read .txt file if it exists and toggle Deep Clone accordingly
	m5_bool_value = mars5_bool.value.copy()
	out = ''
	if voice:
		loaded_tts['voice'] = voice
	if engine == "mars5":
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
		return {
			mars5_transcription: gr.Textbox(value=out),
			mars5_bool: gr.CheckboxGroup(value=m5_bool_value),
		}
	return {
		mars5_transcription: gr.Textbox(),
		mars5_bool: gr.CheckboxGroup(),
	}

def presetChanged(engine, model):
	tortoise_opts_value = tortoise_opts_comp.value.copy()
	if engine == "tortoise":
		if 'cond_free' not in tortoise_presets[model]:
			# True
			if 'cond_free' not in tortoise_opts_value:
				tortoise_opts_value.append('cond_free')
		else:
			# False
			if 'cond_free' in tortoise_opts_value:
				tortoise_opts_value.remove('cond_free')
			return {
			tortoise_num_autoregressive_samples: gr.Slider(value=tortoise_presets[model]['num_autoregressive_samples']),
			tortoise_diffusion_iterations: gr.Slider(value=tortoise_presets[model]['diffusion_iterations']),
			tortoise_opts_comp: gr.CheckboxGroup(value=tortoise_opts_value),
		}
	return {
		tortoise_num_autoregressive_samples: gr.Slider(),
		tortoise_diffusion_iterations: gr.Slider(),
		tortoise_opts_comp: gr.CheckboxGroup(),
	}

with gr.Blocks(title="zefie's Multi-TTS v"+str(version), theme=theme, css=css_style) as demo:
	def getVoices(engine):
		if engine == "mars5":
			# Scan samples and srcwavs, and return each wav individually
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/**/*.wav', recursive=True))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/**/*.wav', recursive=True))])
		elif engine == "coqui" or engine == "tortoise":
			# Scan samples and srcwavs, but return the folder, not each wav
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/*'))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/*'))])
			wavs.extend([[item.replace('./tortoise-tts/tortoise/voices','tortoise'),item] for item in sorted(glob.glob('./tortoise-tts/tortoise/voices/*'))])
		return wavs

	def read_log():
		sys.stdout.flush()
		return sys.stdout.read(), gr.Button();

	def toggleAcceptance(value):
		sys.stdin.seek(0)
		if value:
			sys.stdin.write(f"y\n")
		else:
			sys.stdin.write(f"n\n")

	def clear_log():
		sys.stdout.clear()

	voices = getVoices("coqui")
	voice = voices[0][1]
	with gr.Tab("TTS"):
		with gr.Group() as main_group:
			with gr.Row():
				gr.Markdown("# <p style=\"text-align: center;\">zefie's Multi-TTS v"+str(version)+"</p>")
			with gr.Row():
				with gr.Column():
					tts_select = gr.Radio(tts_engines, type="value", value="coqui", label="TTS Engine")
					model_select = gr.Dropdown(coqui_voice_models, type="value", value=coqui_voice_models[0], label="TTS Model")
					voice_select = gr.Dropdown(choices=voices, value=voice, type="value", visible=True, label="Voice Cloning", info="Place your custom voices in /home/app/srcwav/Desired Name/File1.wav, etc")
					speak_text = gr.Textbox(value="Welcome to the multi text to speech generator", label="Text to speak", lines=3)
				with gr.Column():
					audioout = gr.Audio(show_download_button=True, label="Generated Audio", interactive=False, scale=4)
			with gr.Row():
				submit_btn = gr.Button("Submit", variant="primary")

		with gr.Group() as coqui_opts:
			with gr.Row():
				gr.HTML("<p style=\"padding-left: 10px\">Coqui Advanced Options - Read the <a href='https://coqui.ai/cpml' target='_blank'>Coqui Public Model License (CPML)</a></p>")
			with gr.Row():
				xtts_licence = gr.Checkbox(label="I agree to the CPML", value=False, info="Must be checked before using XTTS models.")
		with gr.Group(visible=False) as tortoise_opts:
			with gr.Row():
				gr.Markdown("<p style=\"padding-left: 10px\">Tortoise Advanced Options</p>")
			with gr.Row():
				tortoise_opts_comp = gr.CheckboxGroup([["Use Deepspeed","use_deepspeed"],["Use KV Cache","kv_cache"],['Conditioning-Free Diffusion','cond_free'],["fp16 (half)","half"]], value=['use_deepspeed','kv_cache','cond_free'])
			with gr.Row():
				tortoise_temperature = gr.Slider(value=0.8, minimum=0, maximum=3, label="Temperature", info="The softmax temperature of the autoregressive model.")
				tortoise_diffusion_temperature = gr.Slider(value=1, minimum=0, maximum=1, label="Difussion Temperature", info="Controls the variance of the noise fed into the diffusion model. Values at 0 are the \"mean\" prediction of the diffusion network and will sound bland and smeared")
			with gr.Row():
				tortoise_num_autoregressive_samples = gr.Slider(value=256, minimum=16, maximum=2048, label="# of Autoregressive Samples", info="Number of samples taken from the autoregressive model, all of which are filtered using CLVP. As Tortoise is a probabilistic model, more samples means a higher probability of creating something \"great\".")
				tortoise_diffusion_iterations = gr.Slider(value=200, minimum=30, maximum=4000, label="Diffusion Iterations", info="Number of diffusion steps to perform. More steps means the network has more chances to iteratively refine the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better, however.")
		with gr.Group(visible=False) as mars5_opts:
			with gr.Row():
				gr.Markdown("<p style=\"padding-left: 10px\">Camb.ai Mars5 Advanced Options</p>")
			with gr.Row():
				mars5_transcription = gr.Textbox("", lines=4, placeholder="Type your transcription here, or provide a .txt file of the same name next to the .wav", label="Voice Cloning Transcription 	(Optional, but recommended)", info="You can place a .txt of the same name next to a .wav to autoload its transcription.")
			with gr.Row():
				mars5_bool = gr.CheckboxGroup([["Deep Clone (requires transcription)","deep_clone"],["Use KV Cache","use_kv_cache"]], value=['use_kv_cache'])
				mars5_temperature = gr.Slider(value=0.7, minimum=0, maximum=3, label="Temperature", info="high temperatures (T>1) favour less probable outputs while low temperatures reduce randomness")
			with gr.Row():
				mars5_top_k = gr.Slider(value=200,minimum=0,maximum=1000, label="top_k", info="used for sampling, keeps tokens with the highest probabilities until a certain number (top_k) is reached")
				mars5_rep_penalty_window = gr.Slider(value=80,minimum=0,maximum=200,label="rep_penalty_window",info="how far in the past to consider when penalizing repetitions. default equates to 5s")
			with gr.Row():
				mars5_top_p = gr.Slider(value=0.2,minimum=0,maximum=1, label="top_p",info="used for sampling, keep the top tokens with cumulative probability >= top_p")
				mars5_freq_penalty = gr.Slider(value=3,minimum=0,maximum=100,label="freq_penalty",info="increasing it would penalize the model more for reptitions")
			with gr.Row():
				mars5_max_prompt_dur = gr.Slider(value=12,minimum=1,maximum=30,label="max_prompt_dur",info="maximum length prompt is allowed, in seconds")
				mars5_presence_penalty = gr.Slider(value=0.4,minimum=0,maximum=1,label="presence_penalty",info="increasing it would increase token diversity")
		with gr.Group(visible=False) as parler_opts:
			with gr.Row():
				gr.Markdown("<p style=\"padding-left: 10px\">Parler Advanced Options</p>")
			with gr.Row():
				parler_description = gr.Textbox("A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", lines=3, placeholder="Type your description here, it should describe how you would like the voice to sound",label="Description",info="Describe how you would like the voice to sound.")
			with gr.Row():
				parler_temperature = gr.Slider(value=1, minimum=0, maximum=3, label="Temperature", info="high temperatures (T>1) favour less probable outputs while low temperatures reduce randomness")
			with gr.Row():
				parler_options = gr.CheckboxGroup([['Compile Mode','compile_mode'],['Include Attn Mask','inc_attn_mask']])
				parler_attn_implementation = gr.Dropdown(['eager','sdpa'],value="eager",label="Attention Implementation")
	with gr.Tab("Logs"):
		with gr.Row():
			console_logs = gr.HTML(elem_classes="bark_console")
		with gr.Row():
			clear_button = gr.Button("Clear Log")
			clear_button.click(clear_log)

	groups_group = {'fn': updateAdvancedVisiblity, 'inputs': tts_select, "outputs": [coqui_opts, tortoise_opts, mars5_opts, parler_opts]}
	voices_group = {'fn': updateVoicesVisibility, 'inputs': [tts_select, model_select, voice_select], 'outputs': voice_select}
	tortoise_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, tortoise_opts_comp, tortoise_temperature, tortoise_diffusion_temperature, tortoise_num_autoregressive_samples, tortoise_diffusion_iterations]}
	mars5_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, mars5_transcription, mars5_bool, mars5_temperature, mars5_top_k, mars5_top_p, mars5_rep_penalty_window, mars5_freq_penalty, mars5_presence_penalty, mars5_max_prompt_dur]}
	parler_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, parler_options, parler_description, parler_attn_implementation,  parler_temperature]}
	voiceChanged_group = {'fn': voiceChanged, 'inputs': [tts_select, voice_select], 'outputs': [mars5_transcription, mars5_bool]}
	presetChanged_group = {'fn': presetChanged, 'inputs': [tts_select, model_select], 'outputs': [tortoise_num_autoregressive_samples, tortoise_diffusion_iterations, tortoise_opts_comp]}


	xtts_licence.change(toggleAcceptance, xtts_licence)
	submit_btn.click(generate_tts, [tts_select, model_select, voice_select, speak_text], audioout)
	tts_select.change(updateModels,tts_select,model_select)
	tts_select.change(**groups_group)
	mars5_bool.change(**mars5_group)
	mars5_temperature.change(**mars5_group)
	mars5_top_k.change(**mars5_group)
	mars5_top_p.change(**mars5_group)
	mars5_rep_penalty_window.change(**mars5_group)
	mars5_freq_penalty.change(**mars5_group)
	mars5_presence_penalty.change(**mars5_group)
	mars5_max_prompt_dur.change(**mars5_group)
	mars5_transcription.change(**mars5_group)
	tortoise_opts_comp.change(**tortoise_group)
	tortoise_temperature.change(**tortoise_group)
	tortoise_diffusion_temperature.change(**tortoise_group)
	tortoise_num_autoregressive_samples.change(**tortoise_group)
	tortoise_diffusion_iterations.change(**tortoise_group)
	parler_description.change(**parler_group)
	parler_attn_implementation.change(**parler_group)
	parler_options.change(**parler_group)
	parler_temperature.change(**parler_group)
	voice_select.change(**voiceChanged_group)
	model_select.change(**voiceChanged_group)
	model_select.change(**presetChanged_group)
	model_select.change(**voices_group)
	tts_select.change(updateOpts, tts_select)
	demo.load(read_log, None, console_logs, every=2)

if __name__ == "__main__":
	demo.queue().launch(server_name="0.0.0.0")

