print("Loading Numpy")
import numpy as np
print("Loading GRadio")
import gradio as gr
print("Loading torch, torchaudio, and librosa...")
import torch, torchaudio, librosa
print("Loading miscellanous modules...")
from TTS.utils.manage import ModelManager
import huggingface_hub as hf
import glob, os, argparse, unicodedata, json, random, psutil, requests, re, time, builtins, sys, argparse, gc, io, psutil, shutil
import scipy.io.wavfile as wav
from string import ascii_letters, digits, punctuation

loaded_tts = {}
globals = ['voice']

for f in globals:
	loaded_tts[f] = None

version = 20240823

paths = [
	'/home/app/.cache/coqui',
	'/home/app/.cache/coqui/tts',
	'/home/app/.cache/coqui/vocoder',
	'/home/app/.cache/coqui/speaker_encoder',
]

for dir in paths:
	if not os.path.exists(dir):
		os.mkdir(dir)

# Get device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
	import nvidia_smi

def getGPUStats(gpu = 0):
	if device == "cuda":
		nvidia_smi.nvmlInit()
		handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
		info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
		nvidia_smi.nvmlShutdown()
		return info
	else:
		return {}

def format_bytes(size):
	# 2**10 = 1024
	power = 2**10
	n = 0
	power_labels = {0 : '', 1: 'kB', 2: 'MB'}
	while size > power:
		if (n >= (len(power_labels) - 1)):
			break;
		else:
			size /= power
			n += 1

	return str(round(size)) + power_labels[n]

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
	['Bark','bark'],
	['Coqui','coqui'],
	['Mars5','mars5'],
	['OpenVoice', 'openvoice'],
	['Parler', 'parler'],
	['TorToiSe','tortoise']
]


# Instead of models, offer presets for TorToiSe
tortoise_presets = {
	'ultra_fast': {'label': 'Ultra Fast', 'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
	'fast': {'label': 'Fast', 'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
	'standard': {'label': 'Standard', 'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
	'high_quality': {'label': 'High Quality', 'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
	'zefie_hq': {'label': "zefie's High Quality", "num_autoregressive_samples": 512, "diffusion_iterations": 400},
}

openvoice_supported_models = [
	"checkpoints/base_speakers/EN",
	"checkpoints/base_speakers/ZH",
]

# Query Coqui for it's TTS model list, mute to prevent dumping the list to console
mute()
manager = ModelManager()
coqui_voice_models = []
for model in manager.list_models():
	if model[:10] == "tts_models":
		coqui_voice_models.append(model)
unmute()

css_style = """
.minheight {
	min-height: 86px;
}

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
bark_voice_models = [item.replace("./bark/assets/prompts/","").replace(".npz","") for item in sorted(glob.glob('./bark/assets/prompts/**/*.npz', recursive=True))]

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

def gc_collect():
	gc.collect()
	if device == 'cuda':
		torch.cuda.empty_cache()

def unload_engines(keep):
	global loaded_tts, globals
	keys = []
	for engine in loaded_tts.keys():
		keys.append(engine)
	for engine in keys:
		if keep != engine and engine not in globals:
			print('Unloading '+engine+'...')
			del loaded_tts[engine]
	gc_collect()

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
		if 'xtts' in model:
			engine = "coqui_xtts"
			if sys.stdin.read() != f"y\n":
				gr.Warning("Please agree to the Coqui Public Model License (CPML) before using XTTS models.")
				print("Please agree to the Coqui Public Model License (CPML) before using XTTS models.")
				import wave
				ifile = wave.open("/home/app/coqui/cpml.wav")
				audio = ifile.getnframes()
				audio = ifile.readframes(audio)
				audio = np.frombuffer(audio, dtype=np.int16)
				del ifile, wave
				return sr, audio
		sys.stdin.seek(0)
		speaktxt = strip_unicode(speaktxt)
		if 'TTS_HOME' not in os.environ:
			os.environ["TTS_HOME"] = "/home/app/.cache/coqui/"
		if engine not in loaded_tts:
			unload_engines(engine)
			from TTS.api import TTS as tts_api
			if engine == "coqui_xtts":
				progress(0.15, "Loading Coqui XTTS...")
				print("Loading Coqui XTTS...")
				from TTS.tts.configs.xtts_config import XttsConfig
				from TTS.tts.models.xtts import Xtts
				loaded_tts[engine] = {
					'Xtts': Xtts,
					'XttsConfig': XttsConfig,
					'model': None,
					'api': tts_api
				}
				del Xtts, XttsConfig
				progress(0.25,"Loaded Coqui XTTS`")
				print("Loaded Coqui XTTS")
			else:
				progress(0.15, "Loading Coqui...")
				print("Loading Coqui...")
				from TTS.api import TTS as tts_api
				loaded_tts[engine] = {
					'api': tts_api,
					'engine': tts_api().to(device),
					'model': None
				}
				progress(0.25,"Loaded Coqui")
				print("Loaded Coqui")
			del tts_api

		if loaded_tts[engine]['model'] != model:
			progress(0.30,"Loading model...")
			print("Loading model...")
			if engine == "coqui_xtts":
				loaded_tts[engine]['api']().download_model_by_name(model_name=model)
				xtts_checkpoint_dir =  os.environ["TTS_HOME"] + "tts/"+model.replace("/","--")+"/"
				xtts_checkpoint = xtts_checkpoint_dir+"model.pth"
				xtts_config = xtts_checkpoint_dir+"config.json"
				xtts_vocab = xtts_checkpoint_dir+"vocab.json"
				xtts_speaker_file = xtts_checkpoint_dir + "speakers_xtts.pth"
				config = loaded_tts[engine]['XttsConfig']()
				config.load_json(xtts_config)
				loaded_tts[engine]['api'] =  loaded_tts[engine]['Xtts'].init_from_config(config)
				mute(True)
				loaded_tts[engine]['api'].load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=advanced_opts['use_deepspeed'], speaker_file_path=xtts_speaker_file)
				unmute()
				if device == "cuda":
					loaded_tts[engine]['api'].cuda()
			else:
				if ',' in model:
					loaded_tts[engine]['api'].tts.configs.xtts_config.X
					tts_path = "./coqui/tts/"+model.split(",")[0]+"/"+model.split(",")[1]
					config_path = "./coqui/tts/"+model.split(",")[0]+"/config.json"
					loaded_tts[engine]['api'].load_tts_model_by_path(loaded_tts[engine]['engine'], tts_path, config_path)
				else:
					loaded_tts[engine]['api'].load_tts_model_by_name(loaded_tts[engine]['engine'], model)
			loaded_tts[engine]['model'] = model

		wavs = glob.glob(voice + "/*.wav")
		if engine == "coqui_xtts":
			progress(0.45,"Computing speaker latents...")
			print("Computing speaker latents...")
			gpt_cond_latent, speaker_embedding = loaded_tts[engine]['api'].get_conditioning_latents(wavs)
			progress(0.50,"Generating...")
			print("Generating...")
			out = loaded_tts[engine]['api'].inference(
				speaktxt,
				advanced_opts['language'],
				gpt_cond_latent,
				speaker_embedding,
				temperature=advanced_opts['temperature'],
				enable_text_splitting=True,
				length_penalty=advanced_opts['length_penalty'],
				top_p=advanced_opts['top_p'],
				top_k=advanced_opts['top_k'],
				repetition_penalty=advanced_opts['repetition_penalty'],
				speed=advanced_opts['speed']
			)
			del gpt_cond_latent, speaker_embedding
			audio = out["wav"]
		else:
			progress(0.50,"Generating...")
			print("Generating...")			
			if loaded_tts[engine]['engine'].is_multi_speaker or loaded_tts[engine]['engine'].is_multi_lingual:
				if len(wavs) > 0:
					if loaded_tts[engine]['engine'].is_multi_lingual:
							ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, speaker_wav=wavs, language=advanced_opts['language'])
					else:
						# not multilingual, just send speaker
						ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, speaker_wav=wavs)
			else:
				# no speaker
				ttsgen = loaded_tts[engine]['engine'].tts(text=speaktxt, sample_rate=sr, channels=channels, bit_depth=bit_depth)
		del wavs
	elif engine == "openvoice":
		lang = model[-2:].lower()
		if 'OPENVOICE_HOME' not in os.environ:
			os.environ["OPENVOICE_HOME"] = "/home/app/.cache/openvoice"			
		model_path = os.environ['OPENVOICE_HOME'];		
		if engine not in loaded_tts:
			unload_engines(engine)
			progress(0.15, "Loading OpenVoice...")
			print("Loading OpenVoice...")
			from openvoice import se_extractor
			from openvoice.api import BaseSpeakerTTS, ToneColorConverter
			loaded_tts[engine] = {
				'se_extractor': se_extractor,
				'BaseSpeakerTTS': BaseSpeakerTTS,
				'ToneColorConverter': ToneColorConverter,
				'model': None
			}
			del se_extractor, BaseSpeakerTTS, ToneColorConverter
			progress(0.25,"Loaded OpenVoice")
			print("Loaded OpenVoice")			
		if loaded_tts[engine]['model'] != model:
			progress(0.30,"Loading model...")
			print("Loading model...")
			hf.snapshot_download(repo_id="myshell-ai/OpenVoice", local_dir=model_path)
			loaded_tts[engine]['base_speaker_tts'] = loaded_tts[engine]['BaseSpeakerTTS'](f'{model_path}/{model}/config.json', device=device)
			loaded_tts[engine]['base_speaker_tts'].load_ckpt(f'{model_path}/{model}/checkpoint.pth')
			ckpt_converter = f'{model_path}/checkpoints/converter'
			loaded_tts[engine]['tone_color_converter'] = loaded_tts[engine]['ToneColorConverter'](f'{ckpt_converter}/config.json', device=device)
			loaded_tts[engine]['tone_color_converter'].load_ckpt(f'{ckpt_converter}/checkpoint.pth')			
			loaded_tts[engine]['source_se'] = torch.load(f'{model_path}/{model}/{lang}_style_se.pth').to(device)
			loaded_tts[engine]['model'] = model
		progress(0.45,"Computing speaker latents...")			
		print("Computing speaker latents...")
		wav, insr = torchaudio.load(voice)
		if insr != 16000:
			print("Resampling voice...")
			wav = torchaudio.functional.resample(wav, insr, 16000)
		torchaudio.save('/tmp/voice.wav', wav, 16000)
		voice = '/tmp/voice.wav'
		target_se, audio_name = loaded_tts[engine]['se_extractor'].get_se(voice, loaded_tts[engine]['tone_color_converter'], target_dir=model_path+'/processed', vad=True)
		output_file = "/tmp/openvoice.wav"
		if lang == "en":
			lang = "English"
		elif lang == "zh":
			lang = "Chinese"
		progress(0.50,"Generating...")
		print("Generating...")
		loaded_tts[engine]['base_speaker_tts'].tts(speaktxt, voice, speaker=advanced_opts['speaker'], language=lang, speed=1.0)
		encode_message = "@MyShell"
		loaded_tts[engine]['tone_color_converter'].convert(
			audio_src_path=voice, 
			src_se=loaded_tts[engine]['source_se'], 
			tgt_se=target_se, 
			output_path=output_file,
			message=encode_message)
		audio, sr = librosa.load(output_file)
		shutil.rmtree(model_path+'/processed')
		os.unlink(output_file)

	elif engine == "bark":
		if engine not in loaded_tts:
			unload_engines(engine)
			progress(0.15, "Loading Bark...")
			print("Loading Bark...")
			import bark
			loaded_tts[engine] = bark
		progress(0.25, "Loaded Bark")
		print("Loaded Bark")
		sr = loaded_tts[engine].SAMPLE_RATE;
		progress(0.50, "Generating...")
		print("Generating...")
		ttsgen = loaded_tts[engine].generate_audio(speaktxt, history_prompt="bark/assets/prompts/"+model+".npz")
	elif engine == "tortoise":
		sr = 24000
		if engine not in loaded_tts:
			unload_engines(engine)
			progress(0.15,"Loading TorToiSe...")
			print("Loading TorToiSe...")
			from tortoise import api, utils
			loaded_tts[engine] = {}
			loaded_tts[engine]['api'] = api
			loaded_tts[engine]['utils'] = utils
			del api, utils
		progress(0.25, "Loaded TorToiSe")
		print("Loaded TorToiSe")
		loaded_tts[engine]['model'] = None
		reference_clips = [loaded_tts[engine]['utils'].audio.load_audio(p, 22050) for p in glob.glob(voice + "/*.wav")]
		progress(0.50, "Generating...")
		print("Generating...")
		tts = loaded_tts[engine]['api'].TextToSpeech(use_deepspeed=advanced_opts['use_deepspeed'], kv_cache=advanced_opts['kv_cache'], half=advanced_opts['half'], device=device)
		pcm_audio = tts.tts(speaktxt, voice_samples=reference_clips, temperature=advanced_opts['temperature'], num_autoregressive_samples=advanced_opts['num_autoregressive_samples'], diffusion_iterations=advanced_opts['diffusion_iterations'], cond_free=advanced_opts['cond_free'], diffusion_temperature=advanced_opts['diffusion_temperature'])
		del tts
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = pcm_audio.detach().cpu().numpy()
	elif engine == "mars5":
		if engine not in loaded_tts:
			unload_engines(engine)
			progress(0.15, "Loading Camb.ai Mars5...")
			print("Loading Camb.ai Mars5...")
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
		del wav
		progress(0.75, "Processing audio...")
		print("Processing audio...")
		audio = output_audio.detach().cpu().numpy()
	elif engine == "parler":
		if engine not in loaded_tts:
			unload_engines(engine)
			progress(0.15, "Loading Parler...")
			print("Loading Parler...")
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
		del tokenizer, description
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
	global loaded_tts
	if (engine == 'coqui' and 'multilingual' in model) or engine == 'tortoise' or engine == 'mars5' or engine == 'openvoice':
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

def updateInfo(engine):
	if engine == "mars5":
		return gr.CheckboxGroup(info="Mars5 prefers short (< max_prompt_dur seconds) speaker wavs. (C) Camb.ai")
	elif engine == "coqui":
		return gr.CheckboxGroup(info="Coqui XTTS and Your TTS will use all wavs in a character's folder for cloning.")
	elif engine == "parler":
		return gr.CheckboxGroup(info="Parler uses textual descriptions to generate its voice.")
	elif engine == "bark":
		return gr.CheckboxGroup(info="Bark uses npz speaker models to generate its voice. You can add your own at /home/app/bark/assets/custom ~ (C) Suno, Inc")
	elif engine == "tortoise":
		return gr.CheckboxGroup(info="TorToiSe will use all wavs in a character's folder for cloning.")
	elif engine == "openvoice":
		return gr.CheckboxGroup(info="OpenVoice allows you to specify the cloned voice's tone. ~ (C) 2024 MyShell.ai")

def updateModels(engine):
	if engine == "bark":
		return gr.Dropdown(choices=bark_voice_models, value=bark_voice_models[0], label="TTS Model")
	elif engine == "coqui":
		return gr.Dropdown(choices=coqui_voice_models, value=coqui_voice_models[0], label="TTS Model")
	elif engine == "tortoise":
		tortoise_pr = [[tortoise_presets[item]['label'],item] for item in tortoise_presets.keys()]
		return gr.Dropdown(choices=tortoise_pr, value='standard', label="Preset")
	elif engine == "mars5":
		return gr.Dropdown(choices=['Camb-ai/mars5-tts'], value='Camb-ai/mars5-tts', label="TTS Model")
	elif engine == "parler":
		return gr.Dropdown(choices=['parler-tts/parler-tts-mini-v1', 'parler-tts/parler-tts-large-v1'], value='parler-tts/parler-tts-large-v1', label="TTS Model")
	elif engine == "openvoice":
		return gr.Dropdown(choices=openvoice_supported_models, value=openvoice_supported_models[0], label="TTS Model")

def updateAdvancedVisiblity(engine):
	if engine == "coqui":
		return {
			coqui_opts: gr.Group(visible=True),
			openvoice_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "openvoice":
		return {
			coqui_opts: gr.Group(visible=False),
			openvoice_opts: gr.Group(visible=True),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "tortoise":
		return {
			coqui_opts: gr.Group(visible=False),
			openvoice_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=True),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "mars5":
		return {
			coqui_opts: gr.Group(visible=False),
			openvoice_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=True),
			parler_opts: gr.Group(visible=False)
		}
	elif engine == "parler":
		return {
			coqui_opts: gr.Group(visible=False),
			openvoice_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=True)
		}
	else:
		return {
			coqui_opts: gr.Group(visible=False),
			openvoice_opts: gr.Group(visible=False),
			tortoise_opts: gr.Group(visible=False),
			mars5_opts: gr.Group(visible=False),
			parler_opts: gr.Group(visible=False)
		}

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
			'temperature': float(args[1]),
			'diffusion_temperature': args[2],
			'num_autoregressive_samples': args[3],
			'diffusion_iterations': args[4]
		}
	elif engine == "mars5":
		advanced_opts = {
			'transcription': args[5],
			'deep_clone': ('deep_clone' in args[6]),
			'use_kv_cache': ('use_kv_cache' in args[6]),
			'temperature': float(args[7]),
			'top_k': args[8],
			'top_p': args[9],
			'rep_penalty_window': args[10],
			'freq_penalty': args[11],
			'presence_penalty': args[12],
			'max_prompt_dur': args[13]
		}
	elif engine == "parler":
		compile_mode, inc_attn_mask = [False,False]
		if 'compile_mode' in args[14]:
			compile_mode = True
		if 'inc_attn_mask' in args[14]:
			inc_attn_mask = True
		advanced_opts = {
			'description': args[15],
			'attn_implementation': args[16],
			'compile_mode': 'default' if compile_mode else False,
			'inc_attn_mask': inc_attn_mask,
			'temperature': float(args[17])
		}
	elif engine == "coqui":
		advanced_opts = {
			'language': args[18],
			'temperature': float(args[19]),
			'length_penalty': float(args[20]),
			'top_p': args[21],
			'top_k': args[22],
			'speed': float(args[23]),
			'repetition_penalty': float(args[24]),
			'use_deepspeed': bool(args[25])
		}
	elif engine == "openvoice":
		advanced_opts = {
			'speaker': args[26]
		}
	else:
		advanced_opts = {}

def voiceChanged(engine, voice):
	global loaded_tts
	# Read .txt file if it exists and toggle Deep Clone accordingly
	if voice:
		loaded_tts['voice'] = voice
	if engine == "mars5":
		out = ''
		m5_bool_value = mars5_bool.value.copy()
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
	if engine == "tortoise":
		tortoise_opts_value = tortoise_opts_comp.value.copy()
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
			xtts_licence: gr.Checkbox(),
			xtts_temperature: gr.Slider(),
			xtts_top_p: gr.Slider(),
			xtts_length_penalty: gr.Slider(),
			xtts_top_k: gr.Slider(),
			xtts_language: gr.Dropdown(),
			xtts_deepspeed: gr.Checkbox(),
			coqui_opts: gr.Group()
		}
	elif engine == "coqui":
		if 'xtts' in model:
			return {
				tortoise_num_autoregressive_samples: gr.Slider(),
				tortoise_diffusion_iterations: gr.Slider(),
				tortoise_opts_comp: gr.CheckboxGroup(),
				xtts_licence: gr.Checkbox(visible=True),
				xtts_temperature: gr.Slider(visible=True),
				xtts_top_p: gr.Slider(visible=True),
				xtts_length_penalty: gr.Slider(visible=True),
				xtts_top_k: gr.Slider(visible=True),
				xtts_speed: gr.Slider(visible=True),
				xtts_repetition_penalty: gr.Slider(visible=True),
				xtts_language: gr.Dropdown(visible=True),
				xtts_deepspeed: gr.Checkbox(visible=True),
				coqui_opts: gr.Group(visible=True)
			}
		elif 'multilingual' in model:
			return {
				tortoise_num_autoregressive_samples: gr.Slider(),
				tortoise_diffusion_iterations: gr.Slider(),
				tortoise_opts_comp: gr.CheckboxGroup(),
				xtts_licence: gr.Checkbox(visible=False),
				xtts_temperature: gr.Slider(visible=False),
				xtts_top_p: gr.Slider(visible=False),
				xtts_length_penalty: gr.Slider(visible=False),
				xtts_top_k: gr.Slider(visible=False),
				xtts_speed: gr.Slider(visible=False),
				xtts_repetition_penalty: gr.Slider(visible=False),
				xtts_language: gr.Dropdown(visible=True),
				xtts_deepspeed: gr.Checkbox(visible=False),
				coqui_opts: gr.Group(visible=True)
			}
		else:
			return {
				tortoise_num_autoregressive_samples: gr.Slider(),
				tortoise_diffusion_iterations: gr.Slider(),
				tortoise_opts_comp: gr.CheckboxGroup(),
				xtts_licence: gr.Checkbox(visible=False),
				xtts_temperature: gr.Slider(visible=False),
				xtts_top_p: gr.Slider(visible=False),
				xtts_length_penalty: gr.Slider(visible=False),
				xtts_top_k: gr.Slider(visible=False),
				xtts_speed: gr.Slider(visible=False),
				xtts_repetition_penalty: gr.Slider(visible=False),
				xtts_language: gr.Dropdown(visible=False),
				xtts_deepspeed: gr.Checkbox(visible=False),
				coqui_opts: gr.Group(visible=False)
			}
	return {
		tortoise_num_autoregressive_samples: gr.Slider(),
		tortoise_diffusion_iterations: gr.Slider(),
		tortoise_opts_comp: gr.CheckboxGroup(),
		xtts_licence: gr.Checkbox(),
		xtts_temperature: gr.Slider(),
		xtts_top_p: gr.Slider(),
		xtts_length_penalty: gr.Slider(),
		xtts_top_k: gr.Slider(),
		xtts_speed: gr.Slider(),
		xtts_repetition_penalty: gr.Slider(),
		xtts_language: gr.Dropdown(),
		xtts_deepspeed: gr.Checkbox(),
		coqui_opts: gr.Group()
	}

def returnMe(value):
	return value

def startTimer(seconds):
	return gr.Timer(active=True,value=seconds)

def cancelTimer():
	return gr.Timer(active=False)

def percentage(part, whole):
	return 100 * float(part)/float(whole)

def getColorByPercent(percent):
	if percent >= 90:
		colors = ['red', 'darkred']
	elif percent >= 75:
		colors = ['orange', 'darkorange']
	elif percent >= 50:
		colors = ['yellow', 'gold']
	else:
		colors = ['green','darkgreen']
	return colors

def updateSysInfo():
	out, out2 = ["",""]
	if device == "cuda":
		gpuinfo = getGPUStats()
		percent = percentage(gpuinfo.used,gpuinfo.total);
		colors = getColorByPercent(percent)

		out += "<strong>GPU RAM</strong>: "
		out += format_bytes(gpuinfo.used)+"/"+format_bytes(gpuinfo.total)
		out += "<div style='width: 300px; background-color: "+colors[0]+"; text-align: right'>"
		out += "<div style='width: "+str(percent)+"%; text-align: left; background-color: "+colors[1]+";'> &nbsp;</div></div>"
		del gpuinfo
	meminfo = psutil.virtual_memory()
	percent = percentage(meminfo.used,meminfo.total);
	colors = getColorByPercent(percent)
	out2 += "<strong>SYS RAM</strong>: "
	out2 += format_bytes(meminfo.used)+"/"+format_bytes(meminfo.total)
	out2 += "<div style='width: 300px; background-color: "+colors[0]+"; text-align: right'>"
	out2 += "<div style='width: "+str(percent)+"%; text-align: left; background-color: "+colors[1]+";'> &nbsp;</div></div>"
	del meminfo
	return out2, out

with gr.Blocks(title="zefie's Multi-TTS v"+str(version), theme=theme, css=css_style) as demo:
	def getVoices(engine):
		if engine == "mars5" or engine == "openvoice":
			# Scan samples and srcwavs, and return each wav individually
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/**/*.wav', recursive=True))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/**/*.wav', recursive=True))])
			wavs.extend([[item.replace('./tortoise-tts/tortoise/voices','tortoise'),item] for item in sorted(glob.glob('./tortoise-tts/tortoise/voices/**/*.wav'))])
		elif engine == "coqui" or engine == "tortoise":
			# Scan samples and srcwavs, but return the folder, not each wav
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/*'))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/*'))])
			wavs.extend([[item.replace('./tortoise-tts/tortoise/voices','tortoise'),item] for item in sorted(glob.glob('./tortoise-tts/tortoise/voices/*'))])
		return wavs

	def previewAudio(file):
		return gr.Audio(label=file,type="filepath",value=file)

	def read_log():
		sys.stdout.flush()
		return sys.stdout.read()

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
	with gr.Row():
		gr.Markdown("# <p style=\"text-align: center;\">zefie's Multi-TTS v"+str(version)+"</p>")
#	with gr.Row():
#		with gr.Column():
#			sys_info = gr.HTML()
#		with gr.Column():
#			gpu_info = gr.HTML()
#		sys_info_timer = gr.Timer(value=3,active=True)
	with gr.Row():
		with gr.Tab("TTS") as tab_tts:
			with gr.Group() as main_group:
				with gr.Row():
					with gr.Column():
						tts_select = gr.Radio(tts_engines, type="value", value="coqui", label="TTS Engine", info="Coqui XTTS and Your TTS will use all wavs in a character's folder for cloning")
						model_select = gr.Dropdown(coqui_voice_models, type="value", value=coqui_voice_models[0], label="TTS Model")
						voice_select = gr.Dropdown(choices=voices, value=voice, type="value", visible=True, label="Voice Cloning", info="Place your custom voices in /home/app/srcwav/Desired Name/File1.wav, etc")
						speak_text = gr.Textbox(value="Welcome to the multi text to speech generator", label="Text to speak", lines=3)
					with gr.Column():
						audioout = gr.Audio(show_download_button=True, label="Generated Audio", type='numpy', interactive=False, scale=4)
				with gr.Row():
					submit_btn = gr.Button("Generate", variant="primary")

			with gr.Group() as coqui_opts:
				with gr.Row():
					gr.HTML("<p style=\"padding-left: 10px\">Coqui Advanced Options - Read the <a href='https://coqui.ai/cpml' target='_blank'>Coqui Public Model License (CPML)</a></p>")
				with gr.Row():
					with gr.Column():
						xtts_language = gr.Dropdown(choices=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","hu","ko","ja","hi"], value="en", label="Language")
					with gr.Column():
						with gr.Row():
							xtts_licence = gr.Checkbox(label="I agree to the CPML", value=False, info="Must be checked before using XTTS models.", elem_classes="minheight")
							xtts_deepspeed = gr.Checkbox(label="Use Deepspeed", value=True, info="Greatly increases inference speed.", elem_classes="minheight")
				with gr.Row():
					with gr.Column():
						xtts_temperature = gr.Slider(value=0.85, minimum=0.01, maximum=3, label="Temperature", info="Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability.")
						xtts_repetition_penalty = gr.Slider(value=2, maximum=5, minimum=0, label="Repetition Penalty", info='A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.')
						xtts_top_p = gr.Slider(value=0.85, minimum=0, maximum=3, label="top_p", info="If < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.")
					with gr.Column():
						xtts_speed = gr.Slider(value=1, minimum=0.1, maximum=3, label="Speed", info="The speed rate of the generated audio. (can produce artifacts if far from 1.0)")
						xtts_length_penalty = gr.Slider(value=1, minimum=-1, maximum=1, label="Length Penalty", info="Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.")
						xtts_top_k = gr.Slider(value=50, minimum=0, maximum=100, label="top_k", info="Lower values mean the decoder produces more \"likely\" (aka boring) outputs.")
			with gr.Group(visible=False) as openvoice_opts:
				with gr.Row():
					openvoice_speaker = gr.Dropdown(choices=["default", "whispering", "shouting", "excited", "cheerful", "terrified", "angry", "sad", "friendly"], value="default", label="Speaker Tone", info="The tone the TTS should apply to the text")
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
					mars5_transcription = gr.Textbox("", lines=4, placeholder="Type your transcription here, or provide a .txt file of the same name next to the .wav", label="Voice Cloning Transcription	 (Optional, but recommended)", info="You can place a .txt of the same name next to a .wav to autoload its transcription.")
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
		with gr.Tab("Browser") as tab_browser:
			with gr.Row():
				with gr.Column():
					wavbrowser = gr.FileExplorer(
						scale=1,
						label="Audio Browser",
						glob="**/**/*.wav",
						value='sample/Ava/ava_short.wav',
						file_count="single",
						root_dir="/home/app",
						elem_id="file",
					)
				with gr.Column():
					audiopreview = gr.Audio(show_download_button=True, label="/home/app/sample/Ava/ava_short.wav", interactive=False, scale=1, type="filepath", value="/home/app/sample/Ava/ava_short.wav")

		with gr.Tab("Logs") as tab_logs:
			with gr.Row():
				log_timer = gr.Timer(active=False)
				fake_console_logs = gr.Textbox(visible=False)
				console_logs = gr.HTML(elem_classes="bark_console")
			with gr.Row():
				log_refresh_rate = gr.Slider(label="Log Refresh Rate (seconds)", value=3, minimum=1, maximum=10)
			with gr.Row():
				clear_button = gr.Button("Clear Log")
				clear_button.click(clear_log)

	groups_group = {'fn': updateAdvancedVisiblity, 'inputs': tts_select, "outputs": [coqui_opts, openvoice_opts, tortoise_opts, mars5_opts, parler_opts]}
	voices_group = {'fn': updateVoicesVisibility, 'inputs': [tts_select, model_select, voice_select], 'outputs': voice_select}
	voiceChanged_group = {'fn': voiceChanged, 'inputs': [tts_select, voice_select], 'outputs': [mars5_transcription, mars5_bool], 'show_progress': False}
	presetChanged_group = {'fn': presetChanged, 'inputs': [tts_select, model_select], 'outputs': [tortoise_num_autoregressive_samples, tortoise_diffusion_iterations, tortoise_opts_comp, xtts_licence, xtts_temperature, xtts_length_penalty, xtts_top_p, xtts_top_k, xtts_speed, xtts_repetition_penalty, xtts_language, xtts_deepspeed, coqui_opts], 'show_progress': False}
	opts_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, tortoise_opts_comp, tortoise_temperature, tortoise_diffusion_temperature, tortoise_num_autoregressive_samples, tortoise_diffusion_iterations, mars5_transcription, mars5_bool, mars5_temperature, mars5_top_k, mars5_top_p, mars5_rep_penalty_window, mars5_freq_penalty, mars5_presence_penalty, mars5_max_prompt_dur, parler_options, parler_description, parler_attn_implementation, parler_temperature, xtts_language, xtts_temperature, xtts_length_penalty, xtts_top_p, xtts_top_k, xtts_speed, xtts_repetition_penalty, xtts_deepspeed, openvoice_speaker]}

	log_timer.tick(fn=read_log, inputs=None, outputs=fake_console_logs)
	tab_logs.select(fn=read_log, inputs=None, outputs=fake_console_logs)
	tab_logs.select(fn=startTimer, inputs=log_refresh_rate, outputs=log_timer)
	tab_tts.select(fn=cancelTimer, outputs=log_timer)
	tab_browser.select(fn=cancelTimer, outputs=log_timer)
	log_refresh_rate.change(fn=startTimer, inputs=log_refresh_rate, outputs=log_timer)
	wavbrowser.change(fn=previewAudio, inputs=wavbrowser, outputs=audiopreview)
	xtts_licence.change(toggleAcceptance, xtts_licence)
	submit_btn.click(**opts_group).then(generate_tts, [tts_select, model_select, voice_select, speak_text], audioout)
	tts_select.change(updateModels, tts_select, model_select)
	tts_select.change(updateInfo, tts_select, tts_select, show_progress=False)
	tts_select.change(**groups_group)
	voice_select.change(**voiceChanged_group)
	model_select.change(**voiceChanged_group)
	model_select.change(**presetChanged_group)
	model_select.change(**voices_group)
	fake_console_logs.change(returnMe, fake_console_logs, console_logs, show_progress=False)
	gc_timer = gr.Timer(value=10, active=True)
	gc_timer.tick(fn=gc_collect)
#	sys_info_timer.tick(fn=updateSysInfo, outputs=[sys_info, gpu_info], show_progress=False)

if __name__ == "__main__":
	demo.queue().launch(server_name="0.0.0.0")

