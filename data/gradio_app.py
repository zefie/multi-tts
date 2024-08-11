print("Loading Numpy")
import numpy as np
print("Loading GRadio")
import gradio as gr
print("Loading torch, torchaudio, and librosa...")
import torchaudio, torch, librosa
print("Preloading Coqui...")
from TTS.api import TTS
from TTS.utils import audio
from TTS.utils.manage import ModelManager
print("Preloading Bark...")
from bark import SAMPLE_RATE, generate_audio, preload_models
print("Preloading Tortoise...")
from tortoise import api,utils
print("Preloading Mars5...")
from mars5.inference import Mars5TTS, InferenceConfig as config_class
print("Preloading Parler TTS...")
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
print("Loading miscellanous modules...")
import glob, os, argparse, unicodedata, json, random, psutil, requests, re, time, builtins, sys
import scipy.io.wavfile as wav
from string import ascii_letters, digits, punctuation

version = 20240811

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


# Instead of models, offer qualities for TorToiSe
tortoise_qualities = [
	["High Quality", "high_quality"],
	["Standard", "standard"],
	["Fast", "fast"],
	["Ultra Fast", "ultra_fast"],
]

# Query Coqui for it's TTS model list, mute to prevent dumping the list to console
mute()
manager = ModelManager()
coqui_voice_models = []
for model in manager.list_models():
	if model[:10] == "tts_models":
		coqui_voice_models.append(model)
unmute()

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
		mars5, config_class = torch.hub.load(model, 'mars5_english', trust_repo=True)
		wav, sr = librosa.load(voice, mono=True)
		wav = torch.from_numpy(wav)
		if sr != mars5.sr:
			wav = torchaudio.functional.resample(wav, sr, mars5.sr)
		sr = mars5.sr
		cfg = config_class(
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
			mars5_bool.value.remove('deep_clone')

		ar_codes, output_audio = mars5.tts(speaktxt, wav, advanced_opts['transcription'], cfg=cfg)
		audio = output_audio.detach().cpu().numpy()
	if engine == "parler":
		pmodel = ParlerTTSForConditionalGeneration.from_pretrained(model).to(device)
		sr = pmodel.config.sampling_rate
		tokenizer = AutoTokenizer.from_pretrained(model)
		input_ids = tokenizer(advanced_opts['description'], return_tensors="pt").input_ids.to(device)
		prompt_input_ids = tokenizer(speaktxt, return_tensors="pt").input_ids.to(device)
		generation = pmodel.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
		audio = generation.cpu().numpy().squeeze()

	if not audio.any():
		audio = ttsgen / np.max(np.abs(ttsgen))

	return sr, audio

with gr.Blocks(title="zefie's Multi-TTS v"+str(version), theme=theme) as demo:
	def updateVoicesVisibility(tts, model):
		if (tts == 'coqui' and 'multilingual' in model) or tts == 'tortoise' or tts == 'mars5':
			voices = getVoices(tts)
			voice = voices[0][1]
			return gr.Dropdown(choices=voices, visible=True, value=voice)
		else:
			for item in tortoise_qualities:
				if model == item[1]:
					return gr.Dropdown(choices=tortoise_qualities, visible=True, value=model)

			return gr.Dropdown(choices=[['','']], visible=False, value="")

	def updateModels(value):
		if value == "bark":
			return gr.Dropdown(choices=bark_voice_models, value=bark_voice_models[0], label="TTS Model")
		if value == "coqui":
			return gr.Dropdown(choices=coqui_voice_models, value=coqui_voice_models[0], label="TTS Model")
		if value == "tortoise":
			return gr.Dropdown(choices=tortoise_qualities, value=tortoise_qualities[1][1], label="Quality")
		if value == "mars5":
			return gr.Dropdown(choices=['Camb-ai/mars5-tts'], value='Camb-ai/mars5-tts', label="TTS Model")
		if value == "parler":
			return gr.Dropdown(choices=['parler-tts/parler-tts-large-v1'], value='parler-tts/parler-tts-large-v1', label="TTS Model")

	def updateAdvancedVisiblity(value):
		if value == "tortoise":
			updateAdvancedOpts(value, tortoise_opt_comp.value)
			return {
				tortoise_opts: gr.Group(visible=True),
				mars5_opts: gr.Group(visible=False),
				parler_opts: gr.Group(visible=False)
			}
		elif value == "mars5":
			updateAdvancedOpts(value, mars5_transcription.value, mars5_bool.value, mars5_temperature.value, mars5_top_k.value, mars5_top_p.value, mars5_rep_penalty_window.value, mars5_freq_penalty.value, mars5_presence_penalty.value, mars5_max_prompt_dur.value)
			return {
				tortoise_opts: gr.Group(visible=False),
				mars5_opts: gr.Group(visible=True),
				parler_opts: gr.Group(visible=False)
			}
		elif value == "parler":
			updateAdvancedOpts(value, parler_description.value)
			return {
				tortoise_opts: gr.Group(visible=False),
				mars5_opts: gr.Group(visible=False),
				parler_opts: gr.Group(visible=True)
			}
		else:
			return {
				tortoise_opts: gr.Group(visible=False),
				mars5_opts: gr.Group(visible=False),
				parler_opts: gr.Group(visible=False)
			}

	def getVoices(value):
		if value == "mars5":
			# Scan samples and srcwavs, and return each wav individually
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/**/*.wav', recursive=True))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/**/*.wav', recursive=True))])
		elif value == "coqui" or value == "tortoise":
			# Scan samples and srcwavs, but return the folder, not each wav
			wavs = [[item.replace('./sample/',''),item] for item in sorted(glob.glob('./sample/*'))]
			wavs.extend([[item.replace('./srcwav/',''),item] for item in sorted(glob.glob('./srcwav/*'))])
		return wavs

	def updateAdvancedOpts(tts, val1, val2 = None, val3 = None, val4 = None, val5 = None, val6 = None, val7 = None, val8 = None, val9 = None, val10 = None):
		# wtf...
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
		elif tts == "parler":
			advanced_opts = {
				'description': val1
			}
		else:
			advanced_opts = {}

	def voiceChanged(tts, voice):
		m5_bool_value = mars5_bool.value.copy()
		out = ''
		# Read .txt file if it exists and toggle Deep Clone accordingly
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
		allow_flagging="never",
		submit_btn="Generate",
		show_progress='full',
	)
	with gr.Group(visible=False) as tortoise_opts:
		tortoise_opt_comp = gr.CheckboxGroup([["Use Deepspeed","use_deepspeed"],["Use KV Cache","kv_cache"],["fp16 (half)","half"]], label="Tortoise Advanced Options", value=['use_deepspeed','kv_cache'])

	with gr.Group(visible=False) as mars5_opts:
		mars5_bool = gr.CheckboxGroup([["Deep Clone (requires transcription)","deep_clone"],["Use KV Cache","use_kv_cache"]],label="Camb.ai Mars5 Advanced Options", value=['use_kv_cache'])
		mars5_transcription = gr.Textbox("", lines=4, placeholder="Type your transcription here, or provide a .txt file of the same name next to the .wav", label="Voice Cloning Transcription (Optional, but recommended)", info="You can place a .txt of the same name next to a .wav to autoload its transcription.")
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
		parler_description = gr.Textbox("A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", lines=3, placeholder="Type your description here, it should describe how you would like the voice to sound.",label="Description",info="Describe how you would like the voice to sound.")

	tts_select.change(updateModels,tts_select,model_select)
	tts_select.change(updateAdvancedVisiblity,tts_select,[tortoise_opts,mars5_opts,parler_opts])
	mars5_bool.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_temperature.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_top_k.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_top_p.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_rep_penalty_window.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_freq_penalty.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_presence_penalty.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	mars5_max_prompt_dur.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	tortoise_opt_comp.change(updateAdvancedOpts,[tts_select,tortoise_opt_comp])
	voice_select.change(voiceChanged, [tts_select, voice_select], [mars5_transcription, mars5_bool])
	model_select.change(voiceChanged, [tts_select, voice_select], [mars5_transcription, mars5_bool])
	model_select.change(updateVoicesVisibility,[tts_select,model_select],voice_select)
	mars5_transcription.change(updateAdvancedOpts,[tts_select,mars5_transcription,mars5_bool,mars5_temperature,mars5_top_k,mars5_top_p,mars5_rep_penalty_window,mars5_freq_penalty,mars5_presence_penalty,mars5_max_prompt_dur])
	parler_description.change(updateAdvancedOpts,[tts_select,parler_description])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

