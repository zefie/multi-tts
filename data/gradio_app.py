print("Loading Numpy")
import numpy as np
print("Loading GRadio")
import gradio as gr
print("Loading torch, torchaudio, and librosa...")
import torch, torchaudio, librosa
print("Loading miscellanous modules...")
from TTS.utils.manage import ModelManager
import glob, os, argparse, unicodedata, json, random, psutil, requests, re, time, builtins, sys, argparse
import scipy.io.wavfile as wav
from string import ascii_letters, digits, punctuation

parser = argparse.ArgumentParser()
parser.add_argument('-s',  '--skip-preload',help='Does not preload TTS modules, instead loads them as needed.', action='store_true')
args = parser.parse_args()

if not args.skip_preload:
	print("Preloading Coqui...")
	from TTS.api import TTS
	from TTS.utils import audio
	from TTS.utils.manage import ModelManager
	print("Preloading Suno Bark...")
	from bark import SAMPLE_RATE, generate_audio, preload_models
	print("Preloading TorToiSe...")
	from tortoise import api,utils
	print("Preloading Camb.ai Mars5...")
	from mars5.inference import Mars5TTS, InferenceConfig as config_class
	print("Preloading Parler...")
	from parler_tts import ParlerTTSForConditionalGeneration
	from transformers import AutoTokenizer

version = 20240812

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
		if args.skip_preload:
			print("Loading Coqui...")
			from TTS.api import TTS
		tts = TTS().to(device)
		print("Loading model...")
		if ',' in model:
			TTS.tts.configs.xtts_config.X
			tts_path = "./coqui/tts/"+model.split(",")[0]+"/"+model.split(",")[1]
			config_path = "./coqui/tts/"+model.split(",")[0]+"/config.json"
			TTS.load_tts_model_by_path(tts, tts_path, config_path)
		else:
			TTS.load_tts_model_by_name(tts, model)
		print("Generating...")
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
		if args.skip_preload:
			print("Loading Bark...")
			from bark import SAMPLE_RATE, generate_audio, preload_models
		print("Generating...")
		ttsgen = generate_audio(speaktxt, history_prompt="bark/assets/prompts/"+model+".npz")
	if engine == "tortoise":
		if args.skip_preload:
			print("Loading TorToiSe...")
			from tortoise import api,utils
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
		print("Generating...")
		tts = api.TextToSpeech(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
		pcm_audio = tts.tts_with_preset(speaktxt, voice_samples=reference_clips, preset=model)
		print("Processing audio...")
		audio = pcm_audio.detach().cpu().numpy()
	if engine == "mars5":
		if args.skip_preload:
			print("Loading Camb.ai Mars5...")
			from mars5.inference import Mars5TTS, InferenceConfig as config_class
		print("Loading model...")
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
		print("Generating...")
		ar_codes, output_audio = mars5.tts(speaktxt, wav, advanced_opts['transcription'], cfg=cfg)
		print("Processing audio...")
		audio = output_audio.detach().cpu().numpy()
	if engine == "parler":
		if args.skip_preload:
			print("Loading Parler...")
			from parler_tts import ParlerTTSForConditionalGeneration
			from transformers import AutoTokenizer
		print("Loading model...")
		pmodel = ParlerTTSForConditionalGeneration.from_pretrained(model, attn_implementation=advanced_opts['attn_implementation']).to(device)
		sr = pmodel.config.sampling_rate
		if advanced_opts['compile_mode']:
			print("Compiling...")
			pmodel.generation_config.cache_implementation = "static"
			pmodel.forward = torch.compile(pmodel.forward, mode=advanced_opts['compile_mode'])
			print("Done.")
		tokenizer = AutoTokenizer.from_pretrained(model)
		description = tokenizer(advanced_opts['description'], return_tensors="pt").to(device)
		prompt = tokenizer(speaktxt, return_tensors="pt").to(device)
		if advanced_opts['inc_attn_mask']:
			model_kwargs = {"input_ids": description.input_ids, "prompt_input_ids": prompt.input_ids, "attention_mask": description.attention_mask, "prompt_attention_mask": prompt.attention_mask}
		else:
			model_kwargs = {"input_ids": description.input_ids, "prompt_input_ids": prompt.input_ids}
		print("Generating...")
		generation = pmodel.generate(**model_kwargs)
		print("Processing audio...")
		audio = generation.cpu().numpy().squeeze()

	if not audio.any():
		print("Processing audio...")
		audio = ttsgen / np.max(np.abs(ttsgen))

	print("Done.")
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
			return gr.Dropdown(choices=['parler-tts/parler-tts-mini-v1', 'parler-tts/parler-tts-large-v1'], value='parler-tts/parler-tts-large-v1', label="TTS Model")

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

			updateAdvancedOpts(value, parler_description.value, parler_attn_implementation.value, parler_options.value, parler_temperature.value)
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

	def updateAdvancedOpts(tts, *args):
		# wtf...
		global advanced_opts
		if tts == "tortoise":
			use_deepspeed, kv_cache, half = [False, False, False]
			if 'use_deepspeed' in args[0]:
				use_deepspeed = True
			if 'kv_cache' in args[0]:
				kv_cache = True
			if 'half' in args[0]:
				half = True

			advanced_opts = {
				'use_deepspeed': use_deepspeed,
				'kv_cache': kv_cache,
				'half': half
			}
		elif tts == "mars5":
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
		elif tts == "parler":
			compile_mode, inc_attn_mask = [False,False]
			if 'compile_mode' in args[2]:
				compile_mode = True
			if 'inc_attn_mask' in args[2]:
				inc_attn_mask = True

			advanced_opts = {
				'description': args[0],
				'attn_implementation': args[1],
				'compile_mode': 'default' if compile_mode else False,
				'inc_attn_mask': inc_attn_mask,
				'temperature': args[3]
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
		parler_description = gr.Textbox("A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", lines=3, placeholder="Type your description here, it should describe how you would like the voice to sound",label="Description",info="Describe how you would like the voice to sound.")
		with gr.Row():
			parler_temperature = gr.Slider(value=1, minimum=0, maximum=3, label="Temperature", info="high temperatures (T>1) favour less probable outputs while low temperatures reduce randomness")
		with gr.Row():
			parler_options = gr.CheckboxGroup([['Compile Mode','compile_mode'],['Include Attn Mask','inc_attn_mask']])
			parler_attn_implementation = gr.Dropdown(['eager','sdpa'],value="eager",label="Attention Implementation")

	groups_group = {'fn': updateAdvancedVisiblity, 'inputs': tts_select, "outputs": [tortoise_opts, mars5_opts, parler_opts]}
	voices_group = {'fn': updateVoicesVisibility, 'inputs': [tts_select, model_select], 'outputs': voice_select}
	tortoise_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, tortoise_opt_comp]}
	mars5_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, mars5_transcription, mars5_bool, mars5_temperature, mars5_top_k, mars5_top_p, mars5_rep_penalty_window, mars5_freq_penalty, mars5_presence_penalty, mars5_max_prompt_dur]}
	parler_group = {'fn': updateAdvancedOpts, 'inputs': [tts_select, parler_description, parler_attn_implementation, parler_options, parler_temperature]}
	voiceChanged_group = {'fn': voiceChanged, 'inputs': [tts_select, voice_select], 'outputs': [mars5_transcription, mars5_bool]}

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
	tortoise_opt_comp.change(**tortoise_group)
	parler_description.change(**parler_group)
	parler_attn_implementation.change(**parler_group)
	parler_options.change(**parler_group)
	parler_temperature.change(**parler_group)
	voice_select.change(**voiceChanged_group)
	model_select.change(**voiceChanged_group)
	model_select.change(**voices_group)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

