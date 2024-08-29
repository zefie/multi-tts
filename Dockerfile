#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    wget curl software-properties-common python3-pip git espeak-ng

RUN cd /root && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
	dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-4 portaudio19-dev

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyreadline3==3.4.1 requests==2.31.0 torch>=2.2.1 \
    pocketsphinx==5.0.3 TTS==0.22.0 funcy==2.0 gradio==4.41.0 \
    deepspeed==0.14.4 Ninja==1.11.1.1 bark==0.1.5 flash-attn==2.6.3

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install librosa torchaudio vocos encodec safetensors regex

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/myshell-ai/MeloTTS.git unidic mecab-python3

RUN python -m unidic download

RUN mkdir -p /home/app

RUN mkdir -p /home/app/coqui && \
    ln -s /home/app/.cache/coqui/tts /home/app/coqui/tts && \
    ln -s /home/app/.cache/coqui/vocoder /home/app/coqui/vocoder && \
    ln -s /home/app/.cache/coqui/speaker_encoder /home/app/coqui/speaker_encoder

COPY ./data/tortoise-tts /home/app/tortoise-tts
COPY ./data/parler-tts /home/app/parler-tts
COPY ./data/OpenVoice /home/app/OpenVoice

RUN --mount=type=cache,target=/root/.cache/pip cd /home/app && \
    pip install ./OpenVoice ./tortoise-tts ./parler-tts nvidia-ml-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade protobuf

RUN groupadd app && useradd -d /home/app -g app -G users app && chown app:app -R /home/app


COPY --chown=app:app ./data /home/app

USER "app"
EXPOSE 7860
WORKDIR "/home/app"
CMD ["python", "gradio_app.py"]
