# zefie's Multi-TTS Docker

Currently includes 
[Camb.ai Mars5](https://github.com/Camb-ai/MARS5-TTS/), [Coqui](https://github.com/coqui-ai/TTS), [MyShell.ai OpenVoice](https://github.com/myshell-ai/OpenVoice/), [Parler](https://github.com/huggingface/parler-tts), [Suno Bark](https://github.com/suno-ai/bark), and [TorToiSe](https://github.com/neonbjb/tortoise-tts).

## Usage:

**New 20240817** The application now runs as the user "app" so if you are upgrading, change your paths from `/root/.cache` to `/home/app/.cache`

Example usage (CLI):

`docker run --rm -it --gpus all -p 7860:7860 -v multitts-cache:/home/app/.cache -v ./srcwav:/home/app/srcwav zefie/multi-tts:latest`

Example usage (Docker Compose):
```yaml
 zefie-multitts:
    image: zefie/multi-tts:latest
    ports:
      - 7860:7860
    volumes:
      - ./srcwav:/home/app/srcwav
      - multitts-cache:/home/app/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [compute, utility]

  volumes:
    multitts-cache: {} 
```

For voice cloning, make a folder under "srcwav" with the name you'd like to appear in the list. Then populate that folder with clean wav files of that person speaking. Example (container path): `/home/app/srcwav/Tom/tom1.wav`

Play around with your source files until you get the voice you desire. The script will import all wavs in said folder. I recommend 16bit Mono, 22050-48000hz for best results.

## Building from Source
`git clone https://github.com/zefie/multi-tts.git --depth=1 && cd multi-tts && docker build -t multi-tts:latest .`

## Screenshots
![Coqui](https://github.com/zefie/multi-tts/blob/main/screenshots/20240817_coqui.png?raw=true)
![Parler](https://github.com/zefie/multi-tts/blob/main/screenshots/20240813_parler.png?raw=true)
![TorToiSe](https://github.com/zefie/multi-tts/blob/main/screenshots/20240813_tortoise.png?raw=true)
![Camb.ai Mars5](https://github.com/zefie/multi-tts/blob/main/screenshots/20240813_mars5.png?raw=true)
