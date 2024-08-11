# zefie's Multi-TTS Docker

Currently includes [Suno Bark](https://github.com/suno-ai/bark), [Coqui](https://github.com/coqui-ai/TTS), [Parler](https://github.com/huggingface/parler-tts), [TorToiSe](https://github.com/neonbjb/tortoise-tts), and [Camb.ai Mars5](https://github.com/Camb-ai/MARS5-TTS/).

## Usage:

Example usage (CLI):

`docker run --rm -it --gpus all -p 7860:7860 -v multitts-cache:/root/.cache -v ./srcwav:/home/app/srcwav zefie/multi-tts`
 
I recommend running from the CLI if you intend to use Coqui and their voice cloning models (e.g. the first 3 in the list), as you will have to press `Y` to accept their license before it downloads. Once you have the 3 models you can run this in a compose.

Example usage (Docker Compose):
```yaml
 zefie-multitts:
    image: zefie/multi-tts:latest
    ports:
      - 7860:7860
    volumes:
      - ./srcwav:/home/app/srcwav
      - multitts-cache:/root/.cache
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
![Coqui](https://github.com/zefie/multi-tts/blob/main/screenshots/20240812_coqui.png?raw=true)
![Parler](https://github.com/zefie/multi-tts/blob/main/screenshots/20240812_parler.png?raw=true)
![Camb.ai Mars5](https://github.com/zefie/multi-tts/blob/main/screenshots/20240812_mars5.png?raw=true)
