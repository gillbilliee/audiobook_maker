import numpy as np
from kokoro import KPipeline
import tempfile

class TTSSDKokoro:
    def __init__(self, lang_code: str = "en", voice: str = "af_heart"):
        import torch

        print(f"Detected torch version: {torch.__version__}")
        print(f"CUDA enabled? → {torch.version.cuda is not None}")
        print(f"torch.cuda.is_available() → {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
            device = 'cuda'
            print("Initializing KPipeline on GPU")
        else:
            device = 'cpu'
            print("Initializing KPipeline on CPU")
        self.pipeline = KPipeline(lang_code=lang_code, device=device)
        self.voice = voice

    def synthesize(self, text: str):
        """
        Synthesize speech using Kokoro TTS.

        :param text: Input text to synthesize
        :return: (audio_data, sample_rate)
        """
        print(f"Synthesizing text: {text[:50]}...")  # Show first 50 chars
        
        # Generate speech using Kokoro TTS
        gen = self.pipeline(text=text, voice=self.voice)
        chunks = list(gen)
        print(f"Generated {len(chunks)} chunks")
        
        # Process all chunks
        audio_chunks = [chunk[2] for chunk in chunks]
        wav_arr = np.concatenate(audio_chunks, axis=0)
        wav_arr = wav_arr[:, np.newaxis]
        
        # Normalize audio to 16-bit range
        # Scale to [-1, 1] first
        wav_arr = (wav_arr - wav_arr.min()) / (wav_arr.max() - wav_arr.min()) * 2 - 1
        # Then scale to 16-bit range
        wav_arr = (wav_arr * 32767).astype(np.int16)
        
        print(f"Audio shape: {wav_arr.shape}")
        print(f"Audio min/max: {wav_arr.min()}, {wav_arr.max()}")
        
        return wav_arr, 24000
