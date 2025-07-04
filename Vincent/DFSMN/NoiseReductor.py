import torch
import typing
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    from noise_delete.configs import UI_PRESETS

except:

    from Vincent.DFSMN.noise_delete.configs import UI_PRESETS

try:

    from noise_delete.reducer import NoiseReducer

except:

    from Vincent.DFSMN.noise_delete.reducer import NoiseReducer

MODEL_PATH = "C:/Users/studi/OneDrive/桌面/PyStuff/Vincent/DFSMN/checkpoint_epoch_145.pth"

class DeNoiser:

    def __init__(self, MODEL_PATH: str, strength: float, lowpass: int = None)-> None:
        
        self.MODEL_PATH = MODEL_PATH
        self.strength = strength
        self.lowpass = lowpass

        print("Initializing DeNoiser_VC0227-0411var... ", end = "")

        self.denoiser  = NoiseReducer(
            
            model_path = self.MODEL_PATH, 
            denoising_strength = self.strength, 
            device = "cuda" if torch.cuda.is_available() else "cpu",
            low_freq_preserve = self.lowpass, 
        )

        print("Done.")


    def DeNoise(self, audio: np.ndarray, strength: float = None, preset: str = "voice")-> np.ndarray:
        
        if strength != None: self.strength = strength
        self.preset = preset
        self.audio = np.array(audio, dtype = "float32")
        if len(self.audio.shape) > 1 and self.audio.shape[1] > 1: self.audio = self.audio.mean(axis=1)

        if self.preset not in UI_PRESETS: 

            print("Invalid preset. Returning the original audio...")
            return 

        self.denoiser.apply_preset(self.preset)

        self.refined = self.denoiser.process_audio(self.audio)
        return self.refined

        # print(f"檔案處理完成，已保存至: {output_path}")
        
        # # 可視化結果
        # original, sr = sf.read(INPUT_FILE)
        # processed, _ = sf.read(output_path)

        # AudioVisualizer.compare_before_after(

        #     original[:,0] if original.ndim > 1 else original,
        #     processed,
        #     sr,
        #     save_dir=OUTPUT_DIR

        # )

        # print(f"Error in DeNoiser_VC0227-0411var: {str(e)}")
    
    def convert(self, path: str):

        if output_file is None:
            input_path = Path("C:/Users/studi/OneDrive/桌面/PyStuff/Vincent/DFSMN/checkpoint_epoch_145.pth")
            output_file = str(input_path.parent / f"{input_path.stem}_denoised{input_path.suffix}")
            
        # 保存處理後的音頻
        try:
            sf.write(output_file, self.refined.cpu().numpy(), 44100)
            self.logger.info(f"已保存處理後的音頻到: {output_file}")
        except Exception as e:
            self.logger.error(f"保存文件失敗: {str(e)}")
            raise

if __name__ == "__main__": 

    print("eghrji")