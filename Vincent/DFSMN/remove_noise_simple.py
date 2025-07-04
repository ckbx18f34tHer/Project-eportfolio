import torch
import soundfile as sf
from pathlib import Path
from noise_delete.reducer import NoiseReducer
from noise_delete.utils import AudioVisualizer

def main():

    MODEL_PATH = r"C:\Users\studi\OneDrive\桌面\PyStuff\Vincent\DFSMN\checkpoint_epoch_145.pth"
    INPUT_FILE = r"C:\Users\studi\OneDrive\桌面\PyStuff\Vincent\DFSMN\faoSceeE.wav"
    OUTPUT_DIR = r"C:\Users\studi\OneDrive\桌面\PyStuff\Vincent\DFSMN"
    
    print(INPUT_FILE)
    Path(OUTPUT_DIR).mkdir(exist_ok = True)

    print("初始化降噪處理器...")

    reducer = NoiseReducer(

        model_path = MODEL_PATH,
        denoising_strength = 0.8, # Discovered Variable
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    reducer.apply_preset("music") # Discovered Variable

    UI_PRESETS = {

        'voice': {
            'denoising_strength': 0.85,
            'low_freq_preserve': 0.3,
            'description': '優化人聲，高降噪，中低頻保留'
        },

        'music': {
            'denoising_strength': 0.65,
            'low_freq_preserve': 0.8,
            'description': '優化音樂，中度降噪，高低頻保留'
        },

        'environment': {
            'denoising_strength': 0.95,
            'low_freq_preserve': 0.1,
            'description': '環境錄音，最大降噪，低低頻保留'
        },

        'balanced': {
            'denoising_strength': 0.75,
            'low_freq_preserve': 0.5,
            'description': '平衡設定，適用於大多數情況'
        }
    }

    print(f"使用運算設備: {reducer.device}")

    print("\n開始處理單一檔案...")

    try:
        output_path = reducer.process_file(

            INPUT_FILE,
            # output_file=Path(OUTPUT_DIR) / "processed.wav"
        )

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

    except Exception as e:

        print(f"檔案處理失敗: {str(e)}")

if __name__ == "__main__":
    
    main()