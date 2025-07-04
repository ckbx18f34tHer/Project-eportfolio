import os
import numpy as np
import tensorflow as tf
import librosa # type: ignore
import matplotlib.pyplot as plt
import json

class ModelConfig:
    
    def __init__(self):

        self.sample_rate = 22050
        self.hop_length = 512 
        self.n_fft = 2048
        self.n_mels = 128
        self.segment_duration = 3.0
        self.time_steps = int(np.ceil(self.sample_rate * self.segment_duration / self.hop_length))
        self.n_chords = 24

def estimate_octave(audio: np.ndarray, sr: int, root_note_idx: int) -> int:

    """
    估計和弦的八度數
    
    Parameters:
        audio: 音訊數據
        sr: 取樣率
        root_note_idx: 根音的音符索引 (0-11, C=0, C#=1, etc.)
    
    Returns:
        預估的八度數 (0-8)
    """

    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        fmin=librosa.note_to_hz('C0'),
        fmax=librosa.note_to_hz('C8')
    )
    
    pit_idx = magnitudes.argmax(axis=0)
    pitches_with_mag = np.array([pitches[p, i] for i, p in enumerate(pit_idx)])
    estimated_freq = np.median(pitches_with_mag[pitches_with_mag > 0])
    
    if estimated_freq > 0:
        midi_number = librosa.hz_to_midi(estimated_freq)
        
        base_midi_number = midi_number - root_note_idx

        octave = (int(base_midi_number) // 12) - 1

        octave = max(0, min(8, octave))
    else:

        octave = 4
    
    return octave

def get_chord_name(root_idx: int, is_minor: bool, octave: int = None) -> str:

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_note = notes[root_idx]
    quality = "min" if is_minor else "maj"
    if octave is not None:
        return f"{root_note}-{octave}-{quality}"
    return f"{root_note}-{quality}"

def predict_chords_from_audio(model_path: str, audio_file: str, output_dir: str = None):

    try:
        config = ModelConfig()
        
        print(f"正在載入音訊檔案: {audio_file}")
        audio, _ = librosa.load(audio_file, sr=config.sample_rate, duration=config.segment_duration)
        
        target_length = int(config.sample_rate * config.segment_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        print("處理音訊中...")
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        if mel_spec.shape[1] > config.time_steps:
            mel_spec = mel_spec[:, :config.time_steps]
        elif mel_spec.shape[1] < config.time_steps:
            pad_width = ((0, 0), (0, config.time_steps - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width)
        
        X = mel_spec.reshape(1, config.n_mels, config.time_steps, 1)
        
        print(f"載入模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("進行預測...")
        chord_pred = model.predict(X, verbose=0)
        
        chord_idx = np.argmax(chord_pred[0])
        root_idx = chord_idx % 12
        is_minor = chord_idx >= 12
        
        print("分析八度數...")
        octave = estimate_octave(audio, config.sample_rate, root_idx)
        
        predicted_chord = get_chord_name(root_idx, is_minor, octave)
        chord_prob = chord_pred[0][chord_idx]
        
        top_3_idx = np.argsort(chord_pred[0])[-3:][::-1]
        top_3_chords = []
        for idx in top_3_idx:
            root = idx % 12
            is_min = idx >= 12

            oct = estimate_octave(audio, config.sample_rate, root)
            chord_name = get_chord_name(root, is_min, oct)
            top_3_chords.append((chord_name, chord_pred[0][idx]))
        
        print("\n預測結果:")
        print(f"最可能的和弦: {predicted_chord} (信心度: {chord_prob:.2%})")
        print("\n前三個最可能的和弦:")
        for chord, prob in top_3_chords:
            print(f"{chord}: {prob:.2%}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            all_chord_names = []
            all_probabilities = []
            for i in range(24):
                root = i % 12
                is_min = i >= 12
                oct = estimate_octave(audio, config.sample_rate, root)
                all_chord_names.append(get_chord_name(root, is_min, oct))
                all_probabilities.append(chord_pred[0][i])
            
            plt.figure(figsize=(15, 6))
            
            bars = plt.bar(all_chord_names, all_probabilities)
            plt.xticks(rotation=45)
            plt.title('Chord Prediction Probability Distribution', pad=20)
            plt.xlabel('Chord')
            plt.ylabel('Probability')
            
            for idx in top_3_idx:
                bars[idx].set_color('red')
                plt.text(idx, chord_pred[0][idx], f'{chord_pred[0][idx]:.2%}', 
                        ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'chord_prediction.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            results = {
                'predicted_chord': predicted_chord,
                'confidence': float(chord_prob),
                'estimated_octave': int(octave),
                'top_3_chords': [
                    {'chord': chord, 'probability': float(prob)} 
                    for chord, prob in top_3_chords
                ],
                'all_probabilities': {
                    chord: float(prob) 
                    for chord, prob in zip(all_chord_names, all_probabilities)
                }
            }
            
            with open(os.path.join(output_dir, 'prediction_results.json'), 'w', 
                     encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            print(f"\n結果已儲存至: {output_dir}")
            
        return predicted_chord, chord_prob, top_3_chords
        
    except Exception as e:
        print(f"預測過程中發生錯誤: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    MODEL_PATH = r"C:\\Users\\studi\\OneDrive\\桌面\\Vincent\\final_model.keras"
    AUDIO_FILE = r"C:\\Users\\studi\\OneDrive\\桌面\\Vincent\\F#-4-maj-chord-0.wav"
    OUTPUT_DIR = r"C:\\Users\studi\\OneDrive\\桌面\\Vincent"
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型檔案: {MODEL_PATH}")
        if not os.path.exists(AUDIO_FILE):
            raise FileNotFoundError(f"找不到音訊檔案: {AUDIO_FILE}")
            
        predict_chords_from_audio(MODEL_PATH, AUDIO_FILE, OUTPUT_DIR)
        
    except Exception as e:
        print(f"錯誤: {str(e)}")