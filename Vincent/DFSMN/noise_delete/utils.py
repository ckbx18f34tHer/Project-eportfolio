"""
降噪工具函數
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time


class AudioVisualizer:
    """用於視覺化音頻處理效果的工具類"""
    
    @staticmethod
    def plot_spectrogram(audio, sr, title="頻譜圖", save_path=None, n_fft=512):
        """繪製音頻頻譜圖"""
        plt.figure(figsize=(10, 4))
        
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # 計算頻譜圖
        D = np.abs(np.fft.rfft(audio.reshape(-1), n=n_fft))
        
        # 計算頻率
        freq = np.fft.rfftfreq(n_fft, 1/sr)
        
        # 繪製頻譜圖 (對數刻度)
        plt.semilogy(freq, D)
        plt.grid(True)
        plt.xlabel('頻率 (Hz)')
        plt.ylabel('振幅')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_waveform(audio, sr, title="波形圖", save_path=None):
        """繪製音頻波形圖"""
        plt.figure(figsize=(10, 3))
        
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))
        
        plt.plot(time_axis, audio)
        plt.grid(True)
        plt.xlabel('時間 (秒)')
        plt.ylabel('振幅')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def compare_before_after(original, processed, sr, title="降噪前後對比", save_dir=None):
        """對比處理前後的音頻"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 波形對比
        plt.figure(figsize=(12, 6))
        
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(processed, torch.Tensor):
            processed = processed.cpu().numpy()
            
        # 保證長度一致
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]
        
        duration = min_len / sr
        time_axis = np.linspace(0, duration, min_len)
        
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, original)
        plt.grid(True)
        plt.ylabel('原始音頻')
        plt.title(f"{title} - 波形")
        
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, processed)
        plt.grid(True)
        plt.xlabel('時間 (秒)')
        plt.ylabel('處理後音頻')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "waveform_comparison.png"))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
        # 頻譜對比
        plt.figure(figsize=(12, 6))
        
        # FFT計算
        n_fft = 1024
        orig_fft = np.abs(np.fft.rfft(original, n=n_fft))
        proc_fft = np.abs(np.fft.rfft(processed, n=n_fft))
        freq = np.fft.rfftfreq(n_fft, 1/sr)
        
        plt.subplot(2, 1, 1)
        plt.semilogy(freq, orig_fft)
        plt.grid(True)
        plt.ylabel('原始頻譜')
        plt.title(f"{title} - 頻譜")
        
        plt.subplot(2, 1, 2)
        plt.semilogy(freq, proc_fft)
        plt.grid(True)
        plt.xlabel('頻率 (Hz)')
        plt.ylabel('處理後頻譜')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "spectrum_comparison.png"))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


class BenchmarkTool:
    """用於性能測試的工具類"""
    
    @staticmethod
    def measure_processing_time(reducer, audio, sr, repetitions=5):
        """測量處理時間和實時係數"""
        audio_duration = len(audio) / sr
        
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float().to(reducer.device)
            
        # 預熱
        _ = reducer._process_audio(audio)
        
        # 計時
        times = []
        for _ in range(repetitions):
            start = time.time()
            _ = reducer._process_audio(audio)
            end = time.time()
            times.append(end - start)
            
        avg_time = sum(times) / len(times)
        rtf = avg_time / audio_duration
        
        return {
            'avg_processing_time': avg_time,
            'audio_duration': audio_duration,
            'real_time_factor': rtf,
            'is_realtime': rtf < 1.0
        }
        
    @staticmethod
    def estimate_max_chunk_size(reducer, sr, target_latency=0.03, max_size=8192):
        """估計能夠在目標延遲內處理的最大數據塊大小"""
        chunk_sizes = [256, 512, 1024, 2048, 4096, 8192]
        results = []
        
        for size in chunk_sizes:
            if size > max_size:
                break
                
            # 創建測試數據
            test_audio = torch.zeros(size, device=reducer.device)
            
            # 預熱
            _ = reducer._process_audio(test_audio)
            
            # 計時
            times = []
            repetitions = 10
            for _ in range(repetitions):
                start = time.time()
                _ = reducer._process_audio(test_audio)
                times.append(time.time() - start)
                
            avg_time = sum(times) / len(times)
            results.append({
                'chunk_size': size,
                'processing_time': avg_time,
                'latency': avg_time + (size / sr),
            })
            
            if avg_time + (size / sr) > target_latency:
                break
                
        # 找出滿足延遲要求的最大塊大小
        valid_sizes = [r for r in results if r['latency'] <= target_latency]
        if valid_sizes:
            return max(valid_sizes, key=lambda x: x['chunk_size'])
        else:
            return {'chunk_size': 256, 'processing_time': results[0]['processing_time'] if results else 0.01, 'latency': 0.02}


class AudioFileHandler:
    """音頻文件處理工具"""
    
    @staticmethod
    def batch_process_directory(reducer, input_dir, output_dir, file_ext='.wav', recursive=False):
        """批量處理目錄下的所有音頻文件"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 遍歷文件
        pattern = f"**/*{file_ext}" if recursive else f"*{file_ext}"
        total_files = 0
        processed_files = 0
        
        for input_file in input_path.glob(pattern):
            total_files += 1
            rel_path = input_file.relative_to(input_path)
            output_file = output_path / rel_path
            
            # 確保輸出目錄存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                reducer.process_file(str(input_file), str(output_file))
                processed_files += 1
            except Exception as e:
                print(f"處理檔案 '{input_file}' 失敗: {str(e)}")
                
        return {
            'total_files': total_files,
            'processed_files': processed_files,
            'failed_files': total_files - processed_files
        }
    
    @staticmethod
    def validate_audio_file(file_path, min_duration=0.1, max_file_size=100*1024*1024):
        """檢查音頻文件有效性"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, "文件不存在"
            
        if file_path.stat().st_size > max_file_size:
            return False, f"文件過大 (>{max_file_size/1024/1024:.1f}MB)"
            
        try:
            import soundfile as sf
            info = sf.info(file_path)
            if info.duration < min_duration:
                return False, f"音頻太短 ({info.duration:.2f}秒)"
            return True, info
        except Exception as e:
            return False, f"無效的音頻文件: {str(e)}"


class SignalProcessingUtil:
    """信號處理工具"""
    
    @staticmethod
    def normalize_audio(audio, target_level=-25):
        """將音頻標準化到指定分貝水平"""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        # 計算當前RMS值
        rms = torch.sqrt(torch.mean(audio**2))
        rms_db = 20 * torch.log10(rms + 1e-8)
        
        # 計算增益
        gain = 10**((target_level - rms_db) / 20)
        
        # 應用增益
        normalized = audio * gain
        
        # 確保不超出範圍
        if torch.max(torch.abs(normalized)) > 0.99:
            normalized = normalized / torch.max(torch.abs(normalized)) * 0.99
            
        return normalized
    
    @staticmethod
    def apply_fade(audio, fade_samples=1000, fade_type='both'):
        """應用淡入淡出效果"""
        if len(audio) <= fade_samples * 2:
            fade_samples = len(audio) // 4

        # 確保音頻張量在正確設備上
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # 獲取設備信息
        device = audio.device  # 新增此行
        result = audio.clone().to(device)

        if fade_type in ['in', 'both']:
            # 指定 fade_in 的設備
            fade_in = torch.linspace(0, 1, fade_samples, device=device)  # 修改此行
            result[:fade_samples] *= fade_in

        if fade_type in ['out', 'both']:
            # 指定 fade_out 的設備
            fade_out = torch.linspace(1, 0, fade_samples, device=device)  # 修改此行
            result[-fade_samples:] *= fade_out

        return result
    
    @staticmethod
    def detect_silence(audio, threshold=0.01, min_duration=0.5, sr=44100):
        """檢測音頻中的靜音段"""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # 計算能量包絡
        frame_length = 512
        hop_length = 128
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        
        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            energy[i] = np.mean(frame**2)
            
        # 識別低於閾值的幀
        silent_frames = energy < threshold**2
        
        # 轉換為時間段
        silent_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            frame_time = i * hop_length / sr
            
            if is_silent and not in_silence:
                # 進入靜音段
                in_silence = True
                silence_start = frame_time
            elif not is_silent and in_silence:
                # 離開靜音段
                in_silence = False
                silence_duration = frame_time - silence_start
                if silence_duration >= min_duration:
                    silent_regions.append((silence_start, frame_time))
                    
        # 處理最後一個可能的靜音段
        if in_silence:
            silence_duration = (len(audio) / sr) - silence_start
            if silence_duration >= min_duration:
                silent_regions.append((silence_start, len(audio) / sr))
                
        return silent_regions