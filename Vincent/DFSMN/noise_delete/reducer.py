import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import logging

try:

    from Vincent.DFSMN.noise_delete.models import EnhancedDFSMN
    from Vincent.DFSMN.noise_delete.utils import AudioVisualizer, BenchmarkTool, AudioFileHandler, SignalProcessingUtil
    from Vincent.DFSMN.noise_delete.configs import AUDIO_PROCESSING_CONFIG, REALTIME_CONFIG, DEVICE_CONFIG, UI_PRESETS

except:

    from noise_delete.models import EnhancedDFSMN
    from noise_delete.utils import AudioVisualizer, BenchmarkTool, AudioFileHandler, SignalProcessingUtil
    from noise_delete.configs import AUDIO_PROCESSING_CONFIG, REALTIME_CONFIG, DEVICE_CONFIG, UI_PRESETS

class NoiseReducer:

    def __init__(self, model_path, denoising_strength=None, 
                 low_freq_preserve=None, device=None, n_fft=None, hop_length=None):
        
        self._setup_logging()

        self.device = device or DEVICE_CONFIG['device'] or ('cuda' if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else 'cpu')
        self.model = self._load_model(model_path)
        self.n_fft = n_fft or AUDIO_PROCESSING_CONFIG['n_fft']
        self.hop_length = hop_length or AUDIO_PROCESSING_CONFIG['hop_length']
        self.window = torch.hann_window(self.n_fft).to(self.device)
        self.denoising_strength = denoising_strength or AUDIO_PROCESSING_CONFIG['default_denoising_strength']
        self.low_freq_preserve = low_freq_preserve or AUDIO_PROCESSING_CONFIG['default_low_freq_preserve']
        
        self._buffer = torch.zeros(0, device=self.device)
        self._overlap = self.n_fft - self.hop_length
        self._sr = None
               
    def _setup_logging(self):

        """設置日誌記錄"""

        self.logger = logging.getLogger("NoiseReducer")

        if not self.logger.handlers:

            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def _load_model(self, model_path):

        """載入降噪模型"""

        self.logger.info(f"載入模型: {model_path}")
        model = EnhancedDFSMN(input_dim=257)

        try:

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            return model
        
        except Exception as e:

            self.logger.error(f"載入模型失敗: {str(e)}")
            raise
        
    def _normalize_spectrum(self, spec):

        return (spec - spec.mean()) / (spec.std() + 1e-8)
    
    def _estimate_noise_profile(self, mag, percentile=None):

        percentile = percentile or AUDIO_PROCESSING_CONFIG['noise_percentile']

        noise_profile = torch.quantile(mag, percentile/100, dim=1, keepdim=True)

        return noise_profile
    
    def _spectral_gating(self, mag, noise_profile, threshold=None, reduction=0.8):

        threshold = threshold or AUDIO_PROCESSING_CONFIG['spectral_gate_threshold']

        snr = mag / (noise_profile + 1e-8)

        mask = 1.0 - torch.exp(-(snr / threshold)**2)
        return mag * mask
    
    def _postprocess_reduction(self, audio):

        """後處理以減少處理後的殘余噪聲"""

        spec = torch.stft(audio,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=self.window,
                          return_complex=True)
        
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        noise_est = self._estimate_noise_profile(mag, percentile=15)

        mag_clean = self._spectral_gating(mag, noise_est, threshold=1.5, reduction=0.6)
        
        spec_clean = mag_clean * torch.exp(1j * phase)
        audio_clean = torch.istft(spec_clean,
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length,
                                 window=self.window,
                                 length=len(audio))
        
        return audio_clean
    
    def process_stream(self, audio_chunk, sample_rate=44100):

        """
        處理音頻流數據塊
        
        參數:
            audio_chunk: numpy數組, 一段音頻樣本
            sample_rate: 採樣率
            
        返回:
            處理後的音頻數據塊
        """

        if self._sr is None:

            self._sr = sample_rate
            self.logger.debug(f"初始化流處理, 採樣率: {sample_rate}Hz")

        elif self._sr != sample_rate:

            self.logger.warning(f"採樣率不一致: 預期 {self._sr} Hz, 收到 {sample_rate} Hz")
            raise ValueError(f"採樣率不一致: 預期 {self._sr} Hz, 收到 {sample_rate} Hz")

        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float().to(self.device)
        
        self._buffer = torch.cat([self._buffer, audio_chunk])

        output_length = len(self._buffer) - self._overlap
        if output_length <= 0: return np.array([])
            
        processable = self._buffer[:output_length]
        result = self._process_audio(processable)

        self._buffer = self._buffer[output_length:]
        
        return result.cpu().numpy()
    
    def flush(self):

        """
        處理並返回緩衝區中的所有剩餘音頻
        """

        if len(self._buffer) == 0:

            return np.array([])

        padding = torch.zeros(self._overlap, device=self.device)
        padded_buffer = torch.cat([self._buffer, padding])
        result = self._process_audio(padded_buffer)

        self._buffer = torch.zeros(0, device=self.device)
        self.logger.debug("緩衝區已清空")
        
        return result.cpu().numpy()
    
    def _process_audio(self, audio_tensor):

        if len(audio_tensor) < self.n_fft:

            self.logger.warning(f"輸入音訊過短 ({len(audio_tensor)} samples), 補零至 {self.n_fft}")
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.n_fft - len(audio_tensor)))

        spec = torch.stft(
            audio_tensor,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            window = self.window,
            return_complex = True,
            pad_mode = 'constant'  # Specified padding method
        )
        
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        
        noise_profile = self._estimate_noise_profile(mag)

        orig_mag = mag.clone()
        mag_norm = self._normalize_spectrum(mag)
        mag_norm = mag_norm.T.unsqueeze(0)
        
        with torch.no_grad():

            output, _ = self.model(mag_norm)

            if self._sr:

                low_freq_bins = int(AUDIO_PROCESSING_CONFIG['low_freq_cutoff'] / (self._sr / self.n_fft))

                if low_freq_bins > 0:

                    mag_norm_T = mag_norm.transpose(1, 2)
                    output_T = output.transpose(1, 2)

                    output_T[:, :low_freq_bins, :] = (
                        output_T[:, :low_freq_bins, :] * (1 - self.low_freq_preserve) + 
                        mag_norm_T[:, :low_freq_bins, :] * self.low_freq_preserve
                    )

                    output = output_T.transpose(1, 2)

            output = output.squeeze(0).T
            mean_mag = mag.mean()
            std_mag = mag.std()
            output = output * (std_mag + 1e-8) + mean_mag
        
        output = self._spectral_gating(output, noise_profile)
        
        enhanced_mag = orig_mag * (1 - self.denoising_strength) + output * self.denoising_strength
        
        enhanced_spec = enhanced_mag * torch.exp(1j * phase)
        

        if self._sr: # mild

            cutoff_freq = 50  # low frequency
            cutoff_bin = max(1, int(cutoff_freq / (self._sr / self.n_fft)))
            high_pass_filter = torch.ones_like(enhanced_spec)

            if cutoff_bin > 1: high_pass_filter[:cutoff_bin, :] = torch.linspace(0.1, 1.0, cutoff_bin).unsqueeze(1) # high frequency

            enhanced_spec = enhanced_spec * high_pass_filter # Filtering

        enhanced_audio = torch.istft(enhanced_spec,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    window=self.window,
                                    length=len(audio_tensor))

        if self.denoising_strength > 0.8:
            enhanced_audio = self._postprocess_reduction(enhanced_audio)
        
        # 確保音頻幅度在合理範圍內
        if torch.max(torch.abs(enhanced_audio)) > 0.99:
            enhanced_audio = enhanced_audio / torch.max(torch.abs(enhanced_audio)) * 0.99
            
        return enhanced_audio
    
    def process_file(self, input_file, output_file=None, normalize=False): 
        """
        處理整個音頻文件
        
        參數:
            input_file: 輸入音頻文件路徑
            output_file: 輸出音頻文件路徑 (如果為None，則使用原文件名加上"_denoised"後綴)
            normalize: 是否標準化輸出音頻
            
        返回:
            處理後的音頻路徑
        """
        self.logger.info(f"處理文件: {input_file}")
        
        # 檢查文件有效性
        valid, info_or_error = AudioFileHandler.validate_audio_file(input_file)
        if not valid:
            self.logger.error(f"無效的輸入文件: {info_or_error}")
            raise ValueError(f"無效的輸入文件: {info_or_error}")
            
        # 讀取音頻
        try:
            audio, sr = sf.read(input_file, dtype='float32')
        except Exception as e:
            self.logger.error(f"讀取文件失敗: {str(e)}")
            raise
            
        # 將音頻轉為單聲道（如果是立體聲）
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
            
        # 設置採樣率
        self._sr = sr
            
        # 轉為張量並處理
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # 淡入淡出處理以避免邊緣效應
        fade_samples = min(int(sr * 0.01), len(audio) // 20)  # 10毫秒或信號長度的5%
        audio_tensor = SignalProcessingUtil.apply_fade(audio_tensor, fade_samples, fade_type='both')
        
        # 處理音頻
        self.logger.info("開始降噪處理...")
        processed_audio = self._process_audio(audio_tensor)
        
        # 標準化音頻電平（如果需要）
        if normalize:
            processed_audio = SignalProcessingUtil.normalize_audio(processed_audio)
            
        # 確定輸出文件路徑
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_denoised{input_path.suffix}")
            
        # 保存處理後的音頻
        try:
            sf.write(output_file, processed_audio.cpu().numpy(), sr)
            self.logger.info(f"已保存處理後的音頻到: {output_file}")
        except Exception as e:
            self.logger.error(f"保存文件失敗: {str(e)}")
            raise
            
        return output_file
        
    def set_denoising_strength(self, strength):

        """設置降噪強度"""

        if 0.0 <= strength <= 1.0:
            self.denoising_strength = strength
            self.logger.debug(f"降噪強度設為: {strength}")
        else:
            self.logger.warning(f"無效的降噪強度: {strength}，應在0.0-1.0範圍內")
            
    def set_low_freq_preserve(self, preserve_ratio):

        """設置低頻保留比例"""

        if 0.0 <= preserve_ratio <= 1.0:
            self.low_freq_preserve = preserve_ratio
            self.logger.debug(f"低頻保留比例設為: {preserve_ratio}")
        else:
            self.logger.warning(f"無效的低頻保留比例: {preserve_ratio}，應在0.0-1.0範圍內")
    
    def apply_preset(self, preset_name):

        """應用預設配置"""

        if preset_name in UI_PRESETS:
            preset = UI_PRESETS[preset_name]
            self.set_denoising_strength(preset['denoising_strength'])
            self.set_low_freq_preserve(preset['low_freq_preserve'])
            self.logger.info(f"已應用預設配置: {preset_name} ({preset['description']})")
            return True
        else:
            self.logger.warning(f"未知的預設配置: {preset_name}")
            return False
            
    def estimate_performance(self, duration=5.0):

        """估計系統性能並返回實時處理能力"""

        # 創建測試音頻
        sr = 44100
        test_samples = int(duration * sr)
        test_audio = torch.randn(test_samples, device=self.device) * 0.1
        
        # 使用基準測試工具
        perf = BenchmarkTool.measure_processing_time(self, test_audio, sr)
        
        # 估計最大塊大小
        chunk_info = BenchmarkTool.estimate_max_chunk_size(self, sr)
        
        return {
            'real_time_factor': perf['real_time_factor'],
            'is_realtime_capable': perf['is_realtime'],
            'recommended_chunk_size': chunk_info['chunk_size'],
            'expected_latency': chunk_info['latency'],
            'device': self.device
        }
    
    def process_audio(self, wave: np.ndarray, normalize=True): 

        # self.logger.info(f"處理文件: {input_file}")
        
        # # 檢查文件有效性
        # valid, info_or_error = AudioFileHandler.validate_audio_file(input_file)
        # if not valid:
        #     self.logger.error(f"無效的輸入文件: {info_or_error}")
        #     raise ValueError(f"無效的輸入文件: {info_or_error}")
            
        # # 讀取音頻
        # try:
        #     audio, sr = sf.read(input_file, dtype='float32')
        # except Exception as e:
        #     self.logger.error(f"讀取文件失敗: {str(e)}")
        #     raise

        audio = np.array(wave, dtype = "float32")
            
        # 將音頻轉為單聲道（如果是立體聲）
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
            
        # 設置採樣率
        self._sr = 44100
            
        # 轉為張量並處理
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # 淡入淡出處理以避免邊緣效應
        fade_samples = min(int(self._sr * 0.01), len(audio) // 20)  # 10毫秒或信號長度的5%
        audio_tensor = SignalProcessingUtil.apply_fade(audio_tensor, fade_samples, fade_type='both')
        
        # 處理音頻
        self.logger.info("開始降噪處理...")
        processed_audio = self._process_audio(audio_tensor)
        
        # 標準化音頻電平（如果需要）
        if normalize:

            processed_audio = SignalProcessingUtil.normalize_audio(processed_audio)

        return processed_audio
            
        