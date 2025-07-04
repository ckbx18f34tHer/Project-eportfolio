"""
降噪模組配置文件
"""

DEFAULT_MODEL_CONFIG = {

    # DFSMN 模型結構參數
    'input_dim': 257,
    'hidden_dims': [512, 384, 256, 128],
    'output_dim': 257,
    'dropout': 0.2,
    
    # 記憶模塊配置
    'memory_size': 50,
    'look_ahead': 5,
    'stride': 1,
    
    # 頻率追蹤器參數
    'freq_tracking_hidden_dim': 512,
    
    # 爆音檢測器參數
    'burst_threshold': 0.6,
    
    # 自適應陷波濾波器參數
    'num_filters': 4
}


AUDIO_PROCESSING_CONFIG = {
    
    'n_fft': 512,
    'hop_length': 256,
    'win_length': None,  # n_fft if None
    'window_type': 'hann',
    
    'low_freq_cutoff': 80,  # PROTECTED below this frequency
    'high_freq_cutoff': 16000,
    
    'default_denoising_strength': 0.95,
    'default_low_freq_preserve': 0.2,
    
    
    'noise_percentile': 10,  # approximating the ratio of noise
    'spectral_gate_threshold': 2.0,  # 頻譜門限閾值
}

# 實時處理參數
REALTIME_CONFIG = {
    'buffer_size': 4096,  # 默認緩衝區大小
    'max_latency': 0.03,  # 目標最大延遲 (秒)
    'min_chunk_size': 256,  # 最小處理塊大小
    'fade_samples': 50,  # 淡變樣本數量
}


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

# 文件處理參數
FILE_PROCESSING = {
    'supported_formats': ['.wav'],
    'max_file_size': 100 * 1024 * 1024,
    'min_duration': 0.1,  # 最小音頻長度 (秒)
    'default_output_format': 'wav',
}

# 日誌配置
LOGGING_CONFIG = {
    'log_level': 'INFO',  # 日誌級別: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_file': 'noise_reducer.log',
    'max_log_size': 1024 * 1024,
    'backup_count': 3,
}

DEVICE_CONFIG = {
    'use_cuda': True,
    'device': None,
    'precision': 'float32'
}