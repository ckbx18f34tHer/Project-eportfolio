"""
降噪模組配置文件
"""

# 默認模型參數
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

# 音頻處理參數
AUDIO_PROCESSING_CONFIG = {
    # STFT 參數
    'n_fft': 512,
    'hop_length': 256,
    'win_length': None,  # 如果為None，使用n_fft
    'window_type': 'hann',
    
    # 頻率保護參數
    'low_freq_cutoff': 80,  # Hz，低於此頻率的內容會被保護
    'high_freq_cutoff': 16000,  # Hz，高於此頻率的內容可能被過濾
    
    # 降噪強度相關
    'default_denoising_strength': 0.95,  # 默認降噪強度 (0.0-1.0)
    'default_low_freq_preserve': 0.2,  # 默認低頻保留比例 (0.0-1.0)
    
    # 噪聲估計
    'noise_percentile': 10,  # 噪聲估計的百分位數 (1-100)
    'spectral_gate_threshold': 2.0,  # 頻譜門限閾值
}

# 實時處理參數
REALTIME_CONFIG = {
    'buffer_size': 4096,  # 默認緩衝區大小
    'max_latency': 0.03,  # 目標最大延遲 (秒)
    'min_chunk_size': 256,  # 最小處理塊大小
    'fade_samples': 50,  # 淡變樣本數量
}

# 用戶界面預設值
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
    'supported_formats': ['.wav'],  # 當前僅支持WAV格式
    'max_file_size': 100 * 1024 * 1024,  # 最大文件大小 (100MB)
    'min_duration': 0.1,  # 最小音頻長度 (秒)
    'default_output_format': 'wav'  # 默認輸出格式
}

# 日誌配置
LOGGING_CONFIG = {
    'log_level': 'INFO',  # 日誌級別: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_file': 'noise_reducer.log',  # 日誌文件名
    'max_log_size': 1024 * 1024,  # 最大日誌文件大小 (1MB)
    'backup_count': 3,  # 備份文件數量
}

# 設備配置
DEVICE_CONFIG = {
    'use_cuda': True,  # 是否使用CUDA (如果可用)
    'device': None,  # 如果為None，自動選擇 (cpu 或 cuda)
    'precision': 'float32'  # 計算精度: float32, float16 (半精度，僅CUDA)
}