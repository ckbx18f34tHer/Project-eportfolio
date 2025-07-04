import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
import librosa
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import json
import logging
import matplotlib.pyplot as plt
import re
import sklearn.metrics

# 檢查 GPU 是否可用
if tf.test.is_built_with_cuda():
    print("TensorFlow 已啟用 CUDA")
else:
    print("TensorFlow 未啟用 CUDA")

print("可用的設備:")
print(tf.config.list_physical_devices())

# 設置 GPU 記憶體增長
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 目前所有 GPU 設置為記憶體增長模式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'找到 {len(gpus)} 個 GPU，已啟用記憶體增長模式')
        
        # 設置可見的 GPU
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # 允許在 GPU 上進行混合精度訓練
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("已啟用混合精度訓練")
    except RuntimeError as e:
        print(f'GPU 設置錯誤: {e}')
else:
    print('未找到可用的 GPU')

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelConfig:
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512 
        self.n_fft = 2048
        self.n_mels = 128
        self.segment_duration = 0.5
        self.time_steps = int(np.ceil(self.sample_rate * self.segment_duration / self.hop_length))
        self.n_chords = 24
        self.conv_filters = [32, 64, 128, 256]
        self.dense_units = [512, 256]
        self.dropout_rate = 0.3
        self.batch_size = 32
        self.learning_rate = 0.001
        self.validation_split = 0.2
        self.pitch_shift_range = 2
        self.time_stretch_range = (0.95, 1.05)

def parse_chord_from_filename(filename: str) -> Tuple[int, bool]:
    """Parse chord information from filename"""
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    minor_qualities = {'min', 'min2', 'min3', 'min6', 'min7', 'min7b5', 'dim', 'dim7'}
    major_qualities = {'maj', 'maj2', 'maj3', 'maj6', 'maj7', 'maj7_2', 'aug', 'aug6'}
    other_qualities = {'sus2', 'sus4', 'seventh', 'sixth', 'perf4', 'perf5', 'tritone', 'octave'}
    
    try:
        parts = re.split(r'[-]', filename)
        root = parts[0]
        
        if len(root) > 1:
            if '#' in root:
                root = root[0] + '#'
            elif 'b' in root:
                root = root[0] + 'b'
                
        chord_quality = 'major'
        for part in parts:
            if part in minor_qualities:
                chord_quality = 'minor'
                break
            elif part in major_qualities or part in other_qualities:
                chord_quality = 'major'
                break
        
        root_idx = note_map.get(root, 0)
        return root_idx, chord_quality == 'minor'
        
    except Exception as e:
        logging.error(f"Error parsing filename {filename}: {str(e)}")
        return 0, False

def create_chord_label(root_idx: int, is_minor: bool, n_chords: int = 24) -> np.ndarray:
    """Create one-hot encoded chord label"""
    label = np.zeros(n_chords)
    label[root_idx + (12 if is_minor else 0)] = 1
    return label

class ChordDataGenerator(tf.keras.utils.Sequence):
    """Data generator for chord detection model"""
    def __init__(self, audio_paths: List[str], config: ModelConfig, 
                 batch_size: int = 32, augment: bool = False):
        self.audio_paths = audio_paths
        self.config = config
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.audio_paths))
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            n_steps = np.random.randint(-self.config.pitch_shift_range, 
                                      self.config.pitch_shift_range + 1)
            audio = librosa.effects.pitch_shift(audio, sr=self.config.sample_rate, 
                                              n_steps=n_steps)
        
        if np.random.random() > 0.5:
            rate = np.random.uniform(*self.config.time_stretch_range)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            
        return audio
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_size = len(batch_indices)
        
        X = np.zeros((batch_size, self.config.n_mels, self.config.time_steps, 1))
        y_chords = np.zeros((batch_size, self.config.n_chords))
        
        for i, idx in enumerate(batch_indices):
            audio_path = self.audio_paths[idx]
            audio, _ = librosa.load(
                audio_path, 
                sr=self.config.sample_rate, 
                duration=self.config.segment_duration
            )
            
            if self.augment:
                audio = self._augment_audio(audio)
            
            target_length = int(self.config.sample_rate * self.config.segment_duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels
            )
            
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            if mel_spec.shape[1] > self.config.time_steps:
                mel_spec = mel_spec[:, :self.config.time_steps]
            elif mel_spec.shape[1] < self.config.time_steps:
                pad_width = ((0, 0), (0, self.config.time_steps - mel_spec.shape[1]))
                mel_spec = np.pad(mel_spec, pad_width)
            
            X[i, :, :, 0] = mel_spec
            
            filename = Path(audio_path).stem
            root_idx, is_minor = parse_chord_from_filename(filename)
            y_chords[i] = create_chord_label(root_idx, is_minor, self.config.n_chords)
        
        return X, y_chords

def save_training_results(model, history, output_dir: str, temp_dir: str):
    """Save training results including model, history, and plots"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and history
        model.save(os.path.join(output_dir, 'final_model.keras'))
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        # Create and save training plots
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        temp_plot_path = os.path.join(temp_dir, 'training_history.png')
        plt.savefig(temp_plot_path)
        plt.close()
        
        import shutil
        shutil.move(temp_plot_path, os.path.join(output_dir, 'training_history.png'))
        
    except Exception as e:
        logging.error(f"Error saving training results: {str(e)}")
        raise

class ChordModelTrainer:
    """Trainer class for the chord detection model"""
    def __init__(self, config: ModelConfig, checkpoint_path: Optional[str] = None):
        self.config = config
        
        # 修改分配策略
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"使用 GPU 訓練，設備數量: {self.strategy.num_replicas_in_sync}")
        else:
            self.strategy = tf.distribute.get_strategy()  # 默認策略
            print("使用 CPU 訓練")
        
        with self.strategy.scope():
            self.model = self._load_model(checkpoint_path) if checkpoint_path else self._build_model()

    
    def _build_model(self) -> Model:
        """Build the model architecture"""
        with tf.device('/GPU:0'):  # 明確指定使用 GPU
            inputs = layers.Input(shape=(self.config.n_mels, self.config.time_steps, 1))
            
            x = inputs
            for filters in self.config.conv_filters:
                x = layers.Conv2D(filters, (3, 3), padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Flatten()(x)
            
            for units in self.config.dense_units:
                x = layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.Dropout(self.config.dropout_rate)(x)
            
            outputs = layers.Dense(self.config.n_chords, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # 使用 mixed_float16 策略的優化器
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizers.Adam(learning_rate=self.config.learning_rate)
            )
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model

    def _load_model(self, checkpoint_path: str) -> Model:
        """Load model from checkpoint"""
        try:
            if checkpoint_path.endswith('.h5'):
                return tf.keras.models.load_model(checkpoint_path)
            else:
                model = self._build_model()
                model.load_weights(checkpoint_path)
                return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return self._build_model()

    def train(self, train_generator, val_generator, epochs: int, 
             checkpoint_dir: str, initial_epoch: int = 0) -> Dict:
        """Train the model"""
        try:
            # 確保在 GPU 上訓練
            with tf.device('/GPU:0'):
                callbacks_list = [
                    callbacks.ModelCheckpoint(
                        os.path.join(checkpoint_dir, 
                                   'model_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min',
                        verbose=1
                    ),
                    callbacks.TensorBoard(
                        log_dir=os.path.join(checkpoint_dir, 'tensorboard_logs'),
                        histogram_freq=1,
                        profile_batch='500,520'
                    ),
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=25,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                        verbose=1
                    ),
                    callbacks.CSVLogger(
                        os.path.join(checkpoint_dir, 'training_log.csv'),
                        append=True
                    )
                ]
                
                history = self.model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                return history.history
                
        except Exception as e:
            logging.error(f"訓練錯誤: {str(e)}")
            return None
        
def prepare_training_data(audio_dir: str, config: ModelConfig) -> Tuple[Optional['ChordDataGenerator'], Optional['ChordDataGenerator']]:
    """Prepare training and validation data generators"""
    try:
        audio_files = []
        for ext in ['*.wav', '*.WAV', '*.Wav']:
            audio_files.extend(Path(audio_dir).glob(ext))
        
        if not audio_files:
            logging.error("No WAV files found")
            return None, None
        
        # Validate files
        valid_audio_paths = []
        for audio_file in audio_files:
            try:
                parse_chord_from_filename(audio_file.stem)
                valid_audio_paths.append(str(audio_file))
            except Exception as e:
                logging.warning(f"Invalid file {audio_file}: {e}")
                continue
        
        if not valid_audio_paths:
            logging.error("No valid audio files found")
            return None, None
        
        # Split data
        train_audio, val_audio = train_test_split(
            valid_audio_paths,
            test_size=config.validation_split,
            random_state=42
        )
        
        # Create generators
        return (
            ChordDataGenerator(train_audio, config, config.batch_size, True),
            ChordDataGenerator(val_audio, config, config.batch_size, False)
        )
        
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None, None

def calculate_chord_accuracies(model, val_generator) -> Dict[str, float]:
    """Calculate accuracy for each chord"""
    logging.info("Calculating chord accuracies...")
    
    try:
        # Collect predictions
        all_preds = []
        all_labels = []
        
        for X, y in val_generator:
            preds = model.predict(X, verbose=0)
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(np.argmax(y, axis=1))
        
        # Calculate accuracies
        chord_names = [
            'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 
            'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B',
            'Cm', 'C#m/Dbm', 'Dm', 'D#m/Ebm', 'Em', 'Fm', 
            'F#m/Gbm', 'Gm', 'G#m/Abm', 'Am', 'A#m/Bbm', 'Bm'
        ]
        
        accuracies = {}
        for idx, name in enumerate(chord_names):
            mask = (np.array(all_labels) == idx)
            if np.any(mask):
                accuracies[name] = sklearn.metrics.accuracy_score(
                    np.array(all_labels)[mask],
                    np.array(all_preds)[mask]
                )
                
        return accuracies
        
    except Exception as e:
        logging.error(f"Error calculating accuracies: {e}")
        return {}

def plot_results(accuracies: Dict[str, float], output_dir: str):
    """Plot and save chord accuracies"""
    if not accuracies:
        logging.warning("No accuracies to plot")
        return
        
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot accuracies
        plt.figure(figsize=(15, 6))
        chords = list(accuracies.keys())
        values = [accuracies[c] * 100 for c in chords]
        
        plt.bar(chords, values)
        plt.title("Chord Detection Accuracy")
        plt.xlabel("Chord")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot and text results
        plt.savefig(os.path.join(output_dir, 'chord_accuracies.png'))
        plt.close()
        
        with open(os.path.join(output_dir, 'chord_accuracies.txt'), 'w') as f:
            f.write("Chord Detection Accuracies:\n")
            for chord, acc in sorted(zip(chords, values), key=lambda x: x[1], reverse=True):
                f.write(f"{chord}: {acc:.1f}%\n")
                
    except Exception as e:
        logging.error(f"Error plotting results: {e}")

def train_model(audio_dir: str, output_dir: str, epochs: int = 100,
                checkpoint_path: Optional[str] = None,
                initial_epoch: int = 0):
    """Main training function"""
    try:
        config = ModelConfig()
        
        # Prepare data
        train_gen, val_gen = prepare_training_data(audio_dir, config)
        if not train_gen or not val_gen:
            return None, None
            
        # Train model
        trainer = ChordModelTrainer(config, checkpoint_path)
        history = trainer.train(
            train_gen,
            val_gen,
            epochs=epochs,
            checkpoint_dir=output_dir,
            initial_epoch=initial_epoch
        )
        
        if history:
            save_training_results(trainer.model, history, output_dir)
            
            accuracies = calculate_chord_accuracies(trainer.model, val_gen)
            plot_results(accuracies, output_dir)
            
            return trainer.model, history
            
        return None, None
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return None, None

if __name__ == "__main__":
    AUDIO_DIR = r"C:\Users\Vince\Desktop\piano_chord_datasets"
    OUTPUT_DIR = r"C:\Users\Vince\Desktop\chord_output"
    CHECKPOINT_PATH = None
    
    model, history = train_model(
        audio_dir=AUDIO_DIR,
        output_dir=OUTPUT_DIR,
        epochs=100,
        checkpoint_path=CHECKPOINT_PATH,
        initial_epoch=0
    )