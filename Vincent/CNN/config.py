import music21
import librosa
import numpy as np
from enum import Enum
from dataclasses import dataclass
from tensorflow.keras import layers # type: ignore

class ChordType(Enum):
    """Enum representing different chord types"""
    MAJOR = (0, "Major")
    MINOR = (1, "Minor")
    DIMINISHED = (2, "Diminished")
    AUGMENTED = (3, "Augmented")
    DOMINANT_SEVENTH = (4, "Dominant Seventh")
    MAJOR_SEVENTH = (5, "Major Seventh")
    MINOR_SEVENTH = (6, "Minor Seventh")
    HALF_DIMINISHED = (7, "Half Diminished")
    FULLY_DIMINISHED = (8, "Fully Diminished")
    SUSPENDED_FOURTH = (9, "Suspended Fourth")
    SUSPENDED_SECOND = (10, "Suspended Second")

@dataclass
class ModelConfig:
    """Configuration class for the chord separation model."""
    sample_rate: int = 22050
    n_mels: int = 128
    hop_length: int = 512
    win_length: int = 2048
    n_fft: int = 2048
    
    min_midi_note: int = 21  # A0
    max_midi_note: int = 108  # C8
    total_notes: int = max_midi_note - min_midi_note + 1
    
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 0.001

class MelSpectrogramProcessor:
    """Handles audio processing and mel spectrogram conversion."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def process(self, audio_path: str) -> np.ndarray:
        """Convert audio file to mel spectrogram."""
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
        
        return mel_spec

class DataAugmenter:
    """Handles data augmentation for audio and MIDI pairs."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def augment(self, features: np.ndarray, labels: dict) -> tuple:
        """Apply augmentation to features and labels."""
        time_stretch_factor = np.random.uniform(0.9, 1.1)
        features_aug = librosa.effects.time_stretch(features, rate=time_stretch_factor)
        
        labels_aug = {
            'notes': self._adjust_labels_timing(labels['notes'], time_stretch_factor),
            'chords': self._adjust_labels_timing(labels['chords'], time_stretch_factor)
        }
        
        pitch_steps = np.random.randint(-2, 3)
        if pitch_steps != 0:
            features_aug = librosa.effects.pitch_shift(
                features_aug, 
                sr=self.config.sample_rate,
                n_steps=pitch_steps
            )
        
        return features_aug, labels_aug
    
    def _adjust_labels_timing(self, labels: np.ndarray, stretch_factor: float) -> np.ndarray:
        """Adjust label timings based on time stretch factor."""
        new_length = int(labels.shape[0] * stretch_factor)
        return librosa.util.fix_length(labels, size=new_length, axis=0)

class ChordSeparator:
    """Handles chord identification and separation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def identify_chord_type(self, notes: list) -> tuple:
        chord = music21.chord.Chord(notes)
        
        root = chord.root().name
        quality = chord.quality
        
        if quality == 'major':
            return ChordType.MAJOR, root
        elif quality == 'minor':
            return ChordType.MINOR, root
        elif quality == 'diminished':
            return ChordType.DIMINISHED, root
        elif quality == 'augmented':
            return ChordType.AUGMENTED, root
        elif quality == 'dominant-seventh':
            return ChordType.DOMINANT_SEVENTH, root
        elif quality == 'major-seventh':
            return ChordType.MAJOR_SEVENTH, root
        elif quality == 'minor-seventh':
            return ChordType.MINOR_SEVENTH, root
        elif chord.isDiminishedSeventh():
            return ChordType.FULLY_DIMINISHED, root
        elif chord.isHalfDiminishedSeventh():
            return ChordType.HALF_DIMINISHED, root
        elif 'sus4' in chord.commonName:
            return ChordType.SUSPENDED_FOURTH, root
        elif 'sus2' in chord.commonName:
            return ChordType.SUSPENDED_SECOND, root
        else:
            return ChordType.MAJOR, root  