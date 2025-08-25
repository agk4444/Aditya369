"""
Voice Analyzer for EQM Emotion Detection

This module provides voice analysis capabilities to extract emotional
features from speech patterns, tone, and acoustic characteristics.
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VoiceFeatures:
    """Container for extracted voice features"""
    pitch_features: Dict[str, float]
    intensity_features: Dict[str, float]
    temporal_features: Dict[str, float]
    spectral_features: Dict[str, float]
    emotional_indicators: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class VoiceAnalysisConfig:
    """Configuration for voice analysis"""
    sample_rate: int = 16000  # Hz
    frame_length: int = 512
    hop_length: int = 256
    n_mfcc: int = 13
    fmin: float = 75.0  # Minimum frequency for pitch detection
    fmax: float = 1000.0  # Maximum frequency for pitch detection
    voice_activity_threshold: float = 0.1


class VoiceAnalyzer:
    """Advanced voice analysis for emotion detection"""

    def __init__(self, config: VoiceAnalysisConfig):
        self.config = config

    def analyze_audio_segment(self, audio_data: np.ndarray, sample_rate: int) -> VoiceFeatures:
        """
        Analyze a segment of audio data for emotional features

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            VoiceFeatures object with emotional analysis
        """
        try:
            # Ensure audio is in the right format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=0)  # Convert to mono

            # Extract all voice features
            pitch_features = self._extract_pitch_features(audio_data, sample_rate)
            intensity_features = self._extract_intensity_features(audio_data)
            temporal_features = self._extract_temporal_features(audio_data)
            spectral_features = self._extract_spectral_features(audio_data, sample_rate)
            emotional_indicators = self._extract_emotional_indicators(
                pitch_features, intensity_features, temporal_features, spectral_features
            )

            metadata = {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'audio_length_seconds': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'voice_activity_detected': self._detect_voice_activity(audio_data)
            }

            return VoiceFeatures(
                pitch_features=pitch_features,
                intensity_features=intensity_features,
                temporal_features=temporal_features,
                spectral_features=spectral_features,
                emotional_indicators=emotional_indicators,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error analyzing audio segment: {e}")
            return self._get_empty_features()

    def _extract_pitch_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract pitch-related features"""
        features = {}

        try:
            # Fundamental frequency (pitch)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                sr=sample_rate,
                frame_length=self.config.frame_length,
                hop_length=self.config.hop_length
            )

            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]
            voiced_flag_clean = voiced_flag[~np.isnan(f0)]

            if len(f0_clean) > 0:
                # Pitch statistics
                features['pitch_mean'] = np.mean(f0_clean)
                features['pitch_std'] = np.std(f0_clean)
                features['pitch_range'] = np.max(f0_clean) - np.min(f0_clean)
                features['pitch_median'] = np.median(f0_clean)

                # Pitch contour features
                if len(f0_clean) > 5:
                    pitch_diff = np.diff(f0_clean)
                    features['pitch_slope'] = np.mean(pitch_diff)
                    features['pitch_variability'] = np.std(pitch_diff)

                # Voice quality
                features['voiced_ratio'] = np.mean(voiced_flag_clean)
                features['pitch_jitter'] = self._calculate_jitter(f0_clean)

            else:
                # No voiced segments detected
                features.update({
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_range': 0.0,
                    'pitch_median': 0.0,
                    'pitch_slope': 0.0,
                    'pitch_variability': 0.0,
                    'voiced_ratio': 0.0,
                    'pitch_jitter': 0.0
                })

        except Exception as e:
            logger.warning(f"Error extracting pitch features: {e}")
            features.update({
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0,
                'pitch_median': 0.0,
                'pitch_slope': 0.0,
                'pitch_variability': 0.0,
                'voiced_ratio': 0.0,
                'pitch_jitter': 0.0
            })

        return features

    def _extract_intensity_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract intensity/loudness features"""
        features = {}

        try:
            # RMS energy (intensity)
            rms = librosa.feature.rms(y=audio, frame_length=self.config.frame_length, hop_length=self.config.hop_length)

            if rms.size > 0:
                rms_values = rms.flatten()
                features['intensity_mean'] = np.mean(rms_values)
                features['intensity_std'] = np.std(rms_values)
                features['intensity_max'] = np.max(rms_values)
                features['intensity_range'] = features['intensity_max'] - np.min(rms_values)

                # Intensity dynamics
                if len(rms_values) > 5:
                    intensity_diff = np.diff(rms_values)
                    features['intensity_slope'] = np.mean(intensity_diff)
                    features['intensity_variability'] = np.std(intensity_diff)

                # Voice activity detection
                features['intensity_above_threshold'] = np.mean(rms_values > self.config.voice_activity_threshold)

        except Exception as e:
            logger.warning(f"Error extracting intensity features: {e}")
            features.update({
                'intensity_mean': 0.0,
                'intensity_std': 0.0,
                'intensity_max': 0.0,
                'intensity_range': 0.0,
                'intensity_slope': 0.0,
                'intensity_variability': 0.0,
                'intensity_above_threshold': 0.0
            })

        return features

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal/speech rate features"""
        features = {}

        try:
            # Zero crossing rate (speech rate indicator)
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.config.frame_length, hop_length=self.config.hop_length)

            if zcr.size > 0:
                zcr_values = zcr.flatten()
                features['zero_crossing_rate_mean'] = np.mean(zcr_values)
                features['zero_crossing_rate_std'] = np.std(zcr_values)

                # Speech pauses (detect silent periods)
                rms = librosa.feature.rms(y=audio, frame_length=self.config.frame_length, hop_length=self.config.hop_length)
                rms_values = rms.flatten()

                # Detect pauses (low energy periods)
                threshold = np.mean(rms_values) * 0.1
                pause_frames = np.sum(rms_values < threshold)
                features['pause_ratio'] = pause_frames / len(rms_values)

                # Speech tempo (words per minute estimation)
                # This is a simplified estimation based on energy bursts
                energy_peaks = librosa.util.peak_pick(rms_values, 3, 3, 3, 5, 0.5, 10)
                if len(energy_peaks) > 5:
                    peak_intervals = np.diff(energy_peaks)
                    avg_interval_frames = np.mean(peak_intervals)
                    # Estimate speaking rate (very approximate)
                    features['estimated_speech_rate'] = 60 / (avg_interval_frames * self.config.hop_length / 44100)  # WPM

        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            features.update({
                'zero_crossing_rate_mean': 0.0,
                'zero_crossing_rate_std': 0.0,
                'pause_ratio': 0.0,
                'estimated_speech_rate': 0.0
            })

        return features

    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract spectral features including MFCCs"""
        features = {}

        try:
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.frame_length,
                hop_length=self.config.hop_length
            )

            if mfccs.size > 0:
                # MFCC statistics
                for i in range(self.config.n_mfcc):
                    mfcc_values = mfccs[i]
                    features[f'mfcc_{i+1}_mean'] = np.mean(mfcc_values)
                    features[f'mfcc_{i+1}_std'] = np.std(mfcc_values)

                # Spectral centroid
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio, sr=sample_rate,
                    n_fft=self.config.frame_length,
                    hop_length=self.config.hop_length
                )

                if spectral_centroid.size > 0:
                    centroid_values = spectral_centroid.flatten()
                    features['spectral_centroid_mean'] = np.mean(centroid_values)
                    features['spectral_centroid_std'] = np.std(centroid_values)

                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio, sr=sample_rate,
                    n_fft=self.config.frame_length,
                    hop_length=self.config.hop_length
                )

                if spectral_rolloff.size > 0:
                    rolloff_values = spectral_rolloff.flatten()
                    features['spectral_rolloff_mean'] = np.mean(rolloff_values)
                    features['spectral_rolloff_std'] = np.std(rolloff_values)

        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            # Set default values for MFCCs and spectral features
            for i in range(self.config.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0

            features.update({
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0,
                'spectral_rolloff_std': 0.0
            })

        return features

    def _extract_emotional_indicators(self, pitch: Dict, intensity: Dict,
                                    temporal: Dict, spectral: Dict) -> Dict[str, float]:
        """Extract emotion-specific indicators from voice features"""
        indicators = {}

        try:
            # Stress and anxiety indicators
            indicators['stress_indicator'] = (
                pitch.get('pitch_variability', 0) * 0.4 +
                intensity.get('intensity_variability', 0) * 0.3 +
                pitch.get('pitch_jitter', 0) * 0.3
            )

            # Confidence indicators
            indicators['confidence_indicator'] = (
                pitch.get('pitch_mean', 0) * 0.01 +  # Higher pitch often indicates confidence
                intensity.get('intensity_mean', 0) * 100 +  # Stronger voice
                (1 - temporal.get('pause_ratio', 0)) * 0.5  # Fewer pauses
            )

            # Energy and excitement indicators
            indicators['energy_indicator'] = (
                intensity.get('intensity_mean', 0) * 50 +
                pitch.get('pitch_variability', 0) * 0.3 +
                temporal.get('zero_crossing_rate_mean', 0) * 1000
            )

            # Calmness indicators
            indicators['calmness_indicator'] = (
                (1 - pitch.get('pitch_variability', 0)) * 0.4 +
                (1 - intensity.get('intensity_variability', 0)) * 0.3 +
                temporal.get('pause_ratio', 0) * 0.3
            )

            # Speech quality indicators
            indicators['speech_clarity'] = (
                pitch.get('voiced_ratio', 0) * 0.7 +
                (1 - pitch.get('pitch_jitter', 0)) * 0.3
            )

        except Exception as e:
            logger.warning(f"Error extracting emotional indicators: {e}")
            indicators.update({
                'stress_indicator': 0.0,
                'confidence_indicator': 0.0,
                'energy_indicator': 0.0,
                'calmness_indicator': 0.0,
                'speech_clarity': 0.0
            })

        return indicators

    def _calculate_jitter(self, pitch_values: np.ndarray) -> float:
        """Calculate pitch jitter (cycle-to-cycle variation)"""
        if len(pitch_values) < 3:
            return 0.0

        try:
            # Calculate period differences
            periods = 1.0 / pitch_values
            period_diffs = np.abs(np.diff(periods))

            # Jitter as percentage of average period
            avg_period = np.mean(periods)
            jitter = np.mean(period_diffs) / avg_period if avg_period > 0 else 0.0

            return jitter

        except Exception:
            return 0.0

    def _detect_voice_activity(self, audio: np.ndarray) -> bool:
        """Detect if audio contains voice activity"""
        try:
            # Simple voice activity detection using energy
            rms = librosa.feature.rms(y=audio, frame_length=self.config.frame_length, hop_length=self.config.hop_length)
            rms_values = rms.flatten()

            # Check if average RMS is above threshold
            return np.mean(rms_values) > self.config.voice_activity_threshold

        except Exception:
            return False

    def _get_empty_features(self) -> VoiceFeatures:
        """Return empty features for error cases"""
        empty_dict = {
            'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
            'pitch_median': 0.0, 'pitch_slope': 0.0, 'pitch_variability': 0.0,
            'voiced_ratio': 0.0, 'pitch_jitter': 0.0,
            'intensity_mean': 0.0, 'intensity_std': 0.0, 'intensity_max': 0.0,
            'intensity_range': 0.0, 'intensity_slope': 0.0, 'intensity_variability': 0.0,
            'intensity_above_threshold': 0.0,
            'zero_crossing_rate_mean': 0.0, 'zero_crossing_rate_std': 0.0,
            'pause_ratio': 0.0, 'estimated_speech_rate': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
            'stress_indicator': 0.0, 'confidence_indicator': 0.0,
            'energy_indicator': 0.0, 'calmness_indicator': 0.0, 'speech_clarity': 0.0
        }

        return VoiceFeatures(
            pitch_features={k: v for k, v in empty_dict.items() if k.startswith('pitch') or k in ['voiced_ratio', 'pitch_jitter']},
            intensity_features={k: v for k, v in empty_dict.items() if k.startswith('intensity')},
            temporal_features={k: v for k, v in empty_dict.items() if k.startswith('zero') or k.startswith('pause') or k.startswith('estimated')},
            spectral_features={k: v for k, v in empty_dict.items() if k.startswith('spectral') or k.startswith('mfcc')},
            emotional_indicators={k: v for k, v in empty_dict.items() if k.endswith('indicator') or k == 'speech_clarity'},
            metadata={'error': 'Failed to extract features', 'analysis_timestamp': datetime.utcnow().isoformat()}
        )


class EmotionFromVoiceAnalyzer:
    """High-level analyzer that combines voice features with emotion detection"""

    def __init__(self, config: VoiceAnalysisConfig):
        self.voice_analyzer = VoiceAnalyzer(config)
        self.emotion_weights = {
            'stress_indicator': 0.25,
            'confidence_indicator': 0.20,
            'energy_indicator': 0.20,
            'calmness_indicator': 0.20,
            'speech_clarity': 0.15
        }

    def analyze_emotion_from_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze emotion from audio data

        Args:
            audio_data: Audio signal
            sample_rate: Sample rate of audio

        Returns:
            Dictionary with emotion analysis results
        """
        try:
            # Extract voice features
            features = self.voice_analyzer.analyze_audio_segment(audio_data, sample_rate)

            # Calculate emotion scores
            emotion_scores = self._calculate_emotion_scores(features.emotional_indicators)

            # Determine primary emotion
            primary_emotion = max(emotion_scores, key=emotion_scores.get)

            return {
                'primary_emotion': primary_emotion,
                'emotion_scores': emotion_scores,
                'voice_features': {
                    'pitch_mean': features.pitch_features.get('pitch_mean', 0),
                    'intensity_mean': features.intensity_features.get('intensity_mean', 0),
                    'speech_rate': features.temporal_features.get('estimated_speech_rate', 0),
                    'voiced_ratio': features.pitch_features.get('voiced_ratio', 0)
                },
                'confidence': emotion_scores[primary_emotion],
                'metadata': features.metadata
            }

        except Exception as e:
            logger.error(f"Error analyzing emotion from audio: {e}")
            return {
                'primary_emotion': 'neutral',
                'emotion_scores': {'neutral': 0.5, 'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 'fear': 0.1, 'surprise': 0.1, 'disgust': 0.0},
                'voice_features': {'pitch_mean': 0, 'intensity_mean': 0, 'speech_rate': 0, 'voiced_ratio': 0},
                'confidence': 0.5,
                'metadata': {'error': str(e), 'analysis_timestamp': datetime.utcnow().isoformat()}
            }

    def _calculate_emotion_scores(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate emotion scores from voice indicators"""
        scores = {}

        # Stress-based emotions
        stress_score = indicators.get('stress_indicator', 0)
        scores['angry'] = min(1.0, stress_score * 0.7)
        scores['fear'] = min(1.0, stress_score * 0.5)

        # Confidence-based emotions
        confidence_score = indicators.get('confidence_indicator', 0)
        scores['happy'] = min(1.0, confidence_score * 0.6)

        # Energy-based emotions
        energy_score = indicators.get('energy_indicator', 0)
        scores['surprise'] = min(1.0, energy_score * 0.5)
        scores['disgust'] = min(1.0, energy_score * 0.3)

        # Calmness-based emotions
        calmness_score = indicators.get('calmness_indicator', 0)
        scores['neutral'] = min(1.0, calmness_score * 0.8)
        scores['sad'] = min(1.0, calmness_score * 0.4)

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {emotion: score / total for emotion, score in scores.items()}

        return scores


# Example usage and demonstration
def demo_voice_analysis():
    """Demonstrate voice analysis functionality"""

    print("🎤 Voice Analysis Demo for EQM Emotion Detection")
    print("=" * 60)

    # Create sample audio data (simulated)
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    n_samples = int(sample_rate * duration)

    print(f"Generating sample audio: {duration}s at {sample_rate}Hz")

    # Simulate different emotional speech patterns
    emotions_to_test = ['neutral', 'happy', 'angry', 'sad', 'fearful']

    config = VoiceAnalysisConfig(sample_rate=sample_rate)
    analyzer = EmotionFromVoiceAnalyzer(config)

    for emotion in emotions_to_test:
        print(f"\n🎭 Testing emotion: {emotion.upper()}")

        # Generate emotion-specific audio patterns (simplified simulation)
        np.random.seed(hash(emotion) % 1000)  # Consistent seed per emotion

        # Base audio signal
        t = np.linspace(0, duration, n_samples)
        audio = np.random.normal(0, 0.1, n_samples)  # Background noise

        # Add emotion-specific patterns
        if emotion == 'happy':
            # Higher pitch, more energy, faster speech
            audio += 0.3 * np.sin(2 * np.pi * 300 * t)  # Higher frequency
            audio += 0.2 * np.random.normal(0, 0.1, n_samples)  # More variation
        elif emotion == 'angry':
            # Higher intensity, pitch variation, faster speech
            audio += 0.4 * np.sin(2 * np.pi * 250 * t)
            audio += 0.3 * np.random.normal(0, 0.15, n_samples)  # High variation
        elif emotion == 'sad':
            # Lower pitch, less intensity, slower patterns
            audio += 0.15 * np.sin(2 * np.pi * 180 * t)  # Lower frequency
            audio += 0.1 * np.random.normal(0, 0.05, n_samples)  # Low variation
        elif emotion == 'fearful':
            # High pitch, high intensity, tremor-like patterns
            audio += 0.35 * np.sin(2 * np.pi * 400 * t)  # Very high frequency
            audio += 0.25 * np.sin(2 * np.pi * 30 * t)  # Tremor frequency
            audio += 0.2 * np.random.normal(0, 0.12, n_samples)
        else:  # neutral
            # Moderate pitch and intensity
            audio += 0.2 * np.sin(2 * np.pi * 220 * t)
            audio += 0.15 * np.random.normal(0, 0.08, n_samples)

        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        print(".1f")
        print(f"  RMS Energy: {np.sqrt(np.mean(audio**2)):.4f}")
        print(f"  Zero Crossings: {np.sum(np.abs(np.diff(np.sign(audio)))) / 2}")

        # Analyze emotion from audio
        result = analyzer.analyze_emotion_from_audio(audio, sample_rate)

        print(f"  🎯 Detected Emotion: {result['primary_emotion']}")
        print(".2f")

        # Show top emotions
        sorted_emotions = sorted(result['emotion_scores'].items(), key=lambda x: x[1], reverse=True)
        print("  📊 Emotion Scores:")
        for emotion_name, score in sorted_emotions[:3]:
            print(" 5.1f")

        # Show voice features
        vf = result['voice_features']
        print("  🎵 Voice Features:")
        print(f"    Pitch: {vf['pitch_mean']:.1f} Hz")
        print(".3f")
        print(".1f")
        print(".1f")

    print("\n🎉 Voice analysis demo completed!")
    print("📝 Note: This demo uses simulated audio patterns.")
    print("   Real emotion detection would analyze actual speech recordings.")


if __name__ == "__main__":
    demo_voice_analysis()