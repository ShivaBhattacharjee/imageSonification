"""
Audio to Image Decoder Module

This module reconstructs images from audio waveforms that were encoded
using the ImageEncoder class. Metadata is now embedded directly in the audio.
"""

import numpy as np
import soundfile as sf
from PIL import Image
import scipy.signal as signal
from typing import Tuple, Optional


class AudioDecoder:
    """Decodes audio waveforms back into images."""
    
    def __init__(self):
        """Initialize the decoder."""
        pass
    
    def extract_metadata_from_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, dict]:
        """
        Extract embedded metadata from audio signal.
        
        Args:
            audio_data: Full audio data with embedded metadata header
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (clean_audio_data, metadata_dict)
        """
        # Header is first 1.0 seconds + 0.2 second gap
        header_duration = 1.0
        gap_duration = 0.2
        total_header_duration = header_duration + gap_duration
        header_samples = int(sample_rate * total_header_duration)
        
        # Extract header section
        header_section = audio_data[:int(sample_rate * header_duration)]
        
        # Check if we have enough data
        if len(header_section) == 0:
            raise ValueError("Audio file too short - no header found")
        
        # Extract clean audio (after header + gap)
        clean_audio = audio_data[header_samples:]
        
        # Check if we have remaining audio data
        if len(clean_audio) == 0:
            raise ValueError("Audio file too short - no data after header")
        
        # Analyze header frequencies using FFT
        fft = np.fft.fft(header_section)
        freqs = np.fft.fftfreq(len(header_section), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Check for valid magnitude data
        if np.max(magnitude) == 0:
            raise ValueError("No signal detected in audio header")
        
        # Find peaks in frequency domain
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Find frequency peaks with lower threshold for better detection
        peaks, _ = signal.find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.05)
        peak_freqs = positive_freqs[peaks]
        
        # Decode metadata from frequencies
        # Magic frequency should be around 1000 Hz
        magic_found = any(950 < f < 1050 for f in peak_freqs)
        if not magic_found:
            raise ValueError("Invalid audio format - metadata header not found")
        
        # Extract width (2000-2999 Hz range)
        width_freqs = [f for f in peak_freqs if 2000 <= f <= 2999]
        width_units = int(width_freqs[0] - 2000) if width_freqs else 256
        
        # Extract width thousands (4000-4450 Hz range)
        width_k_freqs = [f for f in peak_freqs if 4000 <= f <= 4450]
        width_thousands = int((width_k_freqs[0] - 4000) / 50) if width_k_freqs else 0
        
        width = width_units + (width_thousands * 1000)
        width = max(width, 1)  # Ensure minimum width of 1
        
        # Extract height (3000-3999 Hz range)  
        height_freqs = [f for f in peak_freqs if 3000 <= f <= 3999]
        height_units = int(height_freqs[0] - 3000) if height_freqs else 256
        
        # Extract height thousands (5000-5450 Hz range)
        height_k_freqs = [f for f in peak_freqs if 5000 <= f <= 5450]
        height_thousands = int((height_k_freqs[0] - 5000) / 50) if height_k_freqs else 0
        
        height = height_units + (height_thousands * 1000)
        height = max(height, 1)  # Ensure minimum height of 1
        
        # Extract method (6000+ Hz range)
        method_freqs = [f for f in peak_freqs if f >= 6000]
        if method_freqs:
            method_freq = method_freqs[0]
            if 5950 < method_freq < 6050:
                method = 'frequency'
            elif 6050 < method_freq < 6150:
                method = 'amplitude'
            elif 6150 < method_freq < 6250:
                method = 'phase'
            else:
                method = 'frequency'
        else:
            method = 'frequency'
        
        metadata = {
            'original_width': width,
            'original_height': height,
            'channels': 3,  # Assume RGB
            'encoding_method': method,
            'sample_rate': sample_rate
        }
        
        print(f"Detected peak frequencies: {[f'{f:.0f}Hz' for f in sorted(peak_freqs)]}")
        print(f"Width: {width_units} + {width_thousands}*1000 = {width}")
        print(f"Height: {height_units} + {height_thousands}*1000 = {height}")
        print(f"Extracted metadata from audio: {width}x{height}, method: {method}")
        
        return clean_audio, metadata

    def load_audio_and_metadata(self, audio_path: str) -> Tuple[np.ndarray, dict, int]:
        """
        Load audio file and extract embedded metadata.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, metadata, sample_rate)
        """
        # Load audio
        full_audio_data, sample_rate = sf.read(audio_path)
        
        # Extract metadata and clean audio
        audio_data, metadata = self.extract_metadata_from_audio(full_audio_data, sample_rate)
        
        return audio_data, metadata, sample_rate
    
    def frequency_mapping_decode(self, audio_data: np.ndarray, metadata: dict, 
                                sample_rate: int) -> np.ndarray:
        """
        Decode audio using frequency mapping method.
        
        Args:
            audio_data: Audio waveform data
            metadata: Encoding metadata
            sample_rate: Audio sample rate
            
        Returns:
            Reconstructed image array
        """
        width = metadata['original_width']
        height = metadata['original_height']
        
        # Safety checks
        if sample_rate <= 0:
            sample_rate = 44100  # Default sample rate
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Extract instantaneous frequency from audio
        analytic_signal = signal.hilbert(audio_data)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
        
        # Map frequencies back to pixel values
        min_freq = 20
        max_freq = sample_rate // 2
        
        # Ensure we don't have division by zero
        freq_range = max_freq - min_freq
        if freq_range <= 0:
            freq_range = 1
        
        # Normalize frequencies to 0-1 range
        normalized_freq = (instantaneous_frequency - min_freq) / freq_range
        normalized_freq = np.clip(normalized_freq, 0, 1)
        
        # Resize to original image dimensions
        total_pixels = width * height
        if total_pixels <= 0:
            total_pixels = 256 * 256  # Fallback size
        
        if len(normalized_freq) > total_pixels:
            # Downsample
            pixel_data = signal.resample(normalized_freq, total_pixels)
        else:
            # Upsample
            indices = np.linspace(0, len(normalized_freq) - 1, total_pixels)
            pixel_data = np.interp(indices, np.arange(len(normalized_freq)), normalized_freq)
        
        # Reshape to image dimensions
        grayscale_image = pixel_data.reshape(height, width)
        
        # Convert grayscale to RGB
        rgb_image = np.stack([grayscale_image, grayscale_image, grayscale_image], axis=2)
        
        # Convert to 0-255 range
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image
    
    def amplitude_modulation_decode(self, audio_data: np.ndarray, metadata: dict, 
                                  sample_rate: int) -> np.ndarray:
        """
        Decode audio using amplitude demodulation.
        
        Args:
            audio_data: Audio waveform data
            metadata: Encoding metadata
            sample_rate: Audio sample rate
            
        Returns:
            Reconstructed image array
        """
        width = metadata['original_width']
        height = metadata['original_height']
        total_pixels = width * height
        
        # Safety check for valid dimensions
        if total_pixels <= 0:
            width, height = 256, 256
            total_pixels = width * height
        
        # Carrier frequencies used in encoding
        carrier_freqs = [440, 880, 1320]
        
        # Generate time array
        if sample_rate <= 0:
            sample_rate = 44100  # Default sample rate
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        t = np.linspace(0, len(audio_data) / sample_rate, len(audio_data), endpoint=False)
        
        # Demodulate each channel
        channels = []
        
        for freq in carrier_freqs:
            # Create reference carrier
            carrier = np.sin(2 * np.pi * freq * t)
            
            # Demodulate by multiplying with carrier and low-pass filtering
            demodulated = audio_data * carrier
            
            # Low-pass filter to extract envelope
            nyquist = sample_rate / 2
            cutoff = freq / 10  # Cutoff frequency
            
            # Ensure cutoff is valid
            if nyquist <= 0:
                nyquist = 22050  # Default
            if cutoff <= 0:
                cutoff = 50  # Default cutoff
            if cutoff >= nyquist:
                cutoff = nyquist * 0.9  # Keep below nyquist
            
            sos = signal.butter(5, cutoff / nyquist, btype='low', output='sos')
            envelope = signal.sosfilt(sos, demodulated)
            
            # Take absolute value and smooth
            envelope = np.abs(envelope)
            
            # Resize to match pixel count
            if len(envelope) > total_pixels:
                channel_data = signal.resample(envelope, total_pixels)
            else:
                indices = np.linspace(0, len(envelope) - 1, total_pixels)
                channel_data = np.interp(indices, np.arange(len(envelope)), envelope)
            
            # Normalize to 0-1
            if np.max(channel_data) > 0:
                channel_data = channel_data / np.max(channel_data)
            
            channels.append(channel_data.reshape(height, width))
        
        # Combine channels into RGB image
        rgb_image = np.stack(channels, axis=2)
        
        # Convert to 0-255 range
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image
    
    def phase_encoding_decode(self, audio_data: np.ndarray, metadata: dict, 
                            sample_rate: int) -> np.ndarray:
        """
        Decode audio using phase demodulation.
        
        Args:
            audio_data: Audio waveform data
            metadata: Encoding metadata
            sample_rate: Audio sample rate
            
        Returns:
            Reconstructed image array
        """
        width = metadata['original_width']
        height = metadata['original_height']
        total_pixels = width * height
        
        # Safety check for valid dimensions
        if total_pixels <= 0:
            width, height = 256, 256
            total_pixels = width * height
        
        # Base frequencies used in encoding
        base_freq = 440
        freqs = [base_freq, base_freq * 1.5, base_freq * 2]
        
        # Generate time array
        if sample_rate <= 0:
            sample_rate = 44100  # Default sample rate
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        t = np.linspace(0, len(audio_data) / sample_rate, len(audio_data), endpoint=False)
        
        channels = []
        
        for freq in freqs:
            # Create reference signals
            cos_ref = np.cos(2 * np.pi * freq * t)
            sin_ref = np.sin(2 * np.pi * freq * t)
            
            # Demodulate phase
            i_component = audio_data * cos_ref
            q_component = audio_data * sin_ref
            
            # Low-pass filter
            nyquist = sample_rate / 2
            cutoff = freq / 4
            
            # Ensure cutoff is valid
            if nyquist <= 0:
                nyquist = 22050  # Default
            if cutoff <= 0:
                cutoff = 50  # Default cutoff
            if cutoff >= nyquist:
                cutoff = nyquist * 0.9  # Keep below nyquist
            
            sos = signal.butter(5, cutoff / nyquist, btype='low', output='sos')
            
            i_filtered = signal.sosfilt(sos, i_component)
            q_filtered = signal.sosfilt(sos, q_component)
            
            # Extract phase
            phase = np.arctan2(q_filtered, i_filtered)
            
            # Normalize phase to 0-1
            phase_normalized = (phase + np.pi) / (2 * np.pi)
            
            # Resize to match pixel count
            if len(phase_normalized) > total_pixels:
                channel_data = signal.resample(phase_normalized, total_pixels)
            else:
                indices = np.linspace(0, len(phase_normalized) - 1, total_pixels)
                channel_data = np.interp(indices, np.arange(len(phase_normalized)), phase_normalized)
            
            channels.append(channel_data.reshape(height, width))
        
        # Combine channels into RGB image
        rgb_image = np.stack(channels, axis=2)
        
        # Convert to 0-255 range
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image
    
    def decode_audio_to_image(self, audio_path: str, output_path: str) -> dict:
        """
        Main method to decode audio back to an image.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output image file
            
        Returns:
            Dictionary with decoding information
        """
        # Load audio and metadata
        audio_data, metadata, sample_rate = self.load_audio_and_metadata(audio_path)
        
        # Get encoding method from metadata
        method = metadata['encoding_method']
        
        # Choose decoding method
        if method == 'frequency':
            image_array = self.frequency_mapping_decode(audio_data, metadata, sample_rate)
        elif method == 'amplitude':
            image_array = self.amplitude_modulation_decode(audio_data, metadata, sample_rate)
        elif method == 'phase':
            image_array = self.phase_encoding_decode(audio_data, metadata, sample_rate)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Save reconstructed image
        image = Image.fromarray(image_array, 'RGB')
        image.save(output_path)
        
        # Return decoding information
        result = {
            'decoded_image_path': output_path,
            'original_size': (metadata['original_width'], metadata['original_height']),
            'reconstructed_size': image_array.shape[:2],
            'encoding_method': method,
            'audio_duration': len(audio_data) / sample_rate
        }
        
        print(f"Audio decoded to image: {output_path}")
        print(f"Original size: {metadata['original_width']}x{metadata['original_height']}")
        print(f"Decoding method: {method}")
        
        return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python audio_decoder.py <input_audio> <output_image>")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    output_image = sys.argv[2]
    
    decoder = AudioDecoder()
    decoder.decode_audio_to_image(input_audio, output_image)
