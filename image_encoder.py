"""
Image to Audio Encoder Module

This module converts images to audio waveforms using various encoding techniques
inspired by NASA's sonification methods. Metadata is now embedded directly in the audio.
"""

import numpy as np
import soundfile as sf
from PIL import Image
import scipy.signal as signal
from typing import Tuple, Optional


class ImageEncoder:
    """Encodes images into audio waveforms."""
    
    def __init__(self, sample_rate: int = 44100, duration: float = 10.0):
        """
        Initialize the encoder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            duration: Duration of output audio in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_frequency = sample_rate // 2  # Nyquist frequency
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and normalize image data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image array
        """
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array and normalize to 0-1
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        return img_array
    
    def frequency_mapping_encode(self, image_array: np.ndarray) -> np.ndarray:
        """
        Encode image using frequency mapping method.
        Maps pixel brightness to audio frequency.
        
        Args:
            image_array: Normalized image array
            
        Returns:
            Audio waveform as numpy array
        """
        height, width, channels = image_array.shape
        total_samples = int(self.sample_rate * self.duration)
        
        # Convert to grayscale for frequency mapping
        grayscale = np.mean(image_array, axis=2)
        
        # Flatten and resize to match audio duration
        flat_pixels = grayscale.flatten()
        
        # Interpolate to match sample count
        pixel_indices = np.linspace(0, len(flat_pixels) - 1, total_samples)
        interpolated_pixels = np.interp(pixel_indices, np.arange(len(flat_pixels)), flat_pixels)
        
        # Map pixel values to frequencies (20 Hz to max_frequency)
        min_freq = 20
        frequencies = min_freq + interpolated_pixels * (self.max_frequency - min_freq)
        
        # Generate time array
        t = np.linspace(0, self.duration, total_samples, endpoint=False)
        
        # Create audio waveform with varying frequency
        phase = np.cumsum(2 * np.pi * frequencies / self.sample_rate)
        audio = np.sin(phase)
        
        return audio
    
    def amplitude_modulation_encode(self, image_array: np.ndarray) -> np.ndarray:
        """
        Encode image using amplitude modulation.
        Uses pixel values for amplitude variation.
        
        Args:
            image_array: Normalized image array
            
        Returns:
            Audio waveform as numpy array
        """
        height, width, channels = image_array.shape
        total_samples = int(self.sample_rate * self.duration)
        
        # Use all three color channels
        red_channel = image_array[:, :, 0].flatten()
        green_channel = image_array[:, :, 1].flatten()
        blue_channel = image_array[:, :, 2].flatten()
        
        # Create three carrier frequencies for RGB
        carrier_freqs = [440, 880, 1320]  # A4, A5, E6
        
        # Generate time array
        t = np.linspace(0, self.duration, total_samples, endpoint=False)
        
        audio = np.zeros(total_samples)
        
        for i, (channel, freq) in enumerate(zip([red_channel, green_channel, blue_channel], carrier_freqs)):
            # Interpolate channel data to match audio length
            pixel_indices = np.linspace(0, len(channel) - 1, total_samples)
            interpolated_channel = np.interp(pixel_indices, np.arange(len(channel)), channel)
            
            # Create amplitude modulated signal
            carrier = np.sin(2 * np.pi * freq * t)
            modulated = carrier * interpolated_channel
            
            audio += modulated
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def phase_encoding_encode(self, image_array: np.ndarray) -> np.ndarray:
        """
        Encode image using phase encoding.
        Encodes color information in wave phase.
        
        Args:
            image_array: Normalized image array
            
        Returns:
            Audio waveform as numpy array
        """
        height, width, channels = image_array.shape
        total_samples = int(self.sample_rate * self.duration)
        
        # Base frequency
        base_freq = 440
        
        # Convert RGB to phase shifts
        red_channel = image_array[:, :, 0].flatten()
        green_channel = image_array[:, :, 1].flatten()
        blue_channel = image_array[:, :, 2].flatten()
        
        # Generate time array
        t = np.linspace(0, self.duration, total_samples, endpoint=False)
        
        # Interpolate color channels
        pixel_indices = np.linspace(0, len(red_channel) - 1, total_samples)
        
        red_interp = np.interp(pixel_indices, np.arange(len(red_channel)), red_channel)
        green_interp = np.interp(pixel_indices, np.arange(len(green_channel)), green_channel)
        blue_interp = np.interp(pixel_indices, np.arange(len(blue_channel)), blue_channel)
        
        # Create phase shifts from color values (0 to 2π)
        red_phase = red_interp * 2 * np.pi
        green_phase = green_interp * 2 * np.pi
        blue_phase = blue_interp * 2 * np.pi
        
        # Generate audio with phase encoding
        audio = (np.sin(2 * np.pi * base_freq * t + red_phase) +
                np.sin(2 * np.pi * base_freq * 1.5 * t + green_phase) +
                np.sin(2 * np.pi * base_freq * 2 * t + blue_phase)) / 3
        
        return audio
    
    def encode_image_to_audio(self, image_path: str, output_path: str, 
                            method: str = 'frequency') -> dict:
        """
        Main method to encode an image to audio with embedded metadata.
        
        Args:
            image_path: Path to input image
            output_path: Path for output audio file
            method: Encoding method ('frequency', 'amplitude', or 'phase')
            
        Returns:
            Dictionary with encoding metadata
        """
        # Load image
        image_array = self.load_image(image_path)
        height, width, channels = image_array.shape
        
        # Choose encoding method
        if method == 'frequency':
            audio = self.frequency_mapping_encode(image_array)
        elif method == 'amplitude':
            audio = self.amplitude_modulation_encode(image_array)
        elif method == 'phase':
            audio = self.phase_encoding_encode(image_array)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Embed metadata directly in the audio
        audio_with_metadata = self.embed_metadata_in_audio(audio, width, height, method)
        
        # Save audio file
        sf.write(output_path, audio_with_metadata, self.sample_rate)
        
        # Create metadata for return (but don't save separate file)
        metadata = {
            'original_width': width,
            'original_height': height,
            'channels': channels,
            'encoding_method': method,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'image_path': image_path
        }
        
        print(f"Image encoded to audio: {output_path}")
        print(f"Original image size: {width}x{height}")
        print(f"Encoding method: {method}")
        print(f"Metadata embedded in audio - no separate file needed!")
        
        return metadata
    
    def embed_metadata_in_audio(self, audio: np.ndarray, width: int, height: int, method: str) -> np.ndarray:
        """
        Embed metadata directly into the audio signal using a header.
        
        Args:
            audio: Original audio signal
            width: Image width
            height: Image height
            method: Encoding method
            
        Returns:
            Audio with embedded metadata header
        """
        # Create metadata header as audio tones
        # Use specific frequencies to encode the metadata
        header_duration = 1.0  # 1.0 seconds for metadata header
        header_samples = int(self.sample_rate * header_duration)
        t = np.linspace(0, header_duration, header_samples, endpoint=False)
        
        # Simple encoding scheme: use distinct frequencies
        magic_freq = 1000  # Magic number to identify format
        
        # Encode width: limit to 0-9999, use frequency offset
        width_clamped = min(width, 9999)
        width_freq = 2000 + (width_clamped % 1000)  # 2000-2999 Hz range
        
        # Encode height: limit to 0-9999, use frequency offset  
        height_clamped = min(height, 9999)
        height_freq = 3000 + (height_clamped % 1000)  # 3000-3999 Hz range
        
        # For larger dimensions, encode thousands separately
        width_thousands = width_clamped // 1000
        height_thousands = height_clamped // 1000
        
        width_k_freq = 4000 + (width_thousands * 50)  # 4000-4450 Hz
        height_k_freq = 5000 + (height_thousands * 50)  # 5000-5450 Hz
        
        # Encode method as frequency
        method_freqs = {'frequency': 6000, 'amplitude': 6100, 'phase': 6200}
        method_freq = method_freqs.get(method, 6000)
        
        # Create header tones with higher amplitude for better detection
        header = (np.sin(2 * np.pi * magic_freq * t) * 0.2 +
                 np.sin(2 * np.pi * width_freq * t) * 0.2 +
                 np.sin(2 * np.pi * height_freq * t) * 0.2 +
                 np.sin(2 * np.pi * width_k_freq * t) * 0.2 +
                 np.sin(2 * np.pi * height_k_freq * t) * 0.2 +
                 np.sin(2 * np.pi * method_freq * t) * 0.2)
        
        print(f"Encoding metadata: {width}x{height} -> freqs: W={width_freq:.0f}+{width_k_freq:.0f}, H={height_freq:.0f}+{height_k_freq:.0f}")
        
        # Add silence gap
        gap_samples = int(self.sample_rate * 0.2)  # 0.2 second gap
        gap = np.zeros(gap_samples)
        
        # Combine header + gap + original audio
        return np.concatenate([header, gap, audio])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python image_encoder.py <input_image> <output_audio>")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_audio = sys.argv[2]
    
    encoder = ImageEncoder()
    encoder.encode_image_to_audio(input_image, output_audio, method='frequency')
