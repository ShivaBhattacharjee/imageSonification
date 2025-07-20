#!/usr/bin/env python3
"""
Command-line script to encode images to audio.
"""

import sys
import argparse
from image_encoder import ImageEncoder


def main():
    parser = argparse.ArgumentParser(description='Encode an image to audio waveform')
    parser.add_argument('input_image', help='Path to input image file')
    parser.add_argument('output_audio', help='Path to output audio file (.wav)')
    parser.add_argument('--method', choices=['frequency', 'amplitude', 'phase'], 
                       default='frequency', help='Encoding method (default: frequency)')
    parser.add_argument('--duration', type=float, default=10.0, 
                       help='Audio duration in seconds (default: 10.0)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Audio sample rate (default: 44100)')
    
    args = parser.parse_args()
    
    try:
        # Initialize encoder
        encoder = ImageEncoder(sample_rate=args.sample_rate, duration=args.duration)
        
        # Encode image
        metadata = encoder.encode_image_to_audio(
            args.input_image, 
            args.output_audio, 
            args.method
        )
        
        print("Encoding completed successfully!")
        print(f"Output audio: {args.output_audio}")
        print(f"Metadata: {args.output_audio.replace('.wav', '_metadata.json')}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
