#!/usr/bin/env python3
"""
Command-line script to decode audio back to images.
"""

import sys
import argparse
from audio_decoder import AudioDecoder


def main():
    parser = argparse.ArgumentParser(description='Decode audio waveform back to image')
    parser.add_argument('input_audio', help='Path to input audio file (.wav)')
    parser.add_argument('output_image', help='Path to output image file')
    
    args = parser.parse_args()
    
    try:
        # Initialize decoder
        decoder = AudioDecoder()
        
        # Decode audio
        result = decoder.decode_audio_to_image(
            args.input_audio, 
            args.output_image
        )
        
        print("Decoding completed successfully!")
        print(f"Output image: {args.output_image}")
        print(f"Original size: {result['original_size']}")
        print(f"Encoding method: {result['encoding_method']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
