"""
Flask API for Image Sonification

A REST API that handles image to audio encoding and audio to image decoding.
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import io
import base64
import json
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from image_encoder import ImageEncoder
from audio_decoder import AudioDecoder

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'}
ALLOWED_AUDIO_EXTENSIONS = {'wav'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize encoder and decoder
encoder = ImageEncoder()
decoder = AudioDecoder()


def allowed_file(filename, extensions):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions


def generate_unique_filename(original_filename, extension=None):
    """Generate unique filename with UUID."""
    if extension is None:
        extension = original_filename.rsplit('.', 1)[1].lower()
    return f"{uuid.uuid4().hex}.{extension}"


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Image Sonification API is running'
    })


@app.route('/api/encode', methods=['POST'])
def encode_image():
    """
    Encode an image to audio.
    
    Expected form data:
    - image: Image file
    - method: Encoding method ('frequency', 'amplitude', 'phase')
    - duration: Audio duration in seconds (optional, default: 10)
    - sample_rate: Sample rate (optional, default: 44100)
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        # Get parameters
        method = request.form.get('method', 'frequency')
        duration = float(request.form.get('duration', 10.0))
        sample_rate = int(request.form.get('sample_rate', 44100))
        
        # Validate method
        if method not in ['frequency', 'amplitude', 'phase']:
            return jsonify({'error': 'Invalid encoding method'}), 400
        
        # Save uploaded image
        image_filename = generate_unique_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        file.save(image_path)
        
        # Generate output filenames
        audio_filename = generate_unique_filename(file.filename, 'wav')
        audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)
        
        # Update encoder settings
        encoder.duration = duration
        encoder.sample_rate = sample_rate
        
        # Encode image to audio
        metadata = encoder.encode_image_to_audio(image_path, audio_path, method)
        
        # Get image info for response
        image = Image.open(image_path)
        
        # Clean up uploaded image
        os.remove(image_path)
        
        return jsonify({
            'success': True,
            'audio_filename': audio_filename,
            'metadata': {
                'original_width': metadata['original_width'],
                'original_height': metadata['original_height'],
                'encoding_method': metadata['encoding_method'],
                'duration': metadata['duration'],
                'sample_rate': metadata['sample_rate']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/decode', methods=['POST'])
def decode_audio():
    """
    Decode audio back to image.
    
    Expected form data:
    - audio: Audio file (.wav) with embedded metadata
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': 'Invalid audio file type. Only WAV files are supported.'}), 400
        
        # Save uploaded audio
        audio_filename = generate_unique_filename(file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        file.save(audio_path)
        
        # Generate output filename
        image_filename = generate_unique_filename(file.filename, 'png')
        image_path = os.path.join(OUTPUT_FOLDER, image_filename)
        
        # Decode audio to image (metadata is embedded in audio)
        result = decoder.decode_audio_to_image(audio_path, image_path)
        
        # Convert image to base64 for preview
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'image_filename': image_filename,
            'image_data': f"data:image/png;base64,{img_data}",
            'result': {
                'original_size': result['original_size'],
                'reconstructed_size': result['reconstructed_size'],
                'encoding_method': result['encoding_method']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/audio/<filename>')
def download_audio(filename):
    """Download generated audio file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/image/<filename>')
def download_image(filename):
    """Download generated image file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/metadata/<audio_filename>')
def download_metadata(audio_filename):
    """Download metadata file for audio."""
    try:
        metadata_filename = audio_filename.replace('.wav', '_metadata.json')
        file_path = os.path.join(OUTPUT_FOLDER, metadata_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Metadata file not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=metadata_filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/methods', methods=['GET'])
def get_encoding_methods():
    """Get available encoding methods and their descriptions."""
    return jsonify({
        'methods': {
            'frequency': {
                'name': 'Frequency Mapping',
                'description': 'Maps pixel brightness to audio frequency. Bright pixels create higher frequencies.'
            },
            'amplitude': {
                'name': 'Amplitude Modulation',
                'description': 'Uses RGB color values for amplitude variation on different carrier frequencies.'
            },
            'phase': {
                'name': 'Phase Encoding',
                'description': 'Encodes color information in wave phase shifts.'
            }
        }
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Image Sonification API...")
    print("Access the web interface at: http://localhost:8080")
    print("API documentation available at: http://localhost:8080/api/health")

    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=8080)
