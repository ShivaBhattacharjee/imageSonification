// Global variables
let currentImageFile = null;
let currentAudioFile = null;
let lastEncodeResult = null;
let lastDecodeResult = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    setupDragAndDrop();
    updateDurationDisplay();
});

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab and mark button as active
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.closest('.tab-button').classList.add('active');
}

// Setup event listeners
function setupEventListeners() {
    // Duration slider
    const durationSlider = document.getElementById('duration');
    durationSlider.addEventListener('input', updateDurationDisplay);
    
    // File inputs
    document.getElementById('image-input').addEventListener('change', handleImageSelect);
    document.getElementById('audio-input').addEventListener('change', handleAudioSelect);
    
    // Upload areas click handlers
    document.getElementById('image-upload-area').addEventListener('click', () => {
        document.getElementById('image-input').click();
    });
    
    document.getElementById('audio-upload-area').addEventListener('click', () => {
        document.getElementById('audio-input').click();
    });
}

// Setup drag and drop
function setupDragAndDrop() {
    const uploadAreas = [
        { element: 'image-upload-area', handler: handleImageDrop },
        { element: 'audio-upload-area', handler: handleAudioDrop }
    ];
    
    uploadAreas.forEach(({ element, handler }) => {
        const area = document.getElementById(element);
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', () => {
            area.classList.remove('dragover');
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            handler(e.dataTransfer.files[0]);
        });
    });
}

// Update duration display
function updateDurationDisplay() {
    const slider = document.getElementById('duration');
    const display = document.getElementById('duration-value');
    display.textContent = slider.value;
}

// File handlers
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleImageDrop(file);
    }
}

function handleImageDrop(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file', 'error');
        return;
    }
    
    currentImageFile = file;
    displayImagePreview(file);
    document.getElementById('encode-btn').disabled = false;
    showToast('Image loaded successfully', 'success');
}

function handleAudioSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleAudioDrop(file);
    }
}

function handleAudioDrop(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
        showToast('Please select a WAV audio file', 'error');
        return;
    }
    
    currentAudioFile = file;
    displayAudioPreview(file);
    updateDecodeButton();
    showToast('Audio file loaded successfully', 'success');
}

// Display previews
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('image-preview');
        const img = document.getElementById('preview-img');
        const info = document.getElementById('image-info');
        
        img.src = e.target.result;
        info.textContent = `${file.name} (${formatFileSize(file.size)})`;
        preview.style.display = 'block';
        
        // Hide upload area
        document.getElementById('image-upload-area').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function displayAudioPreview(file) {
    const preview = document.getElementById('audio-preview');
    const info = document.getElementById('audio-info');
    
    info.textContent = `${file.name} (${formatFileSize(file.size)})`;
    preview.style.display = 'block';
    
    // Hide upload area
    document.getElementById('audio-upload-area').style.display = 'none';
    
    // TODO: Draw waveform visualization
    drawWaveform(file);
}

function drawWaveform(file) {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 150;
    
    // Simple placeholder visualization
    ctx.fillStyle = '#667eea';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Audio Waveform Preview', canvas.width / 2, canvas.height / 2);
    ctx.fillText('(Visualization placeholder)', canvas.width / 2, canvas.height / 2 + 20);
}

// Clear functions
function clearImage() {
    currentImageFile = null;
    document.getElementById('image-preview').style.display = 'none';
    document.getElementById('image-upload-area').style.display = 'block';
    document.getElementById('image-input').value = '';
    document.getElementById('encode-btn').disabled = true;
    document.getElementById('encode-result').style.display = 'none';
}

function clearAudio() {
    currentAudioFile = null;
    document.getElementById('audio-preview').style.display = 'none';
    document.getElementById('audio-upload-area').style.display = 'block';
    document.getElementById('audio-input').value = '';
    updateDecodeButton();
    document.getElementById('decode-result').style.display = 'none';
}

function updateDecodeButton() {
    const decodeBtn = document.getElementById('decode-btn');
    decodeBtn.disabled = !currentAudioFile;
}

// Encoding function
async function encodeImage() {
    if (!currentImageFile) {
        showToast('No image selected', 'error');
        return;
    }
    
    const method = document.querySelector('input[name="method"]:checked').value;
    const duration = document.getElementById('duration').value;
    const sampleRate = document.getElementById('sample-rate').value;
    
    const formData = new FormData();
    formData.append('image', currentImageFile);
    formData.append('method', method);
    formData.append('duration', duration);
    formData.append('sample_rate', sampleRate);
    
    showLoading('Encoding image to audio...');
    
    try {
        const response = await fetch('/api/encode', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            lastEncodeResult = result;
            displayEncodeResult(result);
            showToast('Image encoded successfully!', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showToast(`Encoding failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Decoding function
async function decodeAudio() {
    if (!currentAudioFile) {
        showToast('No audio file selected', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('audio', currentAudioFile);
    
    showLoading('Decoding audio to image...');
    
    try {
        const response = await fetch('/api/decode', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            lastDecodeResult = result;
            displayDecodeResult(result);
            showToast('Audio decoded successfully!', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showToast(`Decoding failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Display results
function displayEncodeResult(result) {
    const resultDiv = document.getElementById('encode-result');
    const audio = document.getElementById('encoded-audio');
    const method = document.getElementById('result-method');
    const duration = document.getElementById('result-duration');
    const size = document.getElementById('result-size');
    
    audio.src = `/api/download/audio/${result.audio_filename}`;
    method.textContent = getMethodName(result.metadata.encoding_method);
    duration.textContent = result.metadata.duration;
    size.textContent = `${result.metadata.original_width}×${result.metadata.original_height}`;
    
    resultDiv.style.display = 'block';
}

function displayDecodeResult(result) {
    const resultDiv = document.getElementById('decode-result');
    const img = document.getElementById('decoded-image');
    const method = document.getElementById('decode-method');
    const originalSize = document.getElementById('decode-original-size');
    const reconstructedSize = document.getElementById('decode-reconstructed-size');
    
    img.src = result.image_data;
    method.textContent = getMethodName(result.result.encoding_method);
    originalSize.textContent = `${result.result.original_size[0]}×${result.result.original_size[1]}`;
    reconstructedSize.textContent = `${result.result.reconstructed_size[0]}×${result.result.reconstructed_size[1]}`;
    
    resultDiv.style.display = 'block';
}

// Download functions
function downloadAudio() {
    if (lastEncodeResult) {
        window.open(`/api/download/audio/${lastEncodeResult.audio_filename}`, '_blank');
    }
}

function downloadImage() {
    if (lastDecodeResult) {
        window.open(`/api/download/image/${lastDecodeResult.image_filename}`, '_blank');
    }
}

// Utility functions
function getMethodName(method) {
    const names = {
        'frequency': 'Frequency Mapping',
        'amplitude': 'Amplitude Modulation',
        'phase': 'Phase Encoding'
    };
    return names[method] || method;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showLoading(text) {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = text;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
    
    // Remove on click
    toast.addEventListener('click', () => {
        toast.remove();
    });
}

function getToastIcon(type) {
    // No longer needed since we removed icons
    return '';
}
