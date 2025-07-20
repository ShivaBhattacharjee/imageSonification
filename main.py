"""
GUI Application for Image Sonification

A user-friendly interface for encoding images to audio and decoding audio back to images.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import soundfile as sf

from image_encoder import ImageEncoder
from audio_decoder import AudioDecoder


class SonificationGUI:
    """Main GUI application for image sonification."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Image Sonification App")
        self.root.geometry("800x700")
        
        # Initialize encoder and decoder
        self.encoder = ImageEncoder()
        self.decoder = AudioDecoder()
        
        # Variables
        self.current_image_path = None
        self.current_audio_path = None
        self.encoding_method = tk.StringVar(value="frequency")
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI layout."""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Encoder tab
        encoder_frame = ttk.Frame(notebook)
        notebook.add(encoder_frame, text="Image to Audio")
        self.setup_encoder_tab(encoder_frame)
        
        # Decoder tab
        decoder_frame = ttk.Frame(notebook)
        notebook.add(decoder_frame, text="Audio to Image")
        self.setup_decoder_tab(decoder_frame)
        
        # About tab
        about_frame = ttk.Frame(notebook)
        notebook.add(about_frame, text="About")
        self.setup_about_tab(about_frame)
    
    def setup_encoder_tab(self, parent):
        """Set up the image encoder tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="Select Image", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Browse Image", 
                  command=self.browse_image).pack(side=tk.LEFT)
        
        self.image_label = ttk.Label(file_frame, text="No image selected")
        self.image_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Preview frame
        preview_frame = ttk.LabelFrame(parent, text="Image Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.image_canvas = tk.Canvas(preview_frame, bg='white', height=200)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(parent, text="Encoding Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(settings_frame, text="Encoding Method:").grid(row=0, column=0, sticky=tk.W)
        
        method_frame = ttk.Frame(settings_frame)
        method_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Radiobutton(method_frame, text="Frequency Mapping", 
                       variable=self.encoding_method, value="frequency").pack(side=tk.LEFT)
        ttk.Radiobutton(method_frame, text="Amplitude Modulation", 
                       variable=self.encoding_method, value="amplitude").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(method_frame, text="Phase Encoding", 
                       variable=self.encoding_method, value="phase").pack(side=tk.LEFT, padx=(10, 0))
        
        # Duration setting
        ttk.Label(settings_frame, text="Audio Duration (seconds):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        self.duration_var = tk.DoubleVar(value=10.0)
        duration_spin = ttk.Spinbox(settings_frame, from_=1.0, to=60.0, increment=1.0, 
                                   textvariable=self.duration_var, width=10)
        duration_spin.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Encode button
        encode_frame = ttk.Frame(parent)
        encode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.encode_button = ttk.Button(encode_frame, text="Encode to Audio", 
                                       command=self.encode_image, state=tk.DISABLED)
        self.encode_button.pack(side=tk.LEFT)
        
        self.encode_progress = ttk.Progressbar(encode_frame, mode='indeterminate')
        self.encode_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
    
    def setup_decoder_tab(self, parent):
        """Set up the audio decoder tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="Select Audio", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Browse Audio", 
                  command=self.browse_audio).pack(side=tk.LEFT)
        
        self.audio_label = ttk.Label(file_frame, text="No audio selected")
        self.audio_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Audio visualization frame
        viz_frame = ttk.LabelFrame(parent, text="Audio Waveform", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib figure for waveform
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Decode button
        decode_frame = ttk.Frame(parent)
        decode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.decode_button = ttk.Button(decode_frame, text="Decode to Image", 
                                       command=self.decode_audio, state=tk.DISABLED)
        self.decode_button.pack(side=tk.LEFT)
        
        self.decode_progress = ttk.Progressbar(decode_frame, mode='indeterminate')
        self.decode_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
    
    def setup_about_tab(self, parent):
        """Set up the about tab."""
        about_text = """
Image Sonification Application

This application converts images to audio waveforms and back to images, 
inspired by NASA's sonification techniques.

Encoding Methods:

• Frequency Mapping: Maps pixel brightness to audio frequency
• Amplitude Modulation: Uses pixel values for amplitude variation  
• Phase Encoding: Encodes color information in wave phase

How to Use:

1. Image to Audio: Select an image, choose encoding method, and click "Encode to Audio"
2. Audio to Image: Select an encoded audio file and click "Decode to Image"

The application automatically saves metadata with encoded audio files 
to ensure accurate reconstruction.

Supported Formats:
• Images: PNG, JPEG, BMP, TIFF
• Audio: WAV files

Created for educational and research purposes.
        """
        
        text_widget = tk.Text(parent, wrap=tk.WORD, padx=20, pady=20)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def browse_image(self):
        """Browse for an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=os.path.basename(file_path))
            self.encode_button.config(state=tk.NORMAL)
            self.display_image_preview(file_path)
    
    def display_image_preview(self, image_path):
        """Display image preview in the canvas."""
        try:
            # Load and resize image for preview
            image = Image.open(image_path)
            
            # Calculate size to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet rendered, schedule for later
                self.root.after(100, lambda: self.display_image_preview(image_path))
                return
            
            # Resize image to fit canvas while maintaining aspect ratio
            image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                image=photo, anchor=tk.CENTER
            )
            
            # Keep a reference to prevent garbage collection
            self.image_canvas.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def browse_audio(self):
        """Browse for an audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[
                ("Audio files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_audio_path = file_path
            self.audio_label.config(text=os.path.basename(file_path))
            self.decode_button.config(state=tk.NORMAL)
            self.display_audio_waveform(file_path)
    
    def display_audio_waveform(self, audio_path):
        """Display audio waveform visualization."""
        try:
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Clear previous plot
            self.ax.clear()
            
            # Create time array
            time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
            
            # Plot waveform (downsample if too long for display)
            if len(audio_data) > 10000:
                step = len(audio_data) // 10000
                time = time[::step]
                audio_data = audio_data[::step]
            
            self.ax.plot(time, audio_data)
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Audio Waveform')
            self.ax.grid(True, alpha=0.3)
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display waveform: {str(e)}")
    
    def encode_image(self):
        """Encode the selected image to audio."""
        if not self.current_image_path:
            messagebox.showerror("Error", "No image selected")
            return
        
        # Get output file path
        output_path = filedialog.asksaveasfilename(
            title="Save Audio As",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Update encoder settings
        self.encoder.duration = self.duration_var.get()
        
        # Start encoding in separate thread
        self.encode_progress.start()
        self.encode_button.config(state=tk.DISABLED)
        
        def encode_thread():
            try:
                metadata = self.encoder.encode_image_to_audio(
                    self.current_image_path, 
                    output_path, 
                    self.encoding_method.get()
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.encode_complete(output_path))
                
            except Exception as e:
                self.root.after(0, lambda: self.encode_error(str(e)))
        
        threading.Thread(target=encode_thread, daemon=True).start()
    
    def encode_complete(self, output_path):
        """Handle successful encoding completion."""
        self.encode_progress.stop()
        self.encode_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"Image encoded to audio:\n{output_path}")
    
    def encode_error(self, error_message):
        """Handle encoding error."""
        self.encode_progress.stop()
        self.encode_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", f"Encoding failed:\n{error_message}")
    
    def decode_audio(self):
        """Decode the selected audio to image."""
        if not self.current_audio_path:
            messagebox.showerror("Error", "No audio selected")
            return
        
        # Get output file path
        output_path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return
        
        # Start decoding in separate thread
        self.decode_progress.start()
        self.decode_button.config(state=tk.DISABLED)
        
        def decode_thread():
            try:
                result = self.decoder.decode_audio_to_image(
                    self.current_audio_path, 
                    output_path
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.decode_complete(output_path))
                
            except Exception as e:
                self.root.after(0, lambda: self.decode_error(str(e)))
        
        threading.Thread(target=decode_thread, daemon=True).start()
    
    def decode_complete(self, output_path):
        """Handle successful decoding completion."""
        self.decode_progress.stop()
        self.decode_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"Audio decoded to image:\n{output_path}")
    
    def decode_error(self, error_message):
        """Handle decoding error."""
        self.decode_progress.stop()
        self.decode_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", f"Decoding failed:\n{error_message}")


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = SonificationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
