################################
# media_miner-v0.1
################################
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QSpinBox, QComboBox, QFileDialog, QLabel, QGroupBox,
    QLineEdit, QTabWidget, QProgressBar, QRadioButton, QButtonGroup, QListWidget,
    QTextEdit, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Try importing ML libraries (graceful fallback if not available)
try:
    import torch
    from transformers import pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML libraries not available. NSFW detection will be disabled.")


class VideoProcessor(QThread):
    """Background thread for video processing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, video_path, output_dir, num_thumbnails=5, gif_duration=3):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.num_thumbnails = num_thumbnails
        self.gif_duration = gif_duration

    def run(self):
        try:
            thumbnails = self.extract_thumbnails()
            gifs = self.extract_gifs()
            self.finished.emit(thumbnails, gifs)
        except Exception as e:
            self.error.emit(str(e))

    def extract_thumbnails(self):
        """Extract evenly spaced thumbnails from video"""
        thumbnails = []
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame intervals
        interval = max(1, total_frames // self.num_thumbnails)

        for i in range(self.num_thumbnails):
            frame_pos = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save thumbnail
                timestamp = frame_pos / fps
                output_path = os.path.join(self.output_dir, f"thumbnail_{i+1}_{timestamp:.1f}s.png")
                cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                thumbnails.append(output_path)

            self.progress.emit(int((i + 1) / self.num_thumbnails * 50))

        cap.release()
        return thumbnails

    def extract_gifs(self):
        """Extract short GIF clips from video"""
        gifs = []
        cap = cv2.VideoCapture(self.video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create 3 GIFs at different positions
        positions = [0.25, 0.5, 0.75]
        gif_frames_count = int(fps * self.gif_duration)

        for idx, pos in enumerate(positions):
            start_frame = int(total_frames * pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(gif_frames_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize for smaller GIF file
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (320, 240))
                frames.append(frame_resized)

            if frames:
                output_path = os.path.join(self.output_dir, f"clip_{idx+1}_{pos:.0%}.gif")
                imageio.mimsave(output_path, frames, fps=fps//2, loop=0)
                gifs.append(output_path)

            self.progress.emit(50 + int((idx + 1) / len(positions) * 50))

        cap.release()
        return gifs


class ThumbnailGeneratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TMX MediaMiner - AI Thumbnail & GIF Generator")
        self.resize(1400, 800)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
            }
            QListWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px 20px;
                border: 1px solid #3d3d3d;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
        """)

        # Initialize variables
        self.input_file = None
        self.output_dir = None
        self.processor_thread = None
        self.thumbnails = []
        self.gifs = []

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Initialize all UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("TMX MediaMiner - AI Video Analysis Tool")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.setup_input_tab()
        self.setup_thumbnails_tab()
        self.setup_gifs_tab()
        self.setup_settings_tab()

    def setup_input_tab(self):
        """Setup input/output and processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input file section
        input_group = QGroupBox("Input Video/Image")
        input_layout = QVBoxLayout()

        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select video or image file...")
        self.input_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_input)

        input_row.addWidget(self.input_edit)
        input_row.addWidget(btn_browse)
        input_layout.addLayout(input_row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Output directory section
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout()

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output folder...")
        self.output_edit.setReadOnly(True)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self.browse_output)

        output_row.addWidget(self.output_edit)
        output_row.addWidget(btn_output)
        output_layout.addLayout(output_row)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        # Thumbnail count
        thumb_row = QHBoxLayout()
        thumb_row.addWidget(QLabel("Number of Thumbnails:"))
        self.thumb_spin = QSpinBox()
        self.thumb_spin.setRange(1, 20)
        self.thumb_spin.setValue(5)
        thumb_row.addWidget(self.thumb_spin)
        thumb_row.addStretch()
        options_layout.addLayout(thumb_row)

        # GIF duration
        gif_row = QHBoxLayout()
        gif_row.addWidget(QLabel("GIF Duration (seconds):"))
        self.gif_spin = QSpinBox()
        self.gif_spin.setRange(1, 10)
        self.gif_spin.setValue(3)
        gif_row.addWidget(self.gif_spin)
        gif_row.addStretch()
        options_layout.addLayout(gif_row)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        progress_layout.addWidget(self.status_text)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Process button
        self.btn_process = QPushButton("🚀 Process Media")
        self.btn_process.setStyleSheet("QPushButton { font-size: 16px; padding: 12px; }")
        self.btn_process.clicked.connect(self.process_media)
        layout.addWidget(self.btn_process)

        layout.addStretch()
        self.tabs.addTab(tab, "📁 Input & Process")

    def setup_thumbnails_tab(self):
        """Setup thumbnails display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Generated Thumbnails:"))

        self.thumb_list = QListWidget()
        self.thumb_list.itemClicked.connect(self.preview_thumbnail)
        layout.addWidget(self.thumb_list)

        self.thumb_preview = QLabel("Thumbnail preview will appear here")
        self.thumb_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_preview.setMinimumHeight(400)
        self.thumb_preview.setStyleSheet("border: 2px solid #3d3d3d; border-radius: 4px;")
        layout.addWidget(self.thumb_preview)

        self.tabs.addTab(tab, "🖼️ Thumbnails")

    def setup_gifs_tab(self):
        """Setup GIFs display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Generated GIF Clips:"))

        self.gif_list = QListWidget()
        self.gif_list.itemClicked.connect(self.preview_gif)
        layout.addWidget(self.gif_list)

        self.gif_preview = QLabel("GIF preview will appear here")
        self.gif_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_preview.setMinimumHeight(400)
        self.gif_preview.setStyleSheet("border: 2px solid #3d3d3d; border-radius: 4px;")
        layout.addWidget(self.gif_preview)

        self.tabs.addTab(tab, "🎬 GIF Clips")

    def setup_settings_tab(self):
        """Setup settings and info tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ML Status
        ml_group = QGroupBox("AI/ML Status")
        ml_layout = QVBoxLayout()

        ml_status = "✅ Available" if ML_AVAILABLE else "❌ Not Available"
        ml_layout.addWidget(QLabel(f"PyTorch & Transformers: {ml_status}"))

        if not ML_AVAILABLE:
            warning = QLabel("Install AI dependencies for NSFW detection:\npip install torch transformers")
            warning.setStyleSheet("color: #ff6b6b; padding: 10px;")
            ml_layout.addWidget(warning)

        ml_group.setLayout(ml_layout)
        layout.addWidget(ml_group)

        # Info section
        info_group = QGroupBox("About")
        info_layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>TMX MediaMiner</h3>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Extract thumbnails from videos</li>
            <li>Generate GIF clips automatically</li>
            <li>AI-powered content analysis (when ML libs available)</li>
            <li>Support for MP4, AVI, MOV video formats</li>
            <li>Support for PNG, JPG image formats</li>
        </ul>
        <p><b>Dependencies:</b> PyQt6, OpenCV, imageio, Pillow, PyTorch (optional)</p>
        """)
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.tabs.addTab(tab, "⚙️ Settings")

    def browse_input(self):
        """Browse for input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video or Image",
            "",
            "Media Files (*.mp4 *.avi *.mov *.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.input_file = file_path
            self.input_edit.setText(file_path)
            self.log_status(f"✅ Input file selected: {Path(file_path).name}")

    def browse_output(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if directory:
            self.output_dir = directory
            self.output_edit.setText(directory)
            self.log_status(f"✅ Output directory selected: {directory}")

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.append(message)

    def process_media(self):
        """Start media processing"""
        if not self.input_file:
            self.log_status("❌ Please select an input file first")
            return

        if not self.output_dir:
            self.log_status("❌ Please select an output directory first")
            return

        # Disable button during processing
        self.btn_process.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_status("\n🚀 Starting processing...")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Start processing thread
        self.processor_thread = VideoProcessor(
            self.input_file,
            self.output_dir,
            self.thumb_spin.value(),
            self.gif_spin.value()
        )
        self.processor_thread.progress.connect(self.update_progress)
        self.processor_thread.finished.connect(self.processing_complete)
        self.processor_thread.error.connect(self.processing_error)
        self.processor_thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def processing_complete(self, thumbnails, gifs):
        """Handle processing completion"""
        self.thumbnails = thumbnails
        self.gifs = gifs

        # Update thumbnail list
        self.thumb_list.clear()
        for thumb in thumbnails:
            self.thumb_list.addItem(Path(thumb).name)

        # Update GIF list
        self.gif_list.clear()
        for gif in gifs:
            self.gif_list.addItem(Path(gif).name)

        self.log_status(f"✅ Processing complete!")
        self.log_status(f"   - Generated {len(thumbnails)} thumbnails")
        self.log_status(f"   - Generated {len(gifs)} GIF clips")

        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(100)

        # Switch to thumbnails tab
        self.tabs.setCurrentIndex(1)

    def processing_error(self, error_msg):
        """Handle processing error"""
        self.log_status(f"❌ Error: {error_msg}")
        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(0)

    def preview_thumbnail(self, item):
        """Preview selected thumbnail"""
        idx = self.thumb_list.currentRow()
        if 0 <= idx < len(self.thumbnails):
            pixmap = QPixmap(self.thumbnails[idx])
            scaled = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.thumb_preview.setPixmap(scaled)

    def preview_gif(self, item):
        """Preview selected GIF (first frame)"""
        idx = self.gif_list.currentRow()
        if 0 <= idx < len(self.gifs):
            # Load first frame of GIF
            gif_path = self.gifs[idx]
            reader = imageio.get_reader(gif_path)
            first_frame = reader.get_data(0)

            # Convert to QPixmap
            height, width, channel = first_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(first_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.gif_preview.setPixmap(scaled)
            reader.close()


def main():
    app = QApplication(sys.argv)
    window = ThumbnailGeneratorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
