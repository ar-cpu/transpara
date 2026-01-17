import { Component, signal, ViewChild, ElementRef, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Configuration for the Python Backend URL
const BACKEND_URL = 'http://localhost:5000';

type InputMode = 'text' | 'audio-upload' | 'video-upload' | 'live-audio' | 'live-video' | 'pdf' | 'docx';

interface BackendResponse {
  success?: boolean;
  error?: string;
  prediction: string;
  confidence: number;
  probabilities: {
    left: number;
    center: number;
    right: number;
  };
  interpretation: string;
  transcription?: string;
  text?: string;
  too_short?: boolean;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow-y: auto;
    }
    /* Professional sleek scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: #f1f5f9; 
    }
    ::-webkit-scrollbar-thumb {
      background: #94a3b8; 
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #64748b; 
    }
  `]
})
export class AppComponent {
  // State
  currentTab = signal<InputMode>('text');
  isLoading = signal<boolean>(false);
  errorMessage = signal<string | null>(null);
  result = signal<BackendResponse | null>(null);
  textInput = signal<string>('');
  
  // UI State
  logoLoadError = signal<boolean>(false);
  
  // Recording State
  isRecording = signal<boolean>(false);
  recordingTime = signal<number>(0);
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private timerInterval: any;
  private stream: MediaStream | null = null;

  @ViewChild('videoPreview') videoPreview!: ElementRef<HTMLVideoElement>;
  
  // --- Computed Helpers ---
  recordingFormattedTime = computed(() => {
    const mins = Math.floor(this.recordingTime() / 60).toString().padStart(2, '0');
    const secs = (this.recordingTime() % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  });

  // --- Navigation ---
  setTab(tab: InputMode) {
    this.stopRecording(); // Safety cleanup
    this.currentTab.set(tab);
    this.errorMessage.set(null);
    this.result.set(null);
    
    // Auto-start camera if switching to live video
    if (tab === 'live-video') {
      this.initCamera();
    } else {
      this.stopCamera();
    }
  }
  
  handleLogoError() {
    this.logoLoadError.set(true);
  }

  // --- Backend Actions ---

  async analyzeText() {
    if (!this.textInput().trim()) {
      this.errorMessage.set('Please enter text to analyze.');
      return;
    }
    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.result.set(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze_text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: this.textInput() })
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Server error');
      this.result.set(data);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Connection to backend failed. Is app.py running?');
    } finally {
      this.isLoading.set(false);
    }
  }

  async handleFileUpload(event: Event, type: 'audio' | 'video' | 'pdf' | 'docx') {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    
    const file = input.files[0];
    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.result.set(null);

    const formData = new FormData();
    // Map the key to what app.py expects
    formData.append(type, file);

    let endpoint = '';
    switch (type) {
      case 'audio': endpoint = '/api/analyze_audio'; break;
      case 'video': endpoint = '/api/analyze_video'; break;
      case 'pdf': endpoint = '/api/analyze_pdf'; break;
      case 'docx': endpoint = '/api/analyze_docx'; break;
    }

    try {
      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Server error');
      this.result.set(data);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Analysis failed.');
    } finally {
      this.isLoading.set(false);
      input.value = ''; // Reset input
    }
  }

  // --- Live Recording Logic ---

  async toggleRecording(mode: 'audio' | 'video') {
    if (this.isRecording()) {
      this.stopRecording();
    } else {
      await this.startRecording(mode);
    }
  }

  async startRecording(mode: 'audio' | 'video') {
    try {
      this.recordedChunks = [];
      const constraints = mode === 'video' 
        ? { video: true, audio: true }
        : { audio: true };
      
      if (mode === 'video' && this.stream) {
        // reuse active stream
      } else {
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      }

      if (mode === 'video' && this.videoPreview) {
        this.videoPreview.nativeElement.srcObject = this.stream;
      }

      // Check supported mime types
      let mimeType = 'audio/webm';
      if (mode === 'video') {
        mimeType = 'video/webm; codecs=vp8';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'video/webm'; // Fallback
        }
      }

      this.mediaRecorder = new MediaRecorder(this.stream!, { mimeType });
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.processRecordedMedia(mode);
      };

      this.mediaRecorder.start();
      this.isRecording.set(true);
      this.startTimer();

    } catch (err) {
      console.error(err);
      this.errorMessage.set('Could not access microphone/camera.');
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording()) {
      this.mediaRecorder.stop();
      this.isRecording.set(false);
      this.stopTimer();
    }
  }

  private startTimer() {
    this.recordingTime.set(0);
    this.timerInterval = setInterval(() => {
      this.recordingTime.update(t => t + 1);
    }, 1000);
  }

  private stopTimer() {
    clearInterval(this.timerInterval);
    this.recordingTime.set(0);
  }

  async processRecordedMedia(mode: 'audio' | 'video') {
    this.isLoading.set(true);
    
    try {
      const mimeType = mode === 'video' ? 'video/webm' : 'audio/webm';
      const blob = new Blob(this.recordedChunks, { type: mimeType });
      const reader = new FileReader();

      reader.onloadend = async () => {
        const base64data = reader.result as string;
        const endpoint = mode === 'video' ? '/api/analyze_realtime_video' : '/api/analyze_realtime';
        const payload = mode === 'video' ? { video: base64data } : { audio: base64data };

        try {
          const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          const data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Server error');
          
          if (data.too_short) {
            this.errorMessage.set('Recording was too short to analyze. Please speak more.');
          } else {
            this.result.set(data);
          }
        } catch (err: any) {
          this.errorMessage.set(err.message);
        } finally {
          this.isLoading.set(false);
        }
      };
      
      reader.readAsDataURL(blob);

    } catch (err: any) {
      this.errorMessage.set('Processing failed: ' + err.message);
      this.isLoading.set(false);
    }
  }

  // --- Helper Methods ---

  private async initCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (this.videoPreview) {
        this.videoPreview.nativeElement.srcObject = this.stream;
      }
    } catch (e) {
      console.error('Camera init failed', e);
    }
  }

  private stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }
}
