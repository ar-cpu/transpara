import { Component, signal, ViewChild, ElementRef, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from './auth.service';
import { environment } from '../environments/environment';

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
  is_extremist?: boolean;
  extremist_content?: Array<{
    text: string;
    category: string;
    pattern_matched: string;
    start_pos: number;
    end_pos: number;
  }>;
  sentence_analysis?: Array<{
    text: string;
    prediction: string;
    confidence: number;
    start_pos: number;
    end_pos: number;
  }>;
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
    /* professional sleek scrollbar */
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
  authService = inject(AuthService);

  // state
  currentTab = signal<InputMode>('text');
  isLoading = signal<boolean>(false);
  errorMessage = signal<string | null>(null);
  result = signal<BackendResponse | null>(null);
  textInput = signal<string>('');

  // ui state
  logoLoadError = signal<boolean>(false);

  // recording state
  isRecording = signal<boolean>(false);
  recordingTime = signal<number>(0);
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private timerInterval: any;
  private stream: MediaStream | null = null;

  @ViewChild('videoPreview') videoPreview!: ElementRef<HTMLVideoElement>;

  // computed helpers
  recordingFormattedTime = computed(() => {
    const mins = Math.floor(this.recordingTime() / 60).toString().padStart(2, '0');
    const secs = (this.recordingTime() % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  });

  // navigation
  setTab(tab: InputMode) {
    this.stopRecording();
    this.currentTab.set(tab);
    this.errorMessage.set(null);
    this.result.set(null);

    if (tab === 'live-video') {
      this.initCamera();
    } else {
      this.stopCamera();
    }
  }

  handleLogoError() {
    this.logoLoadError.set(true);
  }

  // backend actions

  async analyzeText() {
    if (!this.textInput().trim()) {
      this.errorMessage.set('Please enter text to analyze.');
      return;
    }
    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.result.set(null);

    try {
      const response = await this.authService.authenticatedFetch('/analyze/text', {
        method: 'POST',
        body: JSON.stringify({ text: this.textInput() })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Server error');
      this.result.set(data);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Connection to backend failed.');
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
    formData.append(type, file);

    let endpoint = '';
    switch (type) {
      case 'audio': endpoint = '/analyze/audio'; break;
      case 'video': endpoint = '/analyze/video'; break;
      case 'pdf': endpoint = '/analyze/pdf'; break;
      case 'docx': endpoint = '/analyze/docx'; break;
    }

    try {
      const token = this.authService.getToken();
      // token check removed as it's always present now

      const response = await fetch(`${environment.apiUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          // Authorization not needed
        },
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Server error');
      this.result.set(data);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Analysis failed.');
    } finally {
      this.isLoading.set(false);
      input.value = '';
    }
  }

  // live recording logic

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

      let mimeType = 'audio/webm';
      if (mode === 'video') {
        mimeType = 'video/webm; codecs=vp8';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'video/webm';
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

      const formData = new FormData();
      formData.append('file', blob, mode === 'video' ? 'recording.webm' : 'recording.webm');

      const endpoint = mode === 'video' ? '/analyze/live-video' : '/analyze/live-audio';

      const response = await fetch(`${environment.apiUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          // Authorization not needed
        },
        body: formData
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
  }

  // helper methods

  getExtremistCategories(content: any[] | undefined): string {
    if (!content || content.length === 0) return '';
    const categories = [...new Set(content.map(item => item.category))];
    return categories.join(', ');
  }

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
