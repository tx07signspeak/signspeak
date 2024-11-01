import {Component, OnInit, OnDestroy, Injectable} from '@angular/core';
import {Store} from '@ngxs/store';
import {VideoStateModel} from '../../../core/modules/ngxs/store/video/video.state';
import {InputMode, SignWritingObj} from '../../../modules/translate/translate.state';
import {
  CopySpokenLanguageText,
  SetSignWritingText,
  SetSpokenLanguageText,
} from '../../../modules/translate/translate.actions';
import {Observable} from 'rxjs';
import {io} from 'socket.io-client';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-signed-to-spoken',
  templateUrl: './signed-to-spoken.component.html',
  styleUrls: ['./signed-to-spoken.component.scss'],
})
@Injectable()
export class SignedToSpokenComponent implements OnInit, OnDestroy {
  videoState$!: Observable<VideoStateModel>;
  inputMode$!: Observable<InputMode>;
  spokenLanguage$!: Observable<string>;
  spokenLanguageText$!: Observable<string>;
  private socket: any;
  private lastText: string = '';
  private model: tf.GraphModel | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private isModelLoading = false;
  private modelLoaded: boolean = false;
  constructor(private store: Store) {
    this.videoState$ = this.store.select<VideoStateModel>(state => state.video);
    this.inputMode$ = this.store.select<InputMode>(state => state.translate.inputMode);
    this.spokenLanguage$ = this.store.select<string>(state => state.translate.spokenLanguage);
    this.spokenLanguageText$ = this.store.select<string>(state => state.translate.spokenLanguageText);
    this.store.dispatch(new SetSpokenLanguageText(''));

    // this.socket = io('http://localhost:1234', {
    //   reconnection: true, // Enable reconnection
    //   reconnectionAttempts: 5, // Number of attempts before giving up
    //   reconnectionDelay: 1000, // Initial delay between attempts (1 second)
    //   reconnectionDelayMax: 5000, // Maximum delay between attempts (5 seconds)
    //   randomizationFactor: 0.5, // Randomization factor for delay
    // });
    // this.socket.on('connect', () => {
    //   console.log('Socket.IO connection opened');
    //   this.socket.emit('message', 'Hello from client');
    // });
    // this.socket.on('disconnect', () => {
    //   console.log('Socket.IO connection closed');
    //   setTimeout(() => {
    //     this.socket.connect();
    //   }, 100); // Reconnect
    // });
    // this.socket.on('message', (data: string) => {
    //   if (data !== this.lastText) {
    //     this.store.dispatch(new SetSpokenLanguageText(data));
    //     this.lastText = data;
    //     console.log('message from server: ', data);
    //   }
    // });
  }

  ngOnInit() {
    // this.initializeTensorFlow();
    // this.loadModel();
    // const throttledProcessFrame = this.throttle(imageData => {
    //   this.processVideoFrame(imageData);
    // }, 100); // Adjust the delay (1000 ms = 1 second) as needed
    // const updateTranslation = () => {
    //   const video = document.querySelector('video');
    //   if (video) {
    //     const canvas = document.createElement('canvas');
    //     const context = canvas.getContext('2d');
    //     canvas.width = video.videoWidth;
    //     canvas.height = video.videoHeight;
    //     context.drawImage(video, 0, 0, canvas.width, canvas.height);
    //     if (canvas.width > 0 && canvas.height > 0) {
    //       const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    //       // Process the frame here
    //       throttledProcessFrame(imageData);
    //     }
    //   }
    //   requestAnimationFrame(updateTranslation);
    // };

    // updateTranslation();
    // Wait for metadata to load to ensure video dimensions are available
    // const videoElement = document.querySelector('video');
    // if (videoElement) {
    //   videoElement.addEventListener('loadedmetadata', updateTranslation);
    // }
  }
  private processVideoFrame(imageData: ImageData) {
    // This is a placeholder for the actual processing
    // console.log('Processing frame:', imageData.width, 'x', imageData.height);
    // Convert the ImageData to a TensorFlow tensor
    if (!this.modelLoaded) {
      console.error('Model not loaded yet!');
      return;
    }
    // 4. TODO - Make Detections
    const img = tf.browser.fromPixels(imageData);
    const resized = tf.image.resizeBilinear(img, [640, 480]);
    const casted = resized.cast('int32');
    const expanded = casted.expandDims(0);
    const obj = this.model.executeAsync(expanded);
    // Check if obj is an array and has at least 3 elements (index 2)
    if (Array.isArray(obj) && obj.length > 2) {
      const classes = obj[2].array();
      console.log(classes);
    } else {
      console.error('Unexpected model output format or missing index [2]');
    }
  }

  private throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
      const now = new Date().getTime();
      if (now - lastCall >= delay) {
        lastCall = now;
        func(...args);
      }
    };
  }

  // private async initializeTensorFlow() {
  //   console.log('Initializing TensorFlow.js...');
  //   await tf.ready();

  //   const backends = ['webgpu', 'webgl', 'cpu'];
  //   for (const backend of backends) {
  //     try {
  //       console.log(`Attempting to set backend to ${backend}`);
  //       await tf.setBackend(backend);
  //       await tf.ready();
  //       console.log(`Successfully set backend to ${backend}`);
  //       break;
  //     } catch (error) {
  //       console.warn(`Failed to set backend to ${backend}:`, error);
  //     }
  //   }

  //   console.log('Final backend:', tf.getBackend());
  // }

  private async loadModel() {
    if (this.model) return; // Model already loaded
    if (this.isModelLoading) return; // Model is currently loading

    this.isModelLoading = true;
    try {
      // Wait for TensorFlow.js to be ready
      await tf.ready();
      const modelUrl = 'assets/models/mymodels/model.json';
      this.model = await tf.loadGraphModel(modelUrl);
      this.modelLoaded = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load the model:', error);
    } finally {
      this.isModelLoading = false;
    }
  }

  ngOnDestroy(): void {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
    }
  }
  copyTranslation() {
    this.store.dispatch(CopySpokenLanguageText);
  }
}
