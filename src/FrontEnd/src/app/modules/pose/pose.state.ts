import {Injectable, ɵChangeDetectionScheduler} from '@angular/core';
import {Action, NgxsOnInit, State, StateContext, Store} from '@ngxs/store';
import {PoseService} from './pose.service';
import {LoadPoseEstimationModel, PoseVideoFrame, StoreFramePose} from './pose.actions';
import * as tf from '@tensorflow/tfjs';
import {
  SetSpokenLanguageText,
} from '../../modules/translate/translate.actions';
export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export const EMPTY_LANDMARK: PoseLandmark = {x: 0, y: 0, z: 0};

export interface EstimatedPose {
  faceLandmarks: PoseLandmark[];
  poseLandmarks: PoseLandmark[];
  rightHandLandmarks: PoseLandmark[];
  leftHandLandmarks: PoseLandmark[];
  image: HTMLCanvasElement;
}

export interface PoseStateModel {
  isLoaded: boolean;
  pose: EstimatedPose;
}

const initialState: PoseStateModel = {
  isLoaded: false,
  pose: null,
};
// Define our labelmap
const labelMap = {
    1:{name:'Hello', color:'red'},
    2:{name:'Thank You', color:'yellow'},
    3:{name:'I Love You', color:'lime'},
    4:{name:'Yes', color:'blue'},
    5:{name:'No', color:'purple'},
}

@Injectable()
@State<PoseStateModel>({
  name: 'pose',
  defaults: initialState,
})
export class PoseState implements NgxsOnInit {
  private DetectModel: tf.GraphModel | null = null;
  private isModelLoading = false;
  private modelLoaded: boolean = false;
  // loadPromise must be static, in case multiple PoseService instances are created (during testing)
  static loadPromise: Promise<any>;
  private lastText: string = '';
  constructor(private poseService: PoseService, private store: Store) {}

  async ngxsOnInit(): Promise<void> {
    await this.initializeTensorFlow();
    await this.loadModel();
    const throttledProcessFrame = this.throttle(imageData => {
      this.Detect(imageData);
    }, 16.7); // Adjust the delay (1000 ms = 1 second) as needed
    this.poseService.onResults(results => {
      // TODO: passing the `image` canvas through NGXS bugs the pose. (last verified 2024/02/28)
      // https://github.com/google/mediapipe/issues/2422
      const fakeImage = document.createElement('canvas');
      fakeImage.width = results.image.width;
      fakeImage.height = results.image.height;
      const ctx = fakeImage.getContext('2d');
      ctx.drawImage(results.image, 0, 0, fakeImage.width, fakeImage.height);

      // Since v0.4, "results" include additional parameters
      this.store.dispatch(
        new StoreFramePose({
          faceLandmarks: results.faceLandmarks,
          poseLandmarks: results.poseLandmarks,
          leftHandLandmarks: results.leftHandLandmarks,
          rightHandLandmarks: results.rightHandLandmarks,
          image: fakeImage,
        })
      );
      const imageData = ctx.getImageData(0, 0, fakeImage.width, fakeImage.height);
      throttledProcessFrame(imageData);
    });
  }
  private async Detect(imageData: ImageData) {
    const img = tf.browser.fromPixels(imageData);
    const resized = tf.image.resizeBilinear(img, [640, 480]);
    const casted = resized.cast('int32');
    const expanded = casted.expandDims(0);
    const obj = await this.DetectModel.executeAsync(expanded);
    const boxes = await obj[1].array(); // Bounding boxes
    const classes = await obj[2].array(); // Class labels
    const scores = await obj[4].array(); // Confidence scores
    for (let i = 0; i <= boxes[0].length; i++) {
      if (boxes[0][i] && classes[0][i] && scores[0][i] > 0.8) {
        const text = classes[0][i];
        const data = labelMap[text]['name'];
        console.log(data);
        if (data !== this.lastText) {
          this.store.dispatch(new SetSpokenLanguageText(data));
          this.lastText = data;
        }
      }
    }
    tf.dispose(img);
    tf.dispose(resized);
    tf.dispose(casted);
    tf.dispose(expanded);
    tf.dispose(obj);
  }
  private async initializeTensorFlow() {
    console.log('Initializing TensorFlow.js...');
    await tf.ready();

    const backends = ['webgpu', 'webgl', 'cpu'];
    for (const backend of backends) {
      try {
        console.log(`Attempting to set backend to ${backend}`);
        await tf.setBackend(backend);
        await tf.ready();
        console.log(`Successfully set backend to ${backend}`);
        break;
      } catch (error) {
        console.warn(`Failed to set backend to ${backend}:`, error);
      }
    }

    console.log('Final backend:', tf.getBackend());
  }

  private async loadModel() {
    if (this.DetectModel) return; // Model already loaded
    if (this.isModelLoading) return; // Model is currently loading

    this.isModelLoading = true;
    try {
      // Wait for TensorFlow.js to be ready
      await tf.ready();
      const modelUrl = 'assets/models/mymodels/model.json';
      this.DetectModel = await tf.loadGraphModel(modelUrl);
      this.modelLoaded = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load the model:', error);
    } finally {
      this.isModelLoading = false;
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

  @Action(LoadPoseEstimationModel)
  async loadPose(): Promise<void> {
    await this.poseService.load();
  }

  @Action(PoseVideoFrame)
  async poseFrame({patchState, dispatch}: StateContext<PoseStateModel>, {video}: PoseVideoFrame): Promise<void> {
    await this.poseService.predict(video);
  }

  @Action(StoreFramePose)
  storePose({getState, patchState}: StateContext<PoseStateModel>, {pose}: StoreFramePose): void {
    patchState({isLoaded: true, pose});
  }
}
