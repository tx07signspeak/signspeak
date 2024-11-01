import {Injectable} from '@angular/core';
import {Action, NgxsOnInit, State, StateContext, Store} from '@ngxs/store';
import {PoseService} from './pose.service';
import {LoadPoseEstimationModel, PoseVideoFrame, StoreFramePose} from './pose.actions';
import * as tf from '@tensorflow/tfjs';

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
  constructor(private poseService: PoseService, private store: Store) {}

  ngxsOnInit(): void {
    this.initializeTensorFlow();
    this.loadModel();
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
      const img = tf.browser.fromPixels(imageData);
      const resized = tf.image.resizeBilinear(img, [640, 480]);
      const casted = resized.cast('int32');
      const expanded = casted.expandDims(0);
      const obj = this.DetectModel.executeAsync(expanded);
      console.log(obj);
    });
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
