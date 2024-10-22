import {Injectable} from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class NavigatorService {
  async getCamera(options: MediaTrackConstraints): Promise<MediaStream> {
    let camera: MediaStream;
    try {
      camera = await navigator.mediaDevices.getUserMedia({audio: false, video: options});
    } catch (e) {
      console.error(e.message);

      if (e.message.includes('Permission denied')) {
        throw new Error('permissionDenied');
      } else {
        throw new Error('Live Video Feed');
      }
    }
    if (!camera) {
      throw new Error('Live Video Feed');
    }
    return camera;
  }
}
