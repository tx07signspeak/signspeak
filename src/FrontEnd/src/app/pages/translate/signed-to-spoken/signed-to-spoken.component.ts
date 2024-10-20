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

const FAKE_WORDS = [
  {
    time: 0.618368,
    sw: ['M507x523S15a28494x496'],
    text: 'B',
  },
  {
    time: 0.876432,
    sw: ['M507x523S15a28494x496S26500493x477'],
    text: 'Your',
  },

  {
    time: 1.102468,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S115494'],
    text: 'Your h',
  },
  {
    time: 1.102468,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491'],
    text: 'Your h',
  },
  {
    time: 1.438297,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498'],
    text: 'Your',
  },
  {
    time: 1.628503,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498S20500489x476'],
    text: 'Your',
  },
  {
    time: 1.786967,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498S20600489x476'],
    text: 'Your name',
  },
  {
    time: 1.993408,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S14c39465x545S14c31508x546',
    ],
    text: 'Your name',
  },
  {
    time: 2.163386,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S30300481x477S14c31508x546S14c39465x545S26506539x545S26512445x545',
    ],
    text: 'Your name',
  },
  {
    time: 3.113322,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S30300481x477S14c31508x546S14c39465x545S27102539x545S27116445x545',
    ],
    text: 'What is your name?',
  },
];

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

  constructor(private store: Store) {
    this.videoState$ = this.store.select<VideoStateModel>(state => state.video);
    this.inputMode$ = this.store.select<InputMode>(state => state.translate.inputMode);
    this.spokenLanguage$ = this.store.select<string>(state => state.translate.spokenLanguage);
    this.spokenLanguageText$ = this.store.select<string>(state => state.translate.spokenLanguageText);
    this.store.dispatch(new SetSpokenLanguageText(''));
    this.socket = io('http://localhost:1234', {
      reconnection: true, // Enable reconnection
      reconnectionAttempts: 5, // Number of attempts before giving up
      reconnectionDelay: 1000, // Initial delay between attempts (1 second)
      reconnectionDelayMax: 5000, // Maximum delay between attempts (5 seconds)
      randomizationFactor: 0.5, // Randomization factor for delay
    });
    this.socket.on('connect', () => {
      console.log('Socket.IO connection opened');
      this.socket.emit('message', 'Hello from client');
    });
    this.socket.on('disconnect', () => {
      console.log('Socket.IO connection closed');
      setTimeout(() => {
        this.socket.connect();
      }, 100); // Reconnect
    });
    this.socket.on('message', (data: string) => {
      if (data !== this.lastText) {
        this.store.dispatch(new SetSpokenLanguageText(data));
        this.lastText = data;
        console.log('message from server: ', data);
      }
    });
  }

  ngOnInit(): void {
    // To get the fake translation
    let lastArray = [];
    let lastText = '';

    const f = () => {
      const video = document.querySelector('video');
      if (video) {
        let resultArray = [];
        let resultText = '';
        for (const step of FAKE_WORDS) {
          if (step.time <= video.currentTime) {
            resultText = step.text;
            resultArray = step.sw;
          }
        }

        if (resultText !== lastText) {
          this.store.dispatch(new SetSpokenLanguageText(resultText));
          lastText = resultText;
        }

        if (JSON.stringify(resultArray) !== JSON.stringify(lastArray)) {
          this.store.dispatch(new SetSignWritingText(resultArray));
          lastArray = resultArray;
        }
      }

      requestAnimationFrame(f);
    };
    f();
  }

  // async captureFrames(videoElement: HTMLVideoElement) {
  //   const canvas = document.createElement('canvas');
  //   const context = canvas.getContext('2d', {alpha: false});
  //   let lastFrameTime = 0;
  //   const targetFPS = 1; // Adjust this value to balance performance and quality

  //   canvas.width = videoElement.videoWidth;
  //   canvas.height = videoElement.videoHeight;

  //   const captureLoop = (timestamp: number) => {
  //     if (timestamp - lastFrameTime >= 1000 ) {
  //       if (this.socket.connected) {
  //         context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  //         const frameData = canvas.toDataURL('image/jpeg', 0.8); // Adjust quality as needed
  //         this.socket.emit('video-frame', frameData);
  //         lastFrameTime = timestamp;
  //       } else {
  //         console.log('Socket disconnected. Attempting to reconnect...');
  //         this.socket.connect();
  //       }
  //     }
  //     requestAnimationFrame(captureLoop);
  //   };

  //   requestAnimationFrame(captureLoop);
  // }
  ngOnDestroy(): void {
    // this.socket.disconnect();
    this.store.dispatch(new SetSpokenLanguageText('Disconnected from server'));
  }
  copyTranslation() {
    this.store.dispatch(CopySpokenLanguageText);
  }
}
