# tf_analyze

![LICENSE](https://img.shields.io/badge/license-MIT-green)
![Author](https://img.shields.io/badge/Author-TianyuCui-blue.svg)


**Analyze time domain, frequency domain and time-frequency domain data and Plot figures.**

## Features:
==========
- The frequency domain data is calculated by fft.
- The time-frequency domain data is calculated by fCWT (The fast Continuous Wavelet Transform)
- This code can analyze single signal or multiple signals.

|**analysis for a single signal**    |
|:--------------------------------------------------------------:|
|<img src="https://github.com/cuitianyu20/tf_analyze/blob/main/img/egg.png" alt="fcwtaudio" width="400"/>|
|**analysis for multiple signals**    |
|<img src="https://github.com/cuitianyu20/tf_analyze/blob/main/img/egg2.png" alt="fcwtaudio" width="400"/>|


***
## Dependencies
#### Test well:
- obspy 1.4.0 
- numpy 1.25.0
- fCWT  0.1.18
- matplotlib 3.7.1
***

## Reference
==========
[1] obspy.signal.tf_misfit (https://github.com/obspy/obspy/tree/master).Accessed: 2023-12-20.
[2] fCWT (https://github.com/fastlib/fCWT/tree/main).Accessed: 2023-12-20.

