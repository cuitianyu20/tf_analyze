# SeisDownload
![LICENSE](https://img.shields.io/badge/license-MIT-green)
![Author](https://img.shields.io/badge/Author-TianyuCui-blue.svg)


Analyze time domain, frequency domain and time-frequency domain data and Plot figures.

Notes:
1. The frequency domain data is calculated by fft.
2. The time-frequency domain data is calculated by fCWT (The fast Continuous Wavelet Transform)
3. This code can analyze single signal or multiple signals.

***
## Dependencies
#### Test well:
1. obspy 1.4.0 
2. numpy 1.25.0
3. fCWT  0.1.18
4. matplotlib 3.7.1
***
## Input Parameters
#### Details in codes
```Python
    arrayname: "IU" or "II" or "TA" or "TW" or "IC" or "IU,II,TA,TW,IC" or "*"
    station_name: "ANMO" or "TA01" or "ANMO,TA01" or "*"
    channel: channels (default: ["BHZ", "HHZ", "SHZ", "EHZ"])
    sta_range: 
        domain type:1 (RectangularDomain) sta_range = [sta_lat_min, sta_lat_max, sta_lon_min, sta_lon_max] in degree
                       if limit_distance=True, add distance restriction to the Rectangular domain
                      (RestrictionDomain) [min_dis, max_dis] in degree 
        domain type:2 (CircularDomain) sta_range = [minradius, maxradius] in degree  
                                       mid points: [ref_lat, ref_lon] in degree
        domain type:3 (GlobalDomain) []
    evt_range: [evt_lat_min, evt_lat_max, evt_lon_min, evt_lon_max] in degree (lon: 0 degree ~ 360 degree)
    evt_mag_range: [evt_mag_min, evt_mag_max]
    evt_min_dep: min event depth in km
    wave_len: downloaded waveform length in seconds
    startdate: earthquake catalog start date
    enddate: earthquake catalog end date
    limit_distance: if True, add distance restriction to the Rectangular domain (default: False)
                    min_dis: min distance in degree (default: 0)
                    max_dis: max distance in degree (default: 180)
    delete_mseed: if True, delete corresponding miniseed data if miniseed convert to sac successfully (default: True)
```
***
## Demo
##### Successfully Download One Earthquake Data

```
* Network: all.
* Statin: all.
* Channel: 'BHZ' # Z component
* Recorded waveform length: 1800s
* Event magnitude: greater than M5.
* Event depth: greater than 50 km.
* Time limitation: from 2015-01-01T00:00:00 to 2015-01-10T21:59:59.
* Domain: Rectangular domain and add distance restriction (0°-15°)
```