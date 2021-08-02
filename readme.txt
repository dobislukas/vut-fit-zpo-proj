Autor: Lukas Dobis 
Login: xdobis01
ZPO Projekt 2020/2021
Zadanie: Segmentace obrazu pomocí optického toku

Prekladac: gcc 10.2.0 (Arch User Repository)
Pouzite moduly: opencv 4.5.2 (https://github.com/opencv/opencv)

Zostavenie projektu:

```bash
$ cmake .
$ make
```
Priklady spustenia:

```bash
$ ./segmentVideo data/video.mp4
$ ./segmentVideo -f 0.6 data/video.mp4
$ ./segmentVideo -c 0.5 -b data/video.mp4
$ ./segmentVideo -s ./results/segmentation.avi -c 0.4 data/video.mp4
```

Na vypis priemerneho casu vypoctu 1 segmentacneho snimku sa odkomentuje riadok 19.
