# Music-Source-Separation-master 
#### The scores here are either taken from their respective papers or from the evaluation scores of [SiSEC18](https://arxiv.org/pdf/1804.06267.pdf), and we show the median of SDR. It is worth mentioning that there is no extra data used in our training procedure. In order to make a fair comparison, we only compare with the methods without data augmentation.
|Models|Bass|Drums|Other|Vocals|AVG.|
|:--:|:--:|:--:|:--:|:--:|:--:|
|IRM oracle|7.12|8.45|7.85|9.43|8.21|
|Wave-U-Net [[paper](https://arxiv.org/pdf/1806.03185.pdf)] [[code](https://github.com/f90/Wave-U-Net-Pytorch)]|3.21|4.22|2.25|3.25|3.23|
|UMX [[paper](https://hal.inria.fr/hal-02293689/document)] [[code](https://github.com/sigsep/open-unmix-pytorch)]|5.23|5.73|4.02|6.32|5.33|
|Meta-TasNet [[paper](https://arxiv.org/pdf/2002.07016.pdf)] [[code](https://github.com/pfnet-research/meta-tasnet)]|5.58|5.91|4.19|6.40|5.52|
|MMDenseLSTM [[paper](https://arxiv.org/pdf/1805.02410.pdf)]|5.16|6.41|4.15|6.60|5.58|
|Sams-Net [[paper](https://arxiv.org/pdf/1909.05746.pdf)]|5.25|6.63|4.09|6.61|5.65|
|X-UMX [[paper](https://arxiv.org/pdf/2010.04228.pdf)] [[code](https://github.com/sony/ai-research-code/tree/master/x-umx)]|5.43|6.47|4.64|6.61|5.79|
|Conv-TasNet [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8707065)]|6.53|6.23|4.26|6.21|5.81|
|LaSAFT [[paper](https://arxiv.org/pdf/2010.11631.pdf)] [[code](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT)]|5.63|5.68|4.87|7.33|5.88|
|Spleeter [[paper](https://joss.theoj.org/papers/10.21105/joss.02154.pdf)] [[code](https://github.com/deezer/spleeter)]|5.51|6.71|4.02|6.86|5.91|
|D3Net [[paper](https://arxiv.org/pdf/2010.01733.pdf)]|5.25|7.01|4.53|7.24|6.01|
|DEMUCS [[paper](https://arxiv.org/pdf/1911.13254.pdf?ref=https://githubhelp.com)] [[code](https://github.com/facebookresearch/demucs)]|7.01|6.86|4.42|6.84|6.28|
|ours|7.92|7.33|4.92|7.37|6.89|

#### The code will be released soon...
