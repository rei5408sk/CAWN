## NeRF<sup>2</sup>: Neural Radio-Frequency Radiance Fields
url(https://github.com/XPengZhao/NeRF2)

## Running

### Spectrum prediction

**training the model**

```bash
python nerf2_runner.py --mode train --config configs/rfid-spectrum.yml --dataset_type rfid --gpu 0
```

**Inference the model**

```bash
python nerf2_runner.py --mode test --config configs/rfid-spectrum.yml --dataset_type rfid --gpu 0
```

