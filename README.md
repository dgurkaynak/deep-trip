# deep-dream

Each frame is recursively fed into inception network starting with random noise.

Sample output: *(Be patient, it's ~10MB)*

![Sample output](https://github.com/dgurkaynak/deep-trip/raw/master/output_sample.gif)

### Requirements

 - Python 2.7 (not tested with > 3)
 - Tensorflow >= 1.0
 - Pillow

### Usage

```bash
./download_weights.sh
python trip.py
```
