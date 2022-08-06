This is the code for our paper in Mathematics: [Collaborative Knowledge-Enhanced Recommendation with Self-Supervisions](https://www.mdpi.com/2227-7390/9/17/2129).

## Requirements

- Python 3.8
- PyTorch 1.8.0

## Quick Start
**Firstly**, generate the negative samples for training:

### amazon-book dataset
```bash
python generate_negs.py --dataset amazon-book --sample_num 200
```

### last-fm dataset
```bash
python generate_negs.py --dataset amazon-book --sample_num 100
```

**Secondly**, train the model and predict:

### amazon-book dataset
```bash
python main.py --demo 0 --dataset amazon-book --mode fuse --context_hops 1 --ssl 1 --alpha 0.005 --scale 12 --max_entity_num 16 --max_item_num 4
```

### last-fm dataset
```bash
python main.py --demo 0 --dataset last-fm --mode fuse --context_hops 1 --ssl 1 --alpha 0.5 --scale 10 --max_entity_num 16 --max_item_num 8
```
