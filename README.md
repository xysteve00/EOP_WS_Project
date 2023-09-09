# OBFUSCATING TEXT RECOGNITION

 Modify the configurations in .json config files, get base model (eg. BASEMODEL_CHECKPOINT) ready then run:

1. Train base model and evaluation.

   a. Train
   <!-- ```shell
   python train.py -c configs/config.json
   ``` -->
   ```shell
   python run.py -m train -c experiments/Synth90k_MASTER_Cloak/config.json
   ```

   b. Eval
   ```shell
   python test.py --checkpoint BASEMODEL_CHECKPOINT \
   --img_folder TEST_IMG_PATH \
   --width 160 --height 48 \
   --output_folder OUTPUT_FOLDER \
   --batch_size 64
   ```

2. Train noise model from base neural network model.
```
python train_noise_model.py -c configs/config_1.json  -dist false  -m BASEMODEL_CHECKPOINT

```
