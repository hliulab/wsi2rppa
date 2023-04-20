Source code for
"Computational pathology infer clinically relevant protein levels and drug efficacy in breast cancer by weakly supervised contrastive learning"  

It is a multi-stage model. 
Firstly, [MoCo](https://arxiv.org/pdf/1911.05722.pdf) is used to unsupervised extract tile-level features, 
then the attention-pooling is used to aggregate tile-level features into slide-level features, 
and finally it is used in the downstream tasks,including tumor diagnosis, protein level prediction and drug response prediction, as well as a prognostic risk score.


# Now Updating

## Seg and Tile
You can download your own wsi dataset to the directory slides, 
then run create_patches_fp.py to seg and tile wsis, 
adjust the parameters according to your needs.  
For example, you can use following command for segment and tile.  

``` shell
python create_patches_fp.py --source ../slides/TCGA-BRCA  --patch_size 256 --save_dir ../tile_results --patch --seg --tcga_flag
```
When you run this command, it will run in default parameter, if you want to run with your parameter, you can modify tcga_lung.csv in directory preset, and add ```--preset ../preset/tcga_brca.csv```.
Then the coordinate files will be saved to ```tile_results/patches``` and the mask files that show contours of slides will be saved to ```tile_results/masks```.
Based on the previous step, you can randomly sample blocks for comparison learning training

## Train Contrast Learning Model
We use [Openmmlab](https://openmmlab.org.cn/) to train contrast learning model.

## Extract Tile-Level Features
Run extract_features_fp.py to extract the tile-level features.
For example, you can use following command for extracting features.  

``` shell
python extract_features_fp.py --data_h5_dir ../tile_results --data_slide_dir ../slides/TCGA-BRCA --csv_path ../dataset_csv/sample_data.csv --feat_dir ../FEATURES --data_type tcga_brca --model_path ../MODELS_SAVE/adco_tcga.pth.tar
```
The above command will use the trained MoCo model in ```model_path``` to extract tile features in ```data_slide_dir```
and save the features to ```feat_dir```. 

## Train Protein Level Model
Run train/train_protein_level.py to perform downstream classification task. For example:  
``` shell
python train_protein_level.py --feature_path ../FEATURES --train_csv_path xxx.csv --val_csv_path xxx.csv
```
The above command will use the feature file in ```data_root_dir``` to train the classification model, and then output the test results to ```results_dir```.
User needs to divide the data set into training set, verification set and test set in advance and put them under dataset_csv/tumor, such as:  

``` bash
dataset_csv/protein level
	     ├── train_dataset_1.csv
	     ├── ...
	     ├── train_dataset_3.csv
	     ├── val_dataset_1.csv
	     ├── ...
	     ├── val_dataset_3.csv
```
## Train Tumor Classification Model
Run train/train_tumor.py to perform downstream regression task. For example:  
``` shell
python train_tumor.py --lr 0.0003 --epochs 30 --wsi_path xxx --train_label_path xxx.csv --val_label_path xxx.csv
```
The above command will train regression that using attention-pooling to aggregate tile features by default. User should prepare gene dataset like this:  
``` bash
dataset_csv/tumor
	     ├── train_dataset_1.csv
	     ├── ...
	     ├── train_dataset_3.csv
	     ├── val_dataset_1.csv
	     ├── ...
	     ├── val_dataset_3.csv
```

