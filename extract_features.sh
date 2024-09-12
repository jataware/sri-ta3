#!/bin/bash

# extract_features.sh

# conda create -y -n sri_env python=3.10
# conda activate sri_env
# pip install -e .

# --
# Set up paths

# SRI_DATA_ROOT="/home/paperspace/data/sri"
SRI_DATA_ROOT="/home/ubuntu/data/cmaas/sri_data/"

ln -s $SRI_DATA_ROOT/input_data/national_scale_raster_library/     $(pwd)/data/raster_libraries/
ln -s $SRI_DATA_ROOT/input_data/smidcont_mvt_raster_library/       $(pwd)/data/raster_libraries/
ln -s $SRI_DATA_ROOT/input_data/umidwest_mamanico_raster_library/  $(pwd)/data/raster_libraries/
ln -s $SRI_DATA_ROOT/input_data/ytu_tungsten-skarn_raster_library/ $(pwd)/data/raster_libraries/

ln -s $SRI_DATA_ROOT/models/* $(pwd)/sri_maper/ckpts/

# --

mkdir -p maps

BS=64

# national Lead-Zinc MVT
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=natl_mvt/exp_mvt_maevit_classifier_l22_uscont                \
    model.net.backbone_ckpt=sri_maper/ckpts/natl_pretrain.ckpt              \
    ckpt_path=sri_maper/ckpts/natl_mvt_mae.ckpt

mv gpu_0_result.feather maps/natl_mvt_mae.feather


# national Magmatic Nickel
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=natl_maniac/exp_maniac_maevit_classifier_l22_uscont          \
    model.net.backbone_ckpt=sri_maper/ckpts/natl_pretrain_maniac.ckpt       \
    ckpt_path=sri_maper/ckpts/natl_maniac_mae.ckpt

mv gpu_0_result.feather maps/natl_maniac_mae.feather


# national Tungsten-skarn
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=natl_w/exp_w_maevit_classifier_l22_uscont                    \
    model.net.backbone_ckpt=sri_maper/ckpts/natl_pretrain.ckpt              \
    ckpt_path=sri_maper/ckpts/natl_w_mae.ckpt

mv gpu_0_result.feather maps/natl_w_mae.feather


# national Porphyry Copper
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=natl_cu/exp_cu_maevit_classifier_l22_uscont                  \
    model.net.backbone_ckpt=sri_maper/ckpts/natl_pretrain_cu.ckpt           \
    ckpt_path=sri_maper/ckpts/natl_cu_mae.ckpt

mv gpu_0_result.feather maps/natl_cu_mae.feather


# regional Mafic Magmatic Nickel-Cobalt in Upper-Midwest
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=umidwest_mamanico/exp_mamanico_maevit_classifier_umidwest    \
    model.net.backbone_ckpt=sri_maper/ckpts/umidwest_mamanico_pretrain.ckpt \
    ckpt_path=sri_maper/ckpts/umidwest_mamanico_mae.ckpt

mv gpu_0_result.feather maps/umidwest_mamanico_mae.feather


# regional Tungsten-skarn in Yukon-Tanana Upland
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=ytu_w/exp_w_maevit_classifier_ytu                            \
    model.net.backbone_ckpt=sri_maper/ckpts/ytu_w_pretrain.ckpt             \
    ckpt_path=sri_maper/ckpts/ytu_w_mae.ckpt

mv gpu_0_result.feather maps/ytu_w_mae.feather


# regional MVT Lead-Zinc in "SMidCont"
python sri_maper/src/map.py trainer=gpu logger=csv data.batch_size=$BS enable_attributions=True \
    experiment=smidcont_mvt/exp_mvt_maevit_classifier_smidcont              \
    model.net.backbone_ckpt=sri_maper/ckpts/smidcont_mvt_pretrain.ckpt      \
    ckpt_path=sri_maper/ckpts/smidcont_mvt_mae.ckpt

mv gpu_0_result.feather maps/smidcont_mvt_mae.feather


# --
# Prep outputs, including:
# - deduplicate
# - fix types
# - merge SRI's Sharepoint Likelihood maps

python prep.py --target_idx 0
python prep.py --target_idx 1
python prep.py --target_idx 2
python prep.py --target_idx 3
python prep.py --target_idx 4
python prep.py --target_idx 5
python prep.py --target_idx 6

# --
# Run w/ random_splits, 

python random_splits.py --target_idx 0
python random_splits.py --target_idx 1
python random_splits.py --target_idx 2
python random_splits.py --target_idx 3
python random_splits.py --target_idx 4
python random_splits.py --target_idx 5
python random_splits.py --target_idx 6