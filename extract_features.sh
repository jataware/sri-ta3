python sri_maper/src/pretrain.py \
    experiment=natl_mvt/exp_maevit_pretrain_l22_uscont

python sri_maper/src/train.py \
    experiment=natl_mvt/exp_mvt_maevit_classifier_l22_uscont \
    model.net.backbone_ckpt=logs/maevit_pretrain_l22_uscont/checkpoints/last.ckpt

python sri_mapper/src/test.py

python sri_maper/src/map.py \
    experiment=natl_mvt/exp_maevit_pretrain_l22_uscont \
    data.batch_size=128 \
    enable_attributions=True \
    ckpt_path=logs/maevit_pretrain_l22_uscont/checkpoints/last.ckpt
    

# --

# conda activate sri_env

ln -s \
    /home/paperspace/data/sri/input_data/national_scale_raster_library/ \
    $(pwd)/data/raster_libraries/

ln -s \
    /home/paperspace/data/sri/models/* \
    $(pwd)/sri_maper/ckpts/

# resnet
# python sri_maper/src/map.py \
#     experiment=natl_mvt/exp_mvt_resnet_l22_uscont \
#     data.batch_size=128 \
#     enable_attributions=True \
#     ckpt_path=sri_maper/ckpts/natl_mvt_resnet.ckpt

# !! 
python sri_maper/src/map.py \
    experiment=exp_mvt_maevit_classifier_l22_uscont \
    model.net.backbone_ckpt=sri_maper/ckpts/natl_pretrain.ckpt \
    data.batch_size=64 \
    enable_attributions=True \
    ckpt_path=sri_maper/ckpts/natl_mvt_mae.ckpt



# --

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

  
# find ./ -type f | fgrep exp_maniac_maevit_classifier_l22_uscont
# find ./ -type f | fgrep exp_w_maevit_classifier_l22_uscont
# find ./ -type f | fgrep exp_cu_maevit_classifier_l22_uscont
# find ./ -type f | fgrep exp_mamanico_maevit_classifier_umidwest
# find ./ -type f | fgrep exp_w_maevit_classifier_ytu
# find ./ -type f | fgrep exp_mvt_maevit_classifier_smidcont