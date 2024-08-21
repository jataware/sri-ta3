

# pretrain for 77 layers
python sri_maper/src/pretrain.py experiment=natl_mvt/exp_maevit_pretrain_l22_uscont

# lead-zinc mvt
python sri_maper/src/train.py experiment=natl_mvt/exp_mvt_maevit_classifier_l22_uscont model.net.backbone_ckpt=logs/natl_mvt/exp_maevit_pretrain_l22_uscont/runs/checkpoints/last.ckpt

# tugsten-skarn
python sri_maper/src/train.py experiment=natl_w/exp_w_maevit_classifier_l22_uscont model.net.backbone_ckpt=logs/natl_mvt/exp_maevit_pretrain_l22_uscont/runs/checkpoints/last.ckpt



# pretrains the MAE checkpoint (14 evidence layers)
python sri_maper/src/pretrain.py experiment=natl_maniac/exp_maniac_maevit_pretrain_l22_uscont

# Magmatic Nickel CMA model
python sri_maper/src/train.py experiment=exp_maniac_maevit_classifier_l22_uscont model.net.backbone_ckpt=logs/natl_mvt/exp_maevit_pretrain_l22_uscont/runs/checkpoints/last.ckpt


# pretrains the MAE checkpoint (22 evidence layers
python sri_maper/src/pretrain.py experiment=natl_cu/exp_cu_maevit_pretrain_l22_uscont

# natl polyphory copper cma
python sri_maper/src/train.py experiment=exp_cu_maevit_classifier_l22_uscont model.net.backbone_ckpt=logs/natl_cu/exp_cu_maevit_pretrain_l22_uscont/runs/checkpoints/last.ckpt
