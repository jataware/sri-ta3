### TRAINING
# python sri_maper/src/train.py experiment="exp_mvt/exp_mvt_resnet_l22_uscont.yaml"

#### TESTING
# python sri_maper/src/test.py +experiment=exp_mvt_resnet_l22+h2_all_us.yaml ckpt_path="logs/cmta3-mvt-h2-cUS-classifier-ResNet/runs/2024-02-13_18-19-49/checkpoints/epoch_007.ckpt"

##### PREDICTING
# python sri_maper/src/map.py +experiment="exp_mvt/exp_mvt_resnet_l22_uscont.yaml" data.batch_size=128 ckpt_path="logs/cmta3-classifier-mvt/runs/2024-04-02_22-40-37/checkpoints/auprc_0.968.ckpt"
