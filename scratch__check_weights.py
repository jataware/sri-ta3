import torch
from rich import print

runs = [
    (
        # national Lead-Zinc MVT
        "sri_maper/ckpts/natl_pretrain.ckpt",
        "sri_maper/ckpts/natl_mvt_mae.ckpt",
    ),
    (
        # national Magmatic Nickel
        "sri_maper/ckpts/natl_pretrain_maniac.ckpt",
        "sri_maper/ckpts/natl_maniac_mae.ckpt",
    ),
    (
        # national Tungsten-skarn
        "sri_maper/ckpts/natl_pretrain.ckpt",
        "sri_maper/ckpts/natl_w_mae.ckpt",
    ),
    (
        # national Porphyry Copper
        "sri_maper/ckpts/natl_pretrain_cu.ckpt",
        "sri_maper/ckpts/natl_cu_mae.ckpt",
    ),
    (
        # regional Mafic Magmatic Nickel-Cobalt in Upper-Midwest
        "sri_maper/ckpts/umidwest_mamanico_pretrain.ckpt",
        "sri_maper/ckpts/umidwest_mamanico_mae.ckpt",
    ),
    (
        # regional Tungsten-skarn in Yukon-Tanana Upland
        "sri_maper/ckpts/ytu_w_pretrain.ckpt",
        "sri_maper/ckpts/ytu_w_mae.ckpt",
    ),
    (
        # regional MVT Lead-Zinc in "SMidCont"
        "sri_maper/ckpts/smidcont_mvt_pretrain.ckpt",
        "sri_maper/ckpts/smidcont_mvt_mae.ckpt",
    )
]



for pt_ckpt, ft_ckpt in runs:
    pt = torch.load(pt_ckpt)['state_dict']
    ft = torch.load(ft_ckpt)['state_dict']
    DIM = ft['net.backbone.patch_embedding.0.weight'].shape[1]
    n_correct, n_total = 0, 0
    for k_ft in ft.keys():
        if 'backbone' not in k_ft:
            continue
        
        if 'cls_token' in k_ft:
            continue
        
        
        k_pt = (
            k_ft
            .replace('net.backbone', 'net.encoder')
        )
        assert k_pt in pt.keys(), k_pt
        
        v_pt = pt[k_pt]
        v_ft = ft[k_ft]
        
        if (v_pt.shape == v_ft.shape) and (v_pt == v_ft).all():
            n_correct += 1
        
        n_total += 1
        
    print(f'{pt_ckpt=} {ft_ckpt=} {n_correct=} {n_total=} {DIM=}')

