CUDA_VISIBLE_DEVICES=3 python step0_get_clip_label.py

CUDA_VISIBLE_DEVICES=3 python step1-1_train_res50.py --stage "raw"
CUDA_VISIBLE_DEVICES=3 python step1-2_get_Layercam.py --stage "raw"
CUDA_VISIBLE_DEVICES=3 python step1-3_AMDA.py 
CUDA_VISIBLE_DEVICES=3 python step1-1_train_res50.py --stage "aug"
CUDA_VISIBLE_DEVICES=3 python step1-2_get_Layercam.py --stage "aug"

CUDA_VISIBLE_DEVICES=3 python step2_sam_inference.py

CUDA_VISIBLE_DEVICES=3 python step3-1_train_UNet3D.py --training_csv "../data/BraTS2020/splits/train.csv" --exp "SAM_Sup"
CUDA_VISIBLE_DEVICES=3 python step3-2_UNet_reget_pseg.py
CUDA_VISIBLE_DEVICES=3 python step3-3_S3F.py
CUDA_VISIBLE_DEVICES=3 python step3-1_train_UNet3D.py --training_csv "../data/BraTS2020/splits/top_80_percent.csv" --exp "UNet_Sup"