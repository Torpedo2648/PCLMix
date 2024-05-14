python train_contrast.py --exp "t01_contrast" --fold fold1 --contrast_weight 0.13 --het_weight 1.0 --unsup_m_weight 1.0 --tf_decoder_weight 0.1 --gpu 0
python train_contrast.py --exp "t01_contrast" --fold fold2 --contrast_weight 0.13 --het_weight 1.0 --unsup_m_weight 1.0 --tf_decoder_weight 0.1 --gpu 0
python train_contrast.py --exp "t01_contrast" --fold fold3 --contrast_weight 0.13 --het_weight 1.0 --unsup_m_weight 1.0 --tf_decoder_weight 0.1 --gpu 0
python train_contrast.py --exp "t01_contrast" --fold fold4 --contrast_weight 0.13 --het_weight 1.0 --unsup_m_weight 1.0 --tf_decoder_weight 0.1 --gpu 0
python train_contrast.py --exp "t01_contrast" --fold fold5 --contrast_weight 0.13 --het_weight 1.0 --unsup_m_weight 1.0 --tf_decoder_weight 0.1 --gpu 0
python test_cnn.py --exp "t01_contrast" --gpu 0