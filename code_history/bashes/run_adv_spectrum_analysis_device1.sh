clear

model_type=(resnet50_imagenet)
#att_method=(FGSM_L2_IDP PGD_L2_IDP FGSM_L2_UAP PGD_L2_UAP CW_L2_UAP Deepfool_L2_UAP)
att_method=(FGSM_Linf_IDP PGD_Linf_IDP CW_Linf_IDP)

eps=(0.005 0.01 0.1 1.0 10.0)

for ((i=0;i<${#model_type[*]};i++))
do
    for ((j=0;j<${#att_method[*]};j++))
    do
        for ((k=0;k<${#eps[*]};k++))
        do
            model_type_now=${model_type[i]}
            att_method_now=${att_method[j]}
            eps_now=${eps[k]}
            echo  ''                                 
            echo  'model_type:     '${model_type_now}
            echo  'att_method:     '${att_method_now}
            echo  'eps:            '${eps_now}
            CUDA_VISIBLE_DEVICES=1 python ../remove_code/Attack_compare_spectrum.py $model_type_now $att_method_now $eps_now 0

        done
    done
done
