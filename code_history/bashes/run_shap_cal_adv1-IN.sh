
clear
saved_dir_all='../saved_tests/'
saved_dir=$saved_dir_all$(date +%Y%m%d_%H%M%S)$a
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_shap.log'

model_type=(resnet50_imagenet vgg16_imagenet)
att_method=(FGSM_Linf_IDP PGD_Linf_IDP CW_Linf_IDP)
img_num=100
eps=(0.005 0.01 0.1 1.0 10.0)

echo  'SUMMARY:SHAP'                       |tee $log_name
echo  'saved_dir:        '${saved_dir}     |tee -a $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'att_method:       '${att_method[*]} |tee -a $log_name
echo  'img_num:          '${img_num}       |tee -a $log_name
echo  'eps:              '${eps[*]}        |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Calculating Shapley values *********************************'   |tee -a $log_name

dir_save_shap=$saved_dir'/shapleys'
if [ ! -d "$dir_save_shap" ]; then
        mkdir $dir_save_shap
fi
for ((i=0;i<${#att_method[*]};i++))
do 
    for ((j=0;j<${#model_type[*]};j++))
    do 
        for ((k=0;k<${#eps[*]};k++))
        do 
            model=${model_type[j]}
            att_method_now=${att_method[i]}
            dir_shap=$dir_save_shap
            eps_now=${eps[k]}
            if [ ! -d "$dir_shap" ]; then
                mkdir $dir_shap
            fi
            
            echo  ''                                  |tee -a $log_name
            echo  'model:            '${model}        |tee -a $log_name
            echo  'att_method:       '${att_method_now}  |tee -a $log_name
            echo  'img_num:          '${img_num}      |tee -a $log_name
            echo  'eps_now:          '${eps_now}      |tee -a $log_name
            echo  'dir_shap:         '${dir_shap}     |tee -a $log_name


            CUDA_VISIBLE_DEVICES=2 python ../remove_code/shap_cal_adv.py $model $img_num $dir_shap $att_method_now $eps_now   |tee -a $log_name
            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name

        done
    done
done
