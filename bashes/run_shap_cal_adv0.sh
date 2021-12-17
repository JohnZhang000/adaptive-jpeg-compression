clear
saved_dir='../saved_tests/img_shap'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_shap_'$(date +%Y%m%d_%H%M%S)$a'.log'

model_type=(allconv)
att_method=(FGSM_L2_IDP)
eps=(0.1)

echo  'SUMMARY:SHAP'                       |tee $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'att_method:       '${att_method[*]} |tee -a $log_name
echo  'eps:              '${eps[*]}        |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Calculating Shapley values *********************************'   |tee -a $log_name

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
            
            echo  ''                                     |tee -a $log_name
            echo  'model:            '${model}           |tee -a $log_name
            echo  'att_method:       '${att_method_now}  |tee -a $log_name
            echo  'eps_now:          '${eps_now}         |tee -a $log_name


            CUDA_VISIBLE_DEVICES=1 python ../remove_code/shap_cal_adv.py $model $att_method_now $eps_now   |tee -a $log_name
            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name

        done
    done
done
