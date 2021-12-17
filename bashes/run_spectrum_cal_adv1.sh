clear
saved_dir='../saved_tests/img_attack'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_shap_'$(date +%Y%m%d_%H%M%S)$a'.log'

model_type=(resnet50_imagenet)
att_method=(CW_Linf_IDP)
eps=(0.005 0.01 0.1 1.0 10.0)


echo  'SUMMARY:SPECTRUM'                   |tee $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'att_method:       '${att_method[*]} |tee -a $log_name
echo  'eps:              '${eps[*]}        |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Calculating Shapley values *********************************'   |tee -a $log_name

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
            echo  'model_type:     '${model_type_now} |tee -a $log_name
            echo  'att_method:     '${att_method_now} |tee -a $log_name
            echo  'eps:            '${eps_now}        |tee -a $log_name
            CUDA_VISIBLE_DEVICES=3 python ../remove_code/Attack_compare_spectrum.py $model_type_now $att_method_now $eps_now |tee -a $log_name
            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name

        done
    done
done
