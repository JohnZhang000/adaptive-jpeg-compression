clear
saved_dir='../saved_tests/img_attack'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_other_attack_'$(date +%Y%m%d_%H%M%S)$a'.log'

model_type=(resnet50_imagenet vgg16_imagenet)

echo  'SUMMARY:OTHER'                      |tee $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Calculating Shapley values *********************************'   |tee -a $log_name

for ((j=0;j<${#model_type[*]};j++))
do 
            model=${model_type[j]}
            
            echo  ''                                     |tee -a $log_name
            echo  'model:            '${model}           |tee -a $log_name


            CUDA_VISIBLE_DEVICES=0 python ../remove_code/Attack_defence_pre_post.py $model |tee -a $log_name
            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name

done
