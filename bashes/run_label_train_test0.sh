clear
saved_dir='../saved_tests/img_attack'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_train_test_'$(date +%Y%m%d_%H%M%S)$a'.log'

model_type=(allconv)
device=0

echo  'SUMMARY:LABEL_TRAIN_TEST'           |tee $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'att_method:       '${att_method[*]} |tee -a $log_name
echo  'eps:              '${eps[*]}        |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

echo  ''                                                                                       |tee -a $log_name
echo  '********************************* LABEL_TRAIN_TEST *********************************'   |tee -a $log_name

for ((i=0;i<${#model_type[*]};i++))
do
    
    model_type_now=${model_type[i]}
            
    echo  ''                                  |tee -a $log_name                                 
    echo  'model_type:     '${model_type_now} |tee -a $log_name
       
    echo  ''                                  |tee -a $log_name
    echo  '[labeling train]:'                 |tee -a $log_name     
    CUDA_VISIBLE_DEVICES=$device python ../remove_code/my_spectrum_labeler.py $model_type_now 'train' |tee -a $log_name
    end_time=$(date +%Y%m%d_%H%M%S)$a
    echo  'end_time:       '${end_time}       |tee -a $log_name
    echo  '[labeling test]:'                  |tee -a $log_name     
    CUDA_VISIBLE_DEVICES=$device python ../remove_code/my_spectrum_labeler.py $model_type_now 'test' |tee -a $log_name
    end_time=$(date +%Y%m%d_%H%M%S)$a
    echo  'end_time:       '${end_time}       |tee -a $log_name

    echo  ''                                  |tee -a $log_name
    echo  '[Training]:'                       |tee -a $log_name     
    CUDA_VISIBLE_DEVICES=$device python ../remove_code/my_classifier.py $model_type_now               |tee -a $log_name
    end_time=$(date +%Y%m%d_%H%M%S)$a
    echo  'end_time:       '${end_time}       |tee -a $log_name

    echo  ''                                  |tee -a $log_name
    echo  '[Testing]:'                        |tee -a $log_name     
    CUDA_VISIBLE_DEVICES=$device python ../remove_code/my_proactiver_and_reactiver.py $model_type_now |tee -a $log_name
    end_time=$(date +%Y%m%d_%H%M%S)$a
    echo  'end_time:       '${end_time}       |tee -a $log_name

done
