clear
saved_dir='../saved_tests/img_thresh'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_thresh_'$(date +%Y%m%d_%H%M%S)$a'.log'

threshs=(0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.50)


echo  'SUMMARY:test'                       |tee $log_name
echo  'model_type:       '${threshs[*]} |tee -a $log_nam

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '******************************Test thresh *********************************'   |tee -a $log_name

for ((i=0;i<${#threshs[*]};i++))
do 

            thresh_now=${threshs[i]}
            
            echo  ''                                     |tee -a $log_name
            echo  'thresh_now:            '${thresh_now}           |tee -a $log_name


            CUDA_VISIBLE_DEVICES=0,1 python ../remove_code/Get_qtable.py $thresh_now   |tee -a $log_name
            sleep 3
            CUDA_VISIBLE_DEVICES=0,1 python ../remove_code/Attack_defence_pre_post.py   |tee -a $log_name

            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name


done
