clear
saved_dir='../saved_tests/img_thresh'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_thresh_'$(date +%Y%m%d_%H%M%S)$a'.log'

thresh0=(0.1 0.3 0.5 0.7 0.9)
thresh1=(0.1 0.3 0.5 0.7 0.9)
thresh2=(0.1 0.3 0.5 0.7 0.9)

#thresh0=(0.1 0.9)
#thresh1=(0.1 0.9)
#thresh2=(0.1 0.9)

echo  'SUMMARY:test'                       |tee $log_name
echo  'thresh0:       '${thresh0[*]} |tee -a $log_name
echo  'thresh1:       '${thresh1[*]} |tee -a $log_name
echo  'thresh2:       '${thresh2[*]} |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '******************************Test thresh *********************************'   |tee -a $log_name

for ((i=0;i<${#thresh0[*]};i++))
do 
    for ((j=0;j<${#thresh1[*]};j++))
    do 
        for ((k=0;k<${#thresh2[*]};k++))
        do 

            thresh0_now=${thresh0[i]}
            thresh1_now=${thresh1[j]}
            thresh2_now=${thresh2[k]}
            
            echo  ''                                                            |tee -a $log_name
            echo  'thresh_now: '${thresh0_now}' '${thresh1_now}' '${thresh2_now}   |tee -a $log_name


            CUDA_VISIBLE_DEVICES=0,1 python ../remove_code/Get_qtable.py $thresh0_now $thresh1_now $thresh2_now   |tee -a $log_name
            sleep 3
            CUDA_VISIBLE_DEVICES=0,1 python ../remove_code/Attack_defence_pre_post.py   |tee -a $log_name

            end_time=$(date +%Y%m%d_%H%M%S)$a
            echo  'end_time:       '${end_time}    |tee -a $log_name
        done
    done
done
