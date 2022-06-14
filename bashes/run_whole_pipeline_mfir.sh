clear
saved_dir='../saved_tests/img_attack'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_whole_'$(date +%Y%m%d_%H%M%S)$a'.log'

host1="estar-403"
host2="Jet"
host3="ubuntu204"
host4="QuadCopter"
if [ $host1 == $HOSTNAME ]
then
	devices=1
elif [ $host2 == $HOSTNAME ]
then
	devices=0,1,2
elif [ $host3 == $HOSTNAME ]
then
	devices=0,1,2,3
elif [ $host4 == $HOSTNAME ]
then
	devices=1,2,3
else
	devices=0,1
fi
echo "Host:"$HOSTNAME"  Device:"$devices    |tee $log_name

model_type=(allconv vgg16 resnet50 vgg16_imagenet resnet50_imagenet)
# model_type=(allconv)


echo  'SUMMARY:whole'                      |tee -a $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Starting *********************************'   |tee -a $log_name
echo  'start_time:       '$(date +%Y%m%d_%H%M%S)$a    |tee -a $log_name

for ((j=0;j<${#model_type[*]};j++))
do 
            model=${model_type[j]}
            
            echo  ''                                     |tee -a $log_name
            echo  'model:            '${model}           |tee -a $log_name

            echo  ''                                     |tee -a $log_name
            echo  'Defense:           '                  |tee -a $log_name            
            CUDA_VISIBLE_DEVICES=$devices nohup python ../remove_code/Attack_defence_pre_post_MFIR.py $model  |tee -a $log_name 
            echo  'end_time:       '$(date +%Y%m%d_%H%M%S)$a                    |tee -a $log_name

done