clear
saved_dir='../saved_tests/img_attack'
if [ ! -d "$saved_dir" ]; then
        mkdir $saved_dir
fi
log_name=$saved_dir'/terminal_whole_'$(date +%Y%m%d_%H%M%S)$a'.log'

host1="estar-403"
host2="Jet"
host3="ubuntu204"
host4="1080x4-1"
if [ $host1 == $HOSTNAME ]
then
	devices=0,1
elif [ $host2 == $HOSTNAME ]
then
	devices=0,1,2
elif [ $host3 == $HOSTNAME ]
then
	devices=0,1,2,3
elif [ $host4 == $HOSTNAME ]
then
	devices=3
else
	devices=0,1
fi
echo "Host:"$HOSTNAME"  Device:"$devices    |tee $log_name

model_type=(resnet50_imagenet)
attackers=(bpda)
defenders=(JPEG WEBPF AFO Ours)
epsilons=(0.05 1.0 1.5)

echo  'SUMMARY:bpda'                      |tee -a $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Starting *********************************'   |tee -a $log_name
echo  'start_time:       '$(date +%Y%m%d_%H%M%S)$a    |tee -a $log_name

for ((i=0;i<${#attackers[*]};i++))
do 
    attacker=${attackers[i]}
    for ((j=0;j<${#model_type[*]};j++))
    do 
        model=${model_type[j]}
        for ((k=0;k<${#epsilons[*]};k++))
        do 
            epsilon=${epsilons[k]}

            for ((d=0;d<${#defenders[*]};d++))
            do 
                defender=${defenders[d]}

                echo  ''                                     |tee -a $log_name
                echo  'attacker: '${attacker} 'model:'${model} 'epsilon: '${epsilon} 'defender: '${defender} |tee -a $log_name
                python ../remove_code/BPDA_pytorch.py --model_data $model --attacker $attacker --defender $defender --epsilon $epsilon  -a $log_name
            done
        done
    done
done