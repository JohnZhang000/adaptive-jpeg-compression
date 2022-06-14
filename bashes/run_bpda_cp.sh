# clear
saved_dir='../saved_tests/BPDA'
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

model_type=(vgg16_imagenet)
# model_type=(allconv)
attackers=(bpda)
defenders=(GauA BDR RDG WEBPF_20 WEBPF_50 WEBPF_80 JPEG_20 JPEG_50 JPEG_80 SHIELD FD GD Ours Ours_WEBP)
# defenders=(WEBPF_80 GD Ours Ours_WEBP)
# defenders=(GD Ours_WEBP)
# defenders=(Ours_WEBP)
epsilons=(0.05)
epochs=(50)
lr=0.5
# lr=0.01

echo  'SUMMARY:bpda'                       |tee -a $log_name
echo  'model_type:       '${model_type[*]} |tee -a $log_name
echo  'attackers:        '${attackers[*]}  |tee -a $log_name
echo  'defenders:        '${defenders[*]}  |tee -a $log_name
echo  'epsilons:         '${epsilons[*]}   |tee -a $log_name
echo  'epochs:           '${epochs[*]}     |tee -a $log_name
echo  'lr:               '${lr}            |tee -a $log_name
echo  'log_name:         '${log_name}      |tee -a $log_name

####################################################################################
echo  ''                                                                                                 |tee -a $log_name
echo  '********************************* Starting *********************************'   |tee -a $log_name
echo  'start_time:       '$(date +%Y%m%d_%H%M%S)$a    |tee -a $log_name

for ((j=0;j<${#model_type[*]};j++))
do 
    model=${model_type[j]}
    for ((i=0;i<${#attackers[*]};i++))
    do 
        attacker=${attackers[i]}
        for ((k=0;k<${#epsilons[*]};k++))
        do 
            epsilon=${epsilons[k]}
            for ((e=0;e<${#epochs[*]};e++))
            do 
                epoch=${epochs[e]}
                for ((d=0;d<${#defenders[*]};d++))
                do 
                    defender=${defenders[d]}
                    echo  ''                                     |tee -a $log_name
                    echo  'attacker: '${attacker} 'model:'${model} 'epsilon: '${epsilon} 'epoch: '${epoch} 'defender: '${defender} |tee -a $log_name
                    python ../remove_code/BPDA_pytorch_cp1.py --model_data $model --attacker $attacker --defender $defender --epsilon $epsilon --epoch $epoch --lr $lr |tee -a $log_name
                done
            done
        done
    done
done