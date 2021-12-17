1 Configuration environment

conda env create -f environment.yml

2 Code for analyzing frequency bias
(The cifar image should be organized as an imagenet and its location specified in the .sh file)

cd bashes
bash run_shap_cal_adv0.sh
bash run_shap_cal_adv1.sh
bash run_shap_cal_adv2.sh

bash run_adv_spectrum_analysis_device0.sh
bash run_adv_spectrum_analysis_device1.sh
bash run_adv_spectrum_analysis_device2.sh


cd ../remove_code/
python shap_compare_adv.py dir model
python spectrum_compare_adv.py dir model

3 Code for defense
(location of cifar should be specified in the .py file)

python my_spectrum_label.py allconv train 0
python my_spectrum_label.py allconv test 0

python my_classifier.py

python my_proactive_and_reactive.py

python defence_pre_post.py
python defence_detect.py

(Due to time constraints, the code is further cleaned up, and the more user-friendly code will be exposed on github.)
