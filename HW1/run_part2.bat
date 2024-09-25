@echo off
python part2.py --image_folder "dataset\flickr30k_images" --label_file "dataset\labels.csv" --model_name "LogisticRegressionModel" --test_size 0.2
@REM python my_script.py --image_folder "HW1\dataset\flickr30k_images" --label_file "HW1\dataset\labels.csv" --model_name "KNNModel" --test_size 0.2
@REM python my_script.py --image_folder "HW1\dataset\flickr30k_images" --label_file "HW1\dataset\labels.csv" --model_name "DecisionTreeModel" --test_size 0.2
pause
