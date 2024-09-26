@echo off
@REM python part2.py --image_folder "dataset\flickr30k_images" --label_file "dataset\labels.csv" --model_name "LogisticRegression" --test_size 0.2
python part2.py --image_folder "dataset\flickr30k_images" --label_file "dataset\labels.csv" --model_name "KNN" --test_size 0.2
@REM python part2.py --image_folder "dataset\flickr30k_images" --label_file "dataset\labels.csv" --model_name "DecisionTree" --test_size 0.2
pause
