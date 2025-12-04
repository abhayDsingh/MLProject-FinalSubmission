Step 1:
Extract the repo zip file or pull the repo on to your system.

Step 2:

run the following commands in terminal while in root folder:

(venv) abhaydeepsingh@Abhays-MacBook-Air MLProject % **python3 -m venv venv**

(venv) abhaydeepsingh@Abhays-MacBook-Air MLProject % **source venv/bin/activate**

(venv) abhaydeepsingh@Abhays-MacBook-Air MLProject % **pip install pandas numpy scikit-learn imbalanced-learn**

Step 3:

cd part1

python3 run_classification.py      (or python run_classification.py)

This will give you the classification text files for the 4 data sets and will also print out accuracy metrics in the terminal

Step 4:

cd ..

cd part2

python3 run_spam_detection.py      (or python run_spam_detection.py)

This will give you the resulting spam text file classifying into spam or ham and will also print out accuracy metrics in the terminal

That is all.

