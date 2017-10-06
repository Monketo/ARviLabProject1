import os
import subprocess


testPath = '../datasets/classifieds/test'

with open('test.txt', 'w') as output:
    for imgPath in os.listdir(testPath):
        proc = subprocess.Popen(["python", "GoodsClassifier.py", imgPath, str(5)], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        output.write(imgPath + ' ' + out)
