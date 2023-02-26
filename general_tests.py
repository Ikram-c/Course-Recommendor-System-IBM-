import os
import glob


dir_path = os.path.dirname(os.path.realpath(__file__))





path = 'C:\Users\ikram\Documents\GitHub\Course-Recommendor-System-IBM-'
extension = 'csv'
os.chdir(dir_path)
result = glob.glob('*.{}'.format(extension))
print(result)