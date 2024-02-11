import os
#pip install snakeviz
os.system("python -m cProfile -o temp.dat main.py")
os.system("snakeviz temp.dat")