import sys
f = open(r"c:\Users\39527\Documents\Playground\服创赛\_pyinfo.txt", "w")
f.write(f"Python: {sys.version}\n")
f.write(f"CWD: {sys.getcwd() if hasattr(sys, 'getcwd') else 'N/A'}\n")
import os
f.write(f"CWD2: {os.getcwd()}\n")
f.close()
