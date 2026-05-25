import subprocess, sys, os
os.chdir(r'c:\Users\39527\Documents\Playground\服创赛')
r = subprocess.run([sys.executable, '-m', 'compileall', 'harness', 'sport', '-q'],
                   capture_output=True, text=True, timeout=60)
out = r'c:\Users\39527\Documents\Playground\服创赛\_r1.txt'
try:
    with open(out, 'w') as f:
        f.write(f'RETURN={r.returncode}\nSTDOUT={r.stdout}\nSTDERR={r.stderr}\n')
except Exception as e:
    # try alternate location
    with open(r'c:\Users\39527\Documents\_r1.txt', 'w') as f:
        f.write(f'WRITE_ERROR={e}\nRETURN={r.returncode}\nSTDOUT={r.stdout}\nSTDERR={r.stderr}\n')
