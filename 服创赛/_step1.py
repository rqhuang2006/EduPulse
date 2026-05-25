import sys
sys.path.insert(0, ".")
try:
    from harness.contracts import ContractContext
    f = open("_step1.txt", "w")
    f.write("harness.contracts.ContractContext: OK\n")
    f.close()
except Exception as e:
    f = open("_step1.txt", "w")
    f.write(f"harness.contracts.ContractContext: FAIL - {e}\n")
    f.close()
