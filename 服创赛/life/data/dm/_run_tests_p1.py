import sys, os, traceback
sys.path.insert(0, r"c:\Users\39527\Documents\Playground\服创赛")
os.chdir(r"c:\Users\39527\Documents\Playground\服创赛")
result_path = r"c:\Users\39527\Documents\Playground\服创赛\life\data\dm\_test_results.txt"

def write_result(text):
    try:
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
    except:
        pass

write_result(f"Python: {sys.version}")
write_result(f"CWD: {os.getcwd()}")

# Sport stub test
try:
    from harness.domain_agents.sport import SportAgentStub
    sport = SportAgentStub()
    result = sport.run_domain_pipeline({})
    assert result["status"] == "stub"
    assert result["system_status"] == "stub"
    write_result("TEST sport_stub: PASS")
except Exception as e:
    write_result(f"TEST sport_stub: FAIL - {e}")

# Fusion valid
try:
    from harness.contracts.fusion_input import validate_fusion_input, FusionInputContract
    valid = validate_fusion_input(FusionInputContract(domain_name="life", risk_score=0.78, risk_level="medium"))
    assert valid["ok"] == True
    write_result("TEST fusion_valid: PASS")
except Exception as e:
    write_result(f"TEST fusion_valid: FAIL - {e}")

# Fusion invalid
try:
    invalid = validate_fusion_input({"domain_name": "", "risk_level": "weird"})
    assert invalid["ok"] == False
    write_result("TEST fusion_invalid: PASS")
except Exception as e:
    write_result(f"TEST fusion_invalid: FAIL - {e}")

write_result("DONE_PART1")
