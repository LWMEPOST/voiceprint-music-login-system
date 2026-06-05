import os
import sys

# 确保能找到 backend 模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.voiceprint import engine

THRESHOLD = 0.60

def _run_pair_case(case_name: str, expected_same_person: bool):
    reg_path = rf"d:\XM\LYX\backend\uploads\{case_name}_reg.webm"
    login_path = rf"d:\XM\LYX\backend\uploads\{case_name}_login.webm"
    reg_embedding = engine.extract_feature(reg_path)
    login_embedding = engine.extract_feature(login_path)
    similarity = engine.compare(reg_embedding, login_embedding)

    predict_same = similarity >= THRESHOLD
    passed = (predict_same == expected_same_person)
    relation = "同人" if expected_same_person else "异人"
    result = "PASS" if passed else "FAIL"
    print(f"{case_name} ({relation}) -> score={similarity:.4f}, threshold={THRESHOLD:.2f}, predict={'同人' if predict_same else '异人'} [{result}]")
    return similarity, passed

def test_user_audio():
    print("Running local voiceprint validation with test1/test2...\n")
    test_cases = [
        ("test1", False),  # 异人，应拒绝
        ("test2", True),   # 同人，应通过
    ]
    all_passed = True
    scores = {}
    for case_name, expected_same_person in test_cases:
        try:
            score, passed = _run_pair_case(case_name, expected_same_person)
            scores[case_name] = score
            all_passed = all_passed and passed
        except Exception as e:
            all_passed = False
            print(f"{case_name} -> ERROR: {e}")

    print("\n" + "=" * 70)
    print("Summary:")
    for case_name in ["test1", "test2"]:
        if case_name in scores:
            print(f"  - {case_name}: {scores[case_name]:.4f}")
    print(f"  - final_result: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

if __name__ == "__main__":
    test_user_audio()
