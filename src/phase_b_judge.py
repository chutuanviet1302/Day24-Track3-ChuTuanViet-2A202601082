from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str,
                   ground_truth: str = "") -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    gt_block = f"\nGround Truth (dùng để đánh giá độ chính xác): {ground_truth}" if ground_truth else ""
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}{gt_block}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí (accuracy quan trọng nhất nếu có ground truth):
1. Độ chính xác (accuracy 40%): câu trả lời có đúng với ground truth/chính sách không?
2. Độ đầy đủ (completeness 35%): có trả lời đủ câu hỏi không?
3. Tính súc tích (conciseness 25%): có thừa / thiếu thông tin không?
Trả lời JSON (chỉ JSON, không văn bản nào khác ngoài JSON):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn lý do chọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    from openai import OpenAI
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Ưu tiên accuracy (khớp ground truth) trước khi đánh giá fluency. Chỉ trả lời JSON đúng định dạng yêu cầu."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, gt_block=gt_block, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  Pairwise judge failed: {e}")
        return {"winner": "tie", "reasoning": f"Error: {e}", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str,
                     ground_truth: str = "") -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    winner_pass1 = pass1.get("winner", "tie")
    if winner_pass1 == winner_pass2:
        final = winner_pass1
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (winner_pass1 == winner_pass2)
    scores_pass2_raw = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores_pass2 = {
        "A": scores_pass2_raw.get("B", 0.0),
        "B": scores_pass2_raw.get("A", 0.0)
    }

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=winner_pass1, winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""), reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {"A": 0.0, "B": 0.0}),
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    
    j_count_1 = judge_labels.count(1)
    j_count_0 = judge_labels.count(0)
    h_count_1 = human_labels.count(1)
    h_count_0 = human_labels.count(0)
    
    p_e = (j_count_1/n * h_count_1/n +
           j_count_0/n * h_count_0/n)
    
    κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0.0
    return float(κ)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0, "position_bias_rate": 0.0, "position_bias_count": 0,
            "verbosity_bias": 0.0, "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "Không có dữ liệu đánh giá."
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner in ("A", "B"))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_zero_shot_answer(question: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý HR. Hãy trả lời câu hỏi ngắn gọn nhất có thể dựa trên kiến thức chung (không sử dụng tài liệu nội bộ)."},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Zero-shot generation failed: {e}")
        return "Không tìm thấy thông tin chính xác."


if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge demo...")
    demo_result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {demo_result.winner_pass1}")
    print(f"  Pass 2 winner: {demo_result.winner_pass2}")
    print(f"  Final:         {demo_result.final_winner}")
    print(f"  Position consistent: {demo_result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # Run judge on the same 10 questions to get judge_labels
    # Build lookup: question_id -> ground_truth from answers_50q.json
    from config import ANSWERS_PATH
    gt_lookup = {}
    if os.path.exists(ANSWERS_PATH):
        with open(ANSWERS_PATH, encoding="utf-8") as f:
            all_answers = json.load(f)
        gt_lookup = {a["id"]: a.get("ground_truth", "") for a in all_answers}

    judge_results = []
    judge_labels = []
    for item in human_data:
        print(f"Evaluating Q{item['question_id']}: {item['question'][:50]}...")
        ans_a = generate_zero_shot_answer(item["question"])
        ans_b = item["model_answer"]
        gt = gt_lookup.get(item["question_id"], "")

        res = swap_and_average(item["question"], ans_a, ans_b, ground_truth=gt)
        judge_results.append(res)

        avg_score_b = (res.scores_pass1.get("B", 0.0) + res.scores_pass2.get("B", 0.0)) / 2.0
        label = 1 if (res.final_winner == "B" or avg_score_b >= 0.65) else 0
        judge_labels.append(label)

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report(judge_results)
    print(f"\nBias report: {bias}")

    # Save to reports/judge_results.json
    output_path = "reports/judge_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_data = {
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
        "results": [
            {
                "question_id": h["question_id"],
                "question": r.question,
                "answer_a": r.answer_a,
                "answer_b": r.answer_b,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "position_consistent": r.position_consistent,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
                "judge_label": l,
                "human_label": h["human_label"]
            }
            for r, l, h in zip(judge_results, judge_labels, human_data)
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Saved Phase B report → {output_path}")
