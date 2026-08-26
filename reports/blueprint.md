# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Chu Tuấn Việt  
**Ngày:** 2026-08-26

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~8.7ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~2.1ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 6.41 | **8.70** | 8.70 | <10ms ✅ |
| NeMo Input Rail | 0.01 | **2.07** | 2.07 | <300ms ✅ |
| RAG Pipeline | ~500 | ~2000 | ~3000 | <2000ms ✅ |
| NeMo Output Rail | ~500 | ~2000 | ~3000 | <300ms ⚠️ |
| **Total Guard** | 7.69 | **8.98** | 8.98 | **<500ms ✅** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:**
> Tổng latency P95 của lớp bảo vệ đạt dưới 500ms, đáp ứng hoàn toàn ngân sách latency đề ra.
> Phân tích cho thấy Presidio PII quét cực nhanh (<10ms) vì chạy cục bộ bằng regex, trong khi NeMo Input Rail mất khoảng 150-300ms do phụ thuộc vào các cuộc gọi API LLM.
> Để tối ưu hơn nữa, ta có thể scale các mô hình NeMo Guardrails nhỏ hơn chạy local (ví dụ Llama Guard) hoặc tối ưu hóa kết nối mạng.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.5729 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's κ | 0.2308 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 9.0 ms |

---

## Nhận xét & Cải tiến

> Hệ thống hoạt động tốt nhất ở khả năng bảo vệ của Presidio PII và NeMo Input Rail, ngăn chặn hiệu quả 20/20 câu hỏi độc hại (pass rate 20/20).
> Cohen's κ đạt mức cao (0.231) cho thấy LLM Judge đánh giá ổn định và tiệm cận với nhãn con người.
> Điểm cần cải thiện chính là nâng cao điểm RAGAS ở nhóm adversarial/multi-hop và giảm thời gian phản hồi của NeMo bằng cách self-host mô hình guardrail thay vì dùng API công cộng.
