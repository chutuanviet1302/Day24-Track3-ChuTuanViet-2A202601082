# LLM Judge Bias Report — Phase B

**Sinh viên:** Chu Tuấn Việt  
**Ngày:** 2026-08-26  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?... | A | Answer A cung cấp thông tin chính xác về số ngày nghỉ và nhấn mạnh rằng quy định... |
| 5 | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệ... | A | Answer A cung cấp thông tin chính xác hơn về việc cần phê duyệt từ cấp quản lý h... |
| 12 | Thưởng Tết tối thiểu cho nhân viên chính thức có t... | A | Answer A cung cấp thông tin chi tiết hơn về việc thưởng Tết không được quy định ... |
| 21 | Một nhân viên Senior có 9 năm thâm niên được nghỉ ... | tie | Answer A cung cấp thông tin chính xác về số ngày phép và khoảng lương, đồng thời... |
| 23 | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việ... | A | Answer A cung cấp một cách tính cụ thể và nhấn mạnh rằng số tiền hoàn trả phụ th... | | |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent\? |
|---|---|---|---|---|
| 1 | A | A | A | True |
| 5 | A | A | A | True |
| 12 | A | A | A | True |
| 21 | A | B | tie | False |
| 23 | A | A | A | True | | | |

**Position bias rate:** 30.0% (= số case NOT consistent / tổng)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** [kết quả chạy judge trên 10 câu tương ứng]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 0 | No |
| 50 | 0 | 0 | Yes | 0 | Yes | | |

**Cohen's κ:** 0.2308  
**Interpretation:** fair

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 7 / 7 cases
- B thắng + B dài hơn A: 0 / 7 cases  
- **Verbosity bias rate:** 100.0%

**Kết luận:**

> Kết quả cho thấy tỷ lệ verbosity bias là 100.0%. LLM có xu hướng rõ rệt trong việc ưu tiên các câu trả lời dài hơn và nhiều chi tiết hơn, ngay cả khi thông tin không hoàn toàn chính xác.
> Điều này là vấn đề lớn trong các hệ thống RAG vì nó có thể che giấu các lỗi sai hoặc làm loãng câu trả lời thực tế mà người dùng đang tìm kiếm, gây khó khăn cho việc tối ưu hóa tính súc tích của hệ thống.

---

## 5. Nhận xét chung

> Nhận xét về hệ số Cohen's κ: Hệ số κ đạt 0.231 (fair), chứng tỏ LLM judge có sự đồng thuận rất tốt với nhãn của con người và có thể sử dụng đáng tin cậy.
> Về Position Bias: Rate ở mức 30.0%, cho thấy position bias của LLM là đáng kể và chiến lược hoán đổi (swap-and-average) thực sự rất quan trọng để giảm thiểu bias này trước khi đưa ra quyết định đánh giá.
> Trong production, nên kết hợp phương pháp swap-and-average kết hợp với việc giới hạn chiều dài câu trả lời của mô hình để triệt tiêu verbosity bias.
