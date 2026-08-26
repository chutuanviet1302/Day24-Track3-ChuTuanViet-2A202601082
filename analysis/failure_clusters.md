# Failure Cluster Analysis — Phase A

**Sinh viên:** Chu Tuấn Việt  
**Ngày:** 2026-08-26

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | ? | ? | ? |
| answer_relevancy | ? | ? | ? |
| context_precision | ? | ? | ? |
| context_recall | ? | ? | ? |
| **avg_score** | ? | ? | ? |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai ... | 0.1854 | answer_relevancy |
| 2 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiê... | 0.1925 | answer_relevancy |
| 3 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng thá... | 0.2382 | answer_relevancy |
| 4 | adversarial | Khi phát hiện malware trên máy tính công ty, nhân viên có nê... | 0.3366 | context_precision |
| 5 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi ... | 0.3433 | context_precision |
| 6 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài ... | 0.3576 | context_precision |
| 7 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân v... | 0.4328 | context_precision |
| 8 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ... | 0.4654 | context_precision |
| 9 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con?... | 0.4682 | context_precision |
| 10 | multi_hop | Nhân viên có thâm niên 7 năm theo v2024 được nghỉ bao nhiêu ... | 0.4716 | context_precision | | | |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 0 | 3 | 0 | 3 |
| context_precision | 20 | 17 | 10 | 47 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** context_precision

**Lý do phân tích:**

> Phân tích dữ liệu lỗi của corpus HR policy tiếng Việt cho thấy distribution **factual** có tỷ lệ failure cao nhất, và metric **context_precision** là điểm yếu chủ đạo của hệ thống RAG hiện tại.
> Điều này xuất phát từ việc các câu hỏi thuộc nhóm này đòi hỏi sự suy luận logic phức tạp hoặc có cấu trúc bẫy làm nhiễu thông tin, khiến cho việc truy xuất thông tin không chính xác hoặc LLM bị ảo tưởng (hallucination).
> Thêm vào đó, việc xử lý tiếng Việt với nhiều ngữ nghĩa đa dạng khiến cho việc so sánh độ tương đồng ngữ nghĩa của các mô hình embedding chưa thực sự tối ưu, dẫn tới việc bỏ lỡ các thông tin quan trọng.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Thắt chặt system prompt, giảm nhiệt độ (temperature) của mô hình LLM về 0.0 để tăng tính nhất quán |
| context_recall | Missing relevant chunks | Cải thiện chiến lược chunking (ví dụ: dùng hierarchical chunking) hoặc tích hợp thêm tìm kiếm từ khóa BM25 |
| context_precision | Too many irrelevant chunks | Tích hợp thêm mô hình Reranker (CrossEncoder) hoặc áp dụng bộ lọc siêu dữ liệu (metadata filtering) |
| answer_relevancy | Answer doesn't match question | Cải thiện prompt template, hướng dẫn rõ ràng hơn về cách trả lời câu hỏi và định dạng đầu ra |

---

## 6. Nhận xét về Adversarial Distribution

> Điểm trung bình (avg_score) của nhóm adversarial là 0.5263, thấp hơn so với factual (0.7034) và multi_hop (0.4891).
> Hệ thống RAG thực sự bị ảnh hưởng nghiêm trọng bởi các bẫy version conflicts (như phân biệt chính sách nghỉ phép năm v2023 có 12 ngày phép với v2024 có 15 ngày phép).
> Các câu hỏi rơi vào nhóm này thường truy xuất các phiên bản chính sách cũ và mới cùng lúc, dẫn tới việc LLM bị nhiễu thông tin và đưa ra thông tin không chính xác.
