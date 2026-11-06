# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 4388.
- **Bài nộp:** [day01_lab.ipynb](day01_lab.ipynb); bản nguồn [trong src](../../src/day01_spam_ml_dl_embedding_llm.ipynb) có cùng nội dung.
- **Dữ liệu kết quả:** [lab1_report.json](lab1_report.json).
- **Phạm vi lần cập nhật:** AI hỗ trợ sửa và chạy lại các mô hình local, thử thách MLP và stress test. Phần API thật chưa chạy được vì môi trường thiếu credential; các chỉ số tương ứng để `null`, không thay bằng số 0.

## 1. Bằng chứng và quá trình thực hiện

| Bằng chứng | Trạng thái |
| --- | --- |
| [Notebook bài nộp](day01_lab.ipynb) | Có code và output chạy lại, gồm thử thách và 50 case |
| [JSON báo cáo](lab1_report.json) | JSON chuẩn, 15/15 kiểm tra ghi nhận đạt; kiểm tra LLM là mock |
| [Ảnh tính tay 1](hand_calc_1.jpeg), [ảnh tính tay 2](hand_calc_2.jpeg) | Đủ bước cho 5 phép tính; hiện chưa gom thành tên `hand_calc.jpg` theo hướng dẫn |

Dữ liệu ban đầu: 5.572 tin, 4.825 ham và 747 spam. Sau bỏ 403 bản trùng chính xác còn 5.169 tin (4.516 ham, 653 spam). Train/validation/test lần lượt 3.619/775/775 tin; test có 112 spam. SUB gồm 20 spam và 20 ham lấy từ test. Vocab 2.000 từ chỉ xây từ train; lần sửa sắp thứ tự token trước đếm để ổn định thứ tự khi bằng tần suất.

Phần 1 hoàn thành vector hoá nhị phân, sigmoid, dự đoán xác suất, gradient descent và precision/recall/F1 bằng NumPy. Sigmoid đã được sửa dùng `logaddexp` để không overflow khi đầu vào lớn. Phần 2 dùng MLP 2.000→64→1 với ReLU, BCEWithLogitsLoss và Adam. Phần 3 dùng MiniLM đóng băng, vector 384 chiều, kết hợp kNN/LogReg/MLP. Các code học viên đã có được giữ và bổ sung phần sửa cần thiết; lần sửa có AI hỗ trợ được ghi trong mục 7.

Phần 4 đã có wrapper API, prompt, parser và batch. Preflight kiểm tra credential/tên model trước gọi hàng loạt. Lỗi không bị gán thành ham, không ép dự đoán thiếu thành số nguyên; lỗi cấu hình dừng sớm. Bản cũ có `invalid=40` nên F1=0, throughput 40 tỷ tin/giây và chi phí 0 không phải số đo hợp lệ. Bản mới ghi `missing_credentials`, chưa có kết luận chất lượng LLM. Lượt gọi thử lịch sử thành công không bảo đảm lượt batch thành công; chưa có đủ log để xác định nguyên nhân chính xác của lỗi cũ.

**Nguồn đo:** chạy lại trên CPU, Python 3.14.9, PyTorch 2.15.0+cpu, NumPy 2.5.4, 2 thread. Timestamp UTC: `2026-11-06T17:11:17.504627+00:00`. Số đo local thay thế số cũ trong bảng; thay đổi môi trường và thứ tự vocab có thể làm khác thời gian và một số kết quả MLP. Không trộn thời gian từ Colab với lần chạy local.

## 2. Kết quả chính

| Phương án | Precision | Recall | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 ms/tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.989 | 0.795 | 0.881 | 0.824 | 0.98 | 86557 | 0.03 |
| 2. BoW + MLP (deep learning) | 0.990 | 0.911 | 0.949 | 0.889 | 1.51 | 99894 | 0.10 |
| 3a. Encoder (MiniLM) + kNN | 0.883 | 0.946 | 0.914 | 0.895 | 20.84 | 126 | 15.72 |
| 3b. Encoder (MiniLM) + LogReg | 0.956 | 0.964 | 0.960 | 0.947 | 21.10 | 170 | 13.52 |
| 3c. Encoder (MiniLM) + MLP head | 0.964 | 0.946 | 0.955 | 0.919 | 21.29 | 163 | 18.03 |
| LLM nhỏ zero-shot | — | — | — | Chưa đo | 0 | — | — |
| LLM lớn zero-shot | — | — | — | Chưa đo | 0 | — | — |
| LLM nhỏ few-shot | — | — | — | Chưa đo | 0 | — | — |

Chi phí API của năm mô hình local bằng 0 trong code, chưa tính máy chủ và vận hành. Chi phí LLM chưa đo (`null`); không được coi là miễn phí. Thời gian nhóm embedding gồm encode tập train và train head nếu có. Độ trễ tính từ văn bản thô đến nhãn; throughput cả lô khác năng lực phục vụ nhiều request đồng thời.

MiniLM + LogReg có F1 test cao nhất (0.960); MLP head thấp hơn 0.005. Với BoW, MLP tăng F1 0.068. Bản xuất mới dùng `allow_nan=False` và chuyển mọi giá trị không hữu hạn thành `null`; giữ cả trạng thái, số nhãn hợp lệ, kết quả thử thách và dự đoán từng case.

## 3. Dự đoán và thực tế

Dự đoán ban đầu được giữ nguyên trong `original_prediction_log`, lịch sử trong `original_experiments`. Lần AI chạy lại tái sử dụng nội dung dự đoán; timestamp mới không phải bằng chứng có dự đoán độc lập mới trước thí nghiệm. Log gốc few-shot từng bị ghi đè lúc 16:06:31 sau lần chạy 15:37:10, nên không đủ bằng chứng dự đoán có trước lần chạy đầu.

| Dự đoán gốc | Thực tế lần chạy lại | Giải thích |
| --- | --- | --- |
| `ml_lr`: lr=50 làm loss bùng nổ, không hội tụ | Loss tăng tới 2.499690 nhưng cuối còn 0.004709, tất cả hữu hạn | Bước lớn gây overshoot ban đầu; không đủ để kết luận mọi epoch đều không hội tụ. Sigmoid đã sửa ổn định số học. |
| `hard_eval`: accuracy khoảng 60–70% cho các model | BoW + LogReg 52%, BoW + MLP 58%; nhóm embedding 66–68% | Spam giọng tự nhiên và né từ khoá làm BoW giảm mạnh hơn dự đoán. |
| `tradeoff`: kNN nằm ngoài SLA 50 ms | p95 kNN 15.72 ms, dưới 50 ms | Độ trễ phụ thuộc máy và cách chạy; cần dùng số đo hiện tại. |

Validation loss MLP 64 nhỏ nhất tại index 6 (epoch 7 nếu đếm từ 1), thay nhận xét cũ epoch 4. F1 đang báo là mô hình cuối 30 epoch, chưa khôi phục checkpoint tối ưu. Bỏ ReLU cho F1 0.944, so với có ReLU 0.949 trong lần này.

Ở sáu tin ngụy trang, BoW đúng 4/6 và Encoder + LogReg đúng 5/6. Tin về quà xổ số trường học được cả hai đoán đúng ham; ví dụ BoW sai và embedding đúng là lời mời điện thoại “at no cost”. Dự đoán model lớn/model nhỏ và few-shot chưa kiểm chứng bằng API thật; không kết luận LLM thắng hoặc phòng thủ có hiệu quả khi chưa có nhãn hợp lệ.

## 4. Phân tích 50 case khó

Mỗi mô hình local có đủ 50 nhãn; bảng theo nhóm được lưu trong output, JSON có `hard_eval.summary`, `hard_eval.by_category` và `hard_eval.cases`.

| Model | F1 SUB | F1 50 case | Mức giảm F1 | Accuracy 50 case |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.824 | 0.333 | 0.490 | 52.0% |
| 2. BoW + MLP | 0.889 | 0.462 | 0.427 | 58.0% |
| 3a. Emb + kNN | 0.895 | 0.714 | 0.180 | 68.0% |
| 3b. Emb + LogReg | 0.947 | 0.667 | 0.281 | 66.0% |
| 3c. Emb + MLP head | 0.919 | 0.692 | 0.227 | 68.0% |

BoW + LogReg tụt nhiều nhất (0.490); kNN có F1 cao nhất trên bộ khó (0.714), dù LogReg trên embedding cao nhất ở test thường. Bộ test thường gần phân phối train và vẫn có mẫu gần trùng; bộ viết tay cố ý kiểm tra các cách diễn đạt khác nên điểm giảm.

| Nhóm | BoW + LR | BoW + MLP | Emb + kNN | Emb + LR | Emb + MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| ham_looks_spam | 100.0% | 90.0% | 70.0% | 80.0% | 80.0% |
| manipulation | 12.5% | 50.0% | 100.0% | 50.0% | 50.0% |
| natural_spam | 10.0% | 10.0% | 30.0% | 30.0% | 40.0% |
| obfuscation | 16.7% | 0.0% | 83.3% | 83.3% | 83.3% |
| prompt_injection | 66.7% | 83.3% | 83.3% | 100.0% | 100.0% |
| talks_about_spam | 100.0% | 100.0% | 25.0% | 50.0% | 50.0% |
| transactional_ham | 83.3% | 100.0% | 83.3% | 83.3% | 83.3% |

BoW + LR khó nhất với spam giọng tự nhiên (10% đúng); BoW + MLP khó nhất với né từ khoá (0%). Với kNN, nhóm nói về spam khó nhất (25%); hai head embedding train được khó nhất với spam giọng tự nhiên (30%/40%).

Hai case cụ thể:

1. **#11 — spam giả giọng thân mật:** lời mời khuyến mãi từ “Jess from the gym”. Hai mô hình BoW đoán ham, cả ba embedding đoán spam. Cách mở đầu giống tin người quen làm đặc trưng từ rời rạc thiếu tín hiệu rõ.
2. **#29 — spam thay chữ bằng số:** “C0ngr4tulat10ns! Y0u h4ve w0n…”. Hai mô hình BoW đoán ham, cả ba embedding đoán spam; nhiều token biến dạng không khớp vocab.

Chưa có kết quả LLM 4a/4d trên 50 case vì thiếu credential, nên chưa xác nhận hiệu quả phòng thủ prompt injection. Có thể chuẩn hoá ký tự và gắn cờ đáng ngờ trước model, nhưng cần tránh sửa nhầm mã hợp lệ hoặc chặn tin chỉ đang trích dẫn lệnh. Bộ 50 case dùng để tìm điểm yếu/hồi quy, không ước lượng trực tiếp phân phối lỗi thật.

## 5. Tính tay và thử thách cá nhân

Các giá trị tính tay đã điền và kiểm tra đạt: baseline accuracy 0.87, sigmoid(0) 0.50, F1 ví dụ 0.73, cosine 0.60 và chi phí theo đề 10.50 USD/ngày cho 1 triệu SMS. Hai ảnh có bước tính đi kèm.

SEED 4388 chọn đề số 3 (`4388 % 6 = 2`): thử MLP hidden=4 và hidden=512. Hai cấu hình dùng cùng dữ liệu, seed, Adam lr=0.001, batch=64, 30 epoch; đánh giá checkpoint cuối để so sánh cùng quy trình.

| Hidden | Số tham số | F1 test | Train (s) | Epoch val tốt nhất (từ 1) |
| --- | ---: | ---: | ---: | ---: |
| 4 | 8,009 | 0.949 | 0.84 | 24 |
| 512 | 1,025,025 | 0.935 | 9.57 | 2 |

Công thức tham số: `2002*h + 1`. Mạng 4 nơ-ron ít tham số hơn khoảng 128 lần, train nhanh hơn và F1 cao hơn mạng 512 trong lần này. Mạng 512 có validation tốt nhất sớm (epoch 2), nên cần xem nguy cơ overfit; mạng nhỏ phù hợp khi dữ liệu ít, tài nguyên hạn chế và validation đủ tốt. Chưa thể suy rộng một lần đo thành quy luật cho mọi bài toán.

**Công khai AI:** code, chạy thử và phân tích thử thách lần sửa này có AI hỗ trợ theo yêu cầu người học. Đề gốc quy định không dùng AI; bản sửa không khẳng định hoàn thành độc lập và người học cần tự kiểm tra/giải thích trước khi nộp.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày (~115.74 tin/giây) và p95 < 50 ms, MiniLM + LogReg là lựa chọn ban đầu theo test thường: F1 0.960, recall 0.964, p95 13.52 ms, throughput cả lô 170 tin/giây. Cả năm mô hình local đạt p95 trong lần đo CPU này; vẫn phải kiểm thử tải đồng thời/giờ cao điểm trước khi cam kết phục vụ. BoW + MLP nhẹ hơn với F1 0.949; kNN mạnh hơn trên bộ khó. Chọn model phải xét cả loại lỗi, phân phối và tài nguyên.

Đường Pareto theo p95/F1 SUB: 1. BoW + LogReg (tự code), 2. BoW + MLP (deep learning), 3b. Encoder (MiniLM) + LogReg. MiniLM + LogReg lấn át kNN và MLP head ở hai thước đo đó. SUB chỉ 40 tin; một lỗi đổi accuracy 2.5 điểm phần trăm, còn F1 phụ thuộc TP/FP/FN, nên không xếp hạng chắc chắn từ chênh lệch nhỏ.

Có thể dùng model local xử lý các tin chắc chắn, chuyển các tin lưỡng lự sang kiểm tra thêm. Tỷ lệ chuyển tuyến và chất lượng/độ trễ LLM chưa đo; không cam kết dưới 5% hoặc độ chính xác tuyệt đối.

Theo giả định của đề, 100 input + 1 output token/SMS, giá nhỏ 0.10/0.50 USD mỗi triệu token: 10 triệu SMS/ngày khoảng 105 USD/ngày, 3.150 USD/30 ngày. Với giá lớn giả định 1.00/5.00: 31.500 USD/30 ngày. Đây là phép tính giả định, chưa xác minh giá hiện hành hoặc đo hóa đơn; chưa gồm retry, prompt dài hơn, hạ tầng.

## 7. AI log và điều đã học

| Phạm vi | Yêu cầu | AI hỗ trợ | Kiểm chứng / giới hạn |
| --- | --- | --- | --- |
| Báo cáo các lượt trước | Đọc bài làm, viết report, đồng bộ Git | Tổng hợp output, đổi tên báo cáo, thêm bản notebook nộp | Đối chiếu dữ liệu đã lưu |
| Lần sửa hiện tại | Sửa cả hai notebook: MLP 4/512, LLM, stress test, AI log và JSON NaN | Viết/sửa code, chạy lại local, sửa giải thích và đồng bộ kết quả | Đã chạy CPU, kiểm thử mock API; không có kết quả LLM thật vì thiếu credential |
| Các lần dùng AI trước cuộc trao đổi này | Chưa có thông tin từ người học | Không tự tạo lịch sử | Người học bổ sung nếu có |

Log này cũng được ghi trong hai notebook và JSON. Thử thách có AI hỗ trợ, không được coi là bằng chứng tự làm không AI. Tham chiếu kỹ thuật API: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Điều rút ra: biểu diễn dữ liệu ảnh hưởng mạnh tới chất lượng và mô hình nhiều tham số không luôn tốt hơn. Mock test đạt chỉ chứng minh các nhánh code được kiểm tra, không chứng minh API thật phân loại thành công. Dữ liệu thiếu phải có trạng thái rõ ràng, không biến thành nhãn ham hoặc số đo benchmark giả.
