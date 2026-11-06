# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 4388.
- **Nguồn báo cáo:** code, câu trả lời và output đã lưu trong [notebook bài làm](../../src/day01_spam_ml_dl_embedding_llm.ipynb), cùng [lab1_report.json](lab1_report.json).
- **Phạm vi thực hiện:** Phần 0–4 đã có code và output; Phần 5 đã chạy bảng tổng hợp, biểu đồ đánh đổi, tính tay, nhận đề và xuất báo cáo cuối. Phân loại LLM trên SUB không có nhãn hợp lệ; stress test chưa có bảng kết quả; code thử thách cá nhân còn trống.

## 1. Bằng chứng và công việc đã thực hiện

| Tệp | Trạng thái |
| --- | --- |
| [Notebook trong src](../../src/day01_spam_ml_dl_embedding_llm.ipynb) | Có output Phần 0–5 với các giới hạn nêu dưới đây; chưa có bản `day01_lab.ipynb` trong thư mục nộp |
| [lab1_report.json](lab1_report.json) | Đã có: SEED 4388, 10 dự đoán, 15/15 kiểm tra đạt và 8 hàng kết quả (3 hàng LLM không hợp lệ) |
| [Ảnh tính tay 1](hand_calc_1.jpeg) | Có các bước tính baseline accuracy, sigmoid và F1 |
| [Ảnh tính tay 2](hand_calc_2.jpeg) | Có các bước tính cosine và chi phí LLM theo giả định của đề |
| `hand_calc.jpg` | Chưa có file đúng tên quy định; hiện có hai ảnh JPEG ở trên |

### Phần 0 — Khám phá và chuẩn bị dữ liệu

Dữ liệu ban đầu có 5.572 tin, gồm 4.825 ham và 747 spam. Em kiểm tra độ dài tin nhắn, phát hiện 403 bản trùng và giải thích việc tin trùng xuất hiện ở cả train và test có thể làm điểm đánh giá quá lạc quan. Sau khi bỏ trùng chính xác, dữ liệu còn 5.169 tin: 4.516 ham và 653 spam.

Với SEED 4388, tập train có 3.619 tin (438 spam), validation có 775 tin và test có 775 tin (112 spam). Tập SUB gồm 40 tin cân bằng, 20 spam và 20 ham, lấy từ test. Từ điển BoW gồm 2.000 từ được xây dựng từ tập train.

### Phần 1 — Machine Learning

Em đã điền các hàm `vectorize`, `sigmoid`, `predict_proba`, `train_logreg` và `metrics`. Vector hoá dùng đặc trưng nhị phân, từ xuất hiện nhiều lần vẫn chỉ nhận giá trị 1; logistic regression học bằng gradient descent với NumPy. Bốn ô kiểm tra 1.1–1.4 đều có output đạt.

Em đã huấn luyện logistic regression, đo chất lượng và tốc độ, xem các từ có trọng số mạnh và đọc các tin bị phân loại sai. Em cũng chạy thí nghiệm learning rate 0.001, 0.5 và 50, giải thích cách cập nhật trọng số và nhận diện lỗi đánh giá trên train, dùng dữ liệu chưa bỏ trùng và xây từ điển trên toàn bộ dữ liệu.

### Phần 2 — Deep Learning

Em đã xây MLP `Linear(2000, 64) → ReLU → Linear(64, 1)` và hoàn thành vòng lặp huấn luyện với các bước xoá gradient, forward, tính loss, backward và cập nhật trọng số. Hai ô kiểm tra 2.1–2.2 đều đạt. MLP có 128.129 tham số, so với 2.001 tham số của logistic regression.

Em đã train 30 epoch, theo dõi train/validation loss và thử bỏ ReLU. Output hiện tại in chỉ số validation loss nhỏ nhất là **6** theo `np.argmin`, tức epoch thứ **7** nếu đếm từ 1. Câu trả lời “epoch 4” trong notebook chưa khớp lần chạy này. F1 test 0.944 là kết quả của mô hình sau 30 epoch; code chưa lưu và khôi phục trọng số tại epoch có validation loss thấp nhất.

### Phần 3 — Embedding

Em đã hoàn thành `cosine_similarity`, `top_k_similar` và `knn_predict`; hai ô kiểm tra tương ứng đều đạt. Tìm kiếm dùng chuẩn hoá vector và phép nhân ma trận, sau đó lấy chỉ số theo cosine giảm dần; kNN quyết định bằng bỏ phiếu đa số.

Encoder MiniLM đóng băng tạo vector 384 chiều. Em đã so sánh ba đầu phân loại kNN, LogReg và MLP, chạy truy vấn tìm kiếm ngữ nghĩa, thử sáu tin ngụy trang và chạy ô trực quan hoá. Truy vấn về việc được chọn nhận kỳ nghỉ miễn phí trả về sáu tin spam gần nhất, cosine khoảng 0.503–0.546. Trên sáu tin ngụy trang, BoW đúng 4/6 và Encoder + LogReg đúng 5/6.

### Phần 4 — LLM và tổng hợp

Em đã điền `call_openai`, `build_classification_prompt`, `parse_label` và `llm_classify_batch`. Prompt yêu cầu một nhãn spam/ham, hỗ trợ few-shot và coi nội dung SMS là dữ liệu không đáng tin cậy. Hàm batch có retry, ghi lỗi, đánh dấu dự đoán không hợp lệ bằng NaN và ước tính chi phí theo số từ. Hai ô kiểm tra Phần 4 có output đạt; một lời gọi thử trả về câu chào với thời gian 1.61 giây.

Tuy nhiên, trên SUB 40, cả model nhỏ zero-shot, model lớn zero-shot và model nhỏ few-shot đều in **invalid=40**. Chưa có nhãn hợp lệ để đánh giá chất lượng LLM hoặc lợi ích few-shot. Các câu trả lời trong notebook nói LLM đạt F1 cao, độ trễ vài chục mili giây và prompt phòng thủ cải thiện rõ chưa được output hiện tại chứng minh.

Báo cáo cuối và JSON ghi **15/15 kiểm tra đạt**, gồm 8 kiểm tra Phần 1–3, 2 kiểm tra LLM và 5 kiểm tra tính tay. Kiểm tra hàm đạt không bảo đảm thí nghiệm API trên dữ liệu thật thành công.

## 2. Kết quả chính

Bảng dưới lấy từ `lab1_report.json` vừa cập nhật; năm hàng ML/DL/embedding khớp bảng tổng hợp đã lưu trong notebook mới nhất. Số liệu được làm tròn để trình bày.

| Phương án | Precision test | Recall test | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 ms/tin | USD/1k tin (*) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.989 | 0.795 | 0.881 | 0.824 | 7.70 | 12497 | 0.17 | 0 |
| 2. BoW + MLP (deep learning) | 0.990 | 0.902 | 0.944 | 0.857 | 9.27 | 35610 | 0.41 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.883 | 0.946 | 0.914 | 0.895 | 2.07 | 84 | 37.32 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.956 | 0.964 | 0.960 | 0.947 | 3.27 | 1233 | 25.88 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.964 | 0.946 | 0.955 | 0.919 | 4.40 | 694 | 55.02 | 0 |
| LLM nhỏ zero-shot | — | — | — | Không hợp lệ (40/40 invalid) | 0 | — | — | Chưa xác nhận |
| LLM lớn zero-shot | — | — | — | Không hợp lệ (40/40 invalid) | 0 | — | — | Chưa xác nhận |
| LLM nhỏ few-shot | — | — | — | Không hợp lệ (40/40 invalid) | 0 | — | — | Chưa xác nhận |

JSON lưu F1 LLM bằng 0, p50/p95 là NaN, throughput 40.000.000.000 tin/giây và chi phí 0. Đây không phải kết quả benchmark hợp lệ: hàm batch không có nhãn hợp lệ, `latency_s` cộng bỏ qua NaN và phép tính throughput dùng mẫu số tối thiểu rất nhỏ. Không diễn giải các giá trị này thành LLM phân loại kém, chạy cực nhanh hoặc miễn phí. JSON còn chứa literal `NaN` do cách xuất của Python, không phải giá trị hợp lệ theo chuẩn JSON nghiêm ngặt; báo cáo giữ nguyên tệp nguồn đã xuất.

(*) Giá trị 0 là chi phí API được gán trong code cho mô hình chạy cục bộ, chưa bao gồm máy chủ, CPU/GPU và vận hành. Thời gian của nhóm embedding gồm encode tập train và huấn luyện đầu phân loại nếu có. Độ trễ đo từ văn bản thô tới nhãn trên môi trường chạy notebook.

MiniLM + LogReg đạt F1 cao nhất trong các phương án đã chạy (0.960). Thêm MLP head trên cùng embedding cho F1 0.955, thấp hơn 0.005. Với BoW, MLP nâng F1 từ 0.881 lên 0.944, chủ yếu nhờ recall tăng từ 0.795 lên 0.902.

## 3. Dự đoán và thực tế

JSON đã ghi đủ 10 dự đoán. Bảng sau đối chiếu ba dự đoán có kết quả hợp lệ ở Phần 1–3; các dự đoán LLM chưa thể kiểm chứng vì 40/40 tin không có nhãn hợp lệ.

| Dự đoán đã ghi | Kết quả thực tế | Nhận xét |
| --- | --- | --- |
| `ml_f1` (15:26:41): F1 khoảng 0.9, precision cao hơn recall | F1 0.881; precision 0.989, recall 0.795 | F1 thấp hơn dự đoán khoảng 0.019. BoW bắt được từ khoá đặc trưng nhưng bỏ sót spam có cách diễn đạt khác. |
| `dl_curve` (15:27:15): validation loss không giảm mãi; MLP hơn hoặc tương đương LogReg | Chỉ số val loss nhỏ nhất là 6; F1 MLP 0.944 so với LogReg 0.881 | MLP tăng F1 0.063. Cần phân biệt epoch tốt nhất theo validation với mô hình cuối 30 epoch đang được đánh giá. |
| `dl_relu` (15:27:25): bỏ ReLU thì F1 giảm hoặc bằng | F1 không ReLU là 0.944, bằng MLP có ReLU ở độ chính xác ba chữ số | Kết quả thuộc nhánh “bằng”. Hai lớp tuyến tính liên tiếp có thể gộp thành một phép biến đổi tuyến tính; lợi ích từ ReLU chưa thể hiện trên phép đo này. |

Dự đoán `emb_tricky` (15:27:56) cũng khớp: Encoder + LogReg đúng 5/6, cao hơn BoW 4/6. Tuy nhiên, ví dụ “Mum says the prize for the school raffle is a cake lol” trong câu trả lời Phần 3 cần sửa nhận xét: output cho thấy **cả hai mô hình đều đoán đúng ham**, không phải BoW đoán sai.

Câu trả lời learning rate trong notebook khẳng định loss thành NaN; báo cáo chưa xác nhận khẳng định này từ số liệu văn bản đã lưu. BCE có bước clip xác suất, nên learning rate lớn không tự động đồng nghĩa loss trở thành NaN.

Dự đoán `tradeoff` (16:21:32) cho rằng kNN nằm ngoài SLA 50 ms, nhưng p95 thực tế là 37.32 ms, nên dự đoán này sai. Dựa trên p95 và F1 SUB hợp lệ, đường Pareto gồm BoW + LogReg, BoW + MLP và MiniLM + LogReg; kNN và MiniLM + MLP head bị MiniLM + LogReg lấn át. Khoảng cách F1 trên SUB 40 cần được xem thận trọng; một lỗi làm accuracy thay đổi 2.5 điểm phần trăm, còn mức thay đổi F1 phụ thuộc TP, FP và FN.

Dự đoán `llm_models` và `llm_fewshot` chưa thể xác nhận về F1, tỷ lệ giá hoặc lợi ích few-shot. Log có lần chạy few-shot lúc 15:37:10, trước thời điểm dự đoán được lưu cuối cùng (16:06:31), sau đó chạy lại lúc 16:06:33; vì log ghi đè dự đoán theo key, không thể dùng bản xuất hiện tại chứng minh dự đoán few-shot đã có trước lần chạy đầu tiên.

## 4. Phân tích case khó

**Bộ 50 case đã được khởi tạo và ô đánh giá đã chạy**, sau dự đoán `hard_eval` lúc 16:21:52. Tuy nhiên, output ô đánh giá chỉ lưu cảnh báo `invalid value encountered in cast` khi ép dự đoán sang số nguyên, chưa có bảng F1/accuracy tổng hợp hoặc kết quả theo nhóm. JSON cũng không chứa kết quả stress test. Vì vậy chưa thể kết luận model nào tụt nhiều nhất hay prompt phòng thủ có hiệu quả; các nhận xét trong notebook về mức tụt và hiệu quả phòng thủ mới là nhận định chưa được số liệu đã lưu xác nhận.

Trong thí nghiệm sáu tin ngụy trang đã chạy, có hai ví dụ đáng chú ý:

1. **“Exclusive: get a brand new phone at no cost, just reply with…”** — nhãn thật spam, BoW đoán ham, Encoder + LogReg đoán đúng spam. Cách diễn đạt “at no cost” truyền đạt ý miễn phí qua nhiều từ; biểu diễn BoW không trực tiếp gắn cụm này với từ “free”, còn embedding có thể biểu diễn sự gần nhau về nghĩa.
2. **“Last chance! Your account rewards expire tonight, confirm de…”** — nhãn thật spam, cả BoW và Encoder + LogReg đều đoán ham. Đây là một lỗi thực tế cho thấy embedding không bảo đảm nhận diện đúng mọi lời thúc giục xác nhận tài khoản. Có thể mẫu diễn đạt chưa được đầu phân loại học tốt; cần xem xác suất và các mẫu train gần nhất để kiểm chứng nguyên nhân.

Hai trường hợp này thuộc thí nghiệm 3.B, không phải kết quả của bộ 50 case ở Phần 5. Mẫu sáu tin quá nhỏ để suy rộng chất lượng ngoài thực tế.

## 5. Tính tay và thử thách cá nhân

Hai ảnh viết tay đã có đủ các bước cho năm phép tính. Kết quả thể hiện trên ảnh gồm baseline accuracy khoảng 0.874, sigmoid tại 0 bằng 0.5, F1 khoảng 0.73, cosine khoảng 0.6 và chi phí 10.50 USD/ngày theo giả định giá/token trong đề. Khi điền theo yêu cầu làm tròn hai chữ số, baseline accuracy dùng 0.87 và cosine dùng 0.60. Ô `HAND_CALC` trong notebook mới nhất đã được điền năm giá trị trên; đã có output cả năm kiểm tra đạt. JSON cũng ghi cả năm kiểm tra là `true`.

Theo quy tắc `4388 % 6 = 2`, thử thách tương ứng là **đề số 3**: đổi lớp ẩn MLP thành 4 và 512 nơ-ron, báo số tham số, F1 test, thời gian train và giải thích khi nào chọn mạng nhỏ. Ô nhận đề đã chạy và JSON ghi `challenge: 2` (chỉ số từ 0). Ô code thử thách còn trống, chưa có kết quả thực nghiệm cho đề này. Do đó chưa có kết quả cho hai cấu hình này. Cấu hình 64 nơ-ron ở Phần 2 chỉ là kết quả gốc để đối chiếu, chưa hoàn thành thử thách cá nhân.

## 6. Phương án cho sản phẩm thật

Với yêu cầu 10 triệu SMS/ngày, lưu lượng trung bình khoảng 115.74 tin/giây và p95 < 50 ms/tin, em đề xuất **MiniLM + LogReg** theo lần đo trong JSON: F1 test 0.960, F1 SUB 0.947, precision 0.956, recall 0.964, p95 25.88 ms/tin và throughput cả lô khoảng 1.233 tin/giây. Đây vẫn là số đo trên môi trường notebook.

Bốn phương án đạt p95 < 50 ms trong lần đo này: BoW + LogReg (0.17 ms), BoW + MLP (0.41 ms), MiniLM + kNN (37.32 ms) và MiniLM + LogReg (25.88 ms). MiniLM + MLP head không đạt với 55.02 ms. BoW + MLP là phương án nhẹ hơn, F1 0.944 và recall 0.902; chọn MiniLM + LogReg cần tài nguyên encode lớn hơn nhưng đạt F1 0.960 và recall 0.964. kNN có throughput cả lô khoảng 84 tin/giây, thấp hơn lưu lượng trung bình 116 tin/giây nếu giữ nguyên cấu hình xử lý; đạt p95 từng tin chưa đủ để bảo đảm năng lực phục vụ. Cần kiểm thử tải đồng thời và giờ cao điểm trước khi cam kết SLA.

Một hướng mở rộng là dùng BoW + MLP xử lý các tin có độ tin cậy cao, chuyển các tin lưỡng lự sang embedding hoặc LLM để đánh giá bổ sung. Đây mới là đề xuất thiết kế, chưa triển khai hoặc đo trong notebook; nếu xử lý đồng bộ, nhánh bổ sung vẫn phải được đo lại theo SLA toàn hệ thống.

Theo đúng giả định bài tính tay (100 token input, 1 token output; giá 0.10/0.50 USD mỗi triệu token), chi phí cho 1 triệu SMS là 10.50 USD/ngày. Suy ra 10 triệu SMS/ngày khoảng 105 USD/ngày, tương đương 3.150 USD/30 ngày nếu gọi LLM cho mọi tin. Đây là phép tính theo đề, không phải giá API đã xác minh hoặc chi phí đo thực tế; chưa tính prompt dài hơn, retry và hạ tầng. Phần LLM đã chạy nhưng không có nhãn hợp lệ trên SUB, nên chưa đủ bằng chứng so sánh chất lượng, độ trễ và chi phí thực tế với các mô hình còn lại. Tỷ lệ tin được chuyển qua router cũng chưa được đo; không thể khẳng định nhánh LLM chỉ chiếm dưới 5% hoặc bảo đảm độ chính xác tuyệt đối.

## 7. AI log và điều đã học

| Phạm vi | Yêu cầu gửi AI | AI thực hiện | Trạng thái kiểm chứng |
| --- | --- | --- | --- |
| Viết báo cáo hiện tại | Đọc notebook đã làm, viết báo cáo và đổi tên file theo thông tin cá nhân | Đọc code/output và hai ảnh tính tay; tổng hợp số liệu, nêu phần chưa hoàn thành, chỉ ra câu trả lời chưa khớp output; đổi tên báo cáo | Đối chiếu code/output notebook và JSON; sau khi đồng bộ main, cập nhật bảng theo JSON mới, xác nhận 15/15 kiểm tra và 10 dự đoán, phân biệt kết quả LLM không hợp lệ và stress test chưa đủ output; không sửa code hoặc hoàn thành TODO thay người học |

Notebook mới nhất chưa có AI log chi tiết để xác nhận các lần dùng AI trước đó. Bảng trên ghi nhận việc hỗ trợ viết và cập nhật báo cáo trong cuộc trao đổi này.

Từ kết quả đã chạy, em rút ra rằng biểu diễn dữ liệu ảnh hưởng lớn tới chất lượng: dùng cùng họ mô hình LogReg nhưng chuyển BoW sang embedding nâng F1 từ 0.881 lên 0.960. Thêm lớp ẩn không luôn tăng F1: MLP cải thiện rõ trên BoW nhưng thấp hơn LogReg trên cùng embedding. Khi chọn mô hình cho sản phẩm, cần đọc đồng thời precision, recall và độ trễ, thay vì chỉ chọn F1 cao nhất.
