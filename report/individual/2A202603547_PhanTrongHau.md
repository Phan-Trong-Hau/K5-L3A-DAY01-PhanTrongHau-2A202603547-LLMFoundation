# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 4388.
- **Nguồn báo cáo:** code, câu trả lời và output đã lưu trong [notebook bài làm](../../src/day01_spam_ml_dl_embedding_llm.ipynb).
- **Phạm vi thực hiện:** Phần 0–3 đã có code và output; Phần 4–5 chưa chạy. Báo cáo ghi nhận trạng thái hiện có, không coi các ô mẫu hoặc TODO chưa làm là kết quả thực nghiệm.

## 1. Bằng chứng và công việc đã thực hiện

| Tệp | Trạng thái |
| --- | --- |
| [Notebook trong src](../../src/day01_spam_ml_dl_embedding_llm.ipynb) | Có output Phần 0–3; chưa có bản `day01_lab.ipynb` trong thư mục nộp |
| `lab1_report.json` | Chưa có; ô “Báo cáo cuối” chưa chạy |
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

Tổng cộng có **8 ô kiểm tra đạt ở Phần 1–3** theo output đã lưu. Đây chưa phải tổng kết toàn bộ lab vì các kiểm tra LLM, tính tay và ô báo cáo cuối chưa chạy.

## 2. Kết quả chính

Các số dưới đây được chép từ output hiện có, đã làm tròn như notebook. Ô bảng tổng hợp Phần 5 chưa chạy nên chưa có giá trị F1 SUB 40 được hiển thị để trích dẫn.

| Phương án | Precision test | Recall test | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 ms/tin | USD/1k tin (*) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.989 | 0.795 | 0.881 | Chưa hiển thị | 15.5 | 5.570 | 0.14 | 0 |
| BoW + MLP | 0.990 | 0.902 | 0.944 | Chưa hiển thị | 24.1 | 12.740 | 4.52 | 0 |
| MiniLM + kNN | 0.883 | 0.946 | 0.914 | Chưa hiển thị | 3.6 | 53 | 88.50 | 0 |
| MiniLM + LogReg | 0.956 | 0.964 | 0.960 | Chưa hiển thị | 6.0 | 679 | 72.40 | 0 |
| MiniLM + MLP head | 0.964 | 0.946 | 0.955 | Chưa hiển thị | 6.8 | 1.110 | 50.03 | 0 |
| LLM zero-shot / few-shot | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | — | — | — | — |

(*) Giá trị 0 là chi phí API được gán trong code cho mô hình chạy cục bộ, chưa bao gồm máy chủ, CPU/GPU và vận hành. Thời gian của nhóm embedding gồm encode tập train và huấn luyện đầu phân loại nếu có. Độ trễ đo từ văn bản thô tới nhãn trên môi trường chạy notebook.

MiniLM + LogReg đạt F1 cao nhất trong các phương án đã chạy (0.960). Thêm MLP head trên cùng embedding cho F1 0.955, thấp hơn 0.005. Với BoW, MLP nâng F1 từ 0.881 lên 0.944, chủ yếu nhờ recall tăng từ 0.795 lên 0.902.

## 3. Dự đoán và thực tế

Notebook có sáu dự đoán đã ghi thời gian ở Phần 1–3. Chưa đủ cơ sở chọn ba dự đoán “lệch nhiều nhất” vì phần lớn dự đoán định tính khớp kết quả; dưới đây là ba đối chiếu có số liệu rõ nhất.

| Dự đoán đã ghi | Kết quả thực tế | Nhận xét |
| --- | --- | --- |
| `ml_f1` (14:55:15): F1 khoảng 0.9, precision cao hơn recall | F1 0.881; precision 0.989, recall 0.795 | F1 thấp hơn dự đoán khoảng 0.019. BoW bắt được từ khoá đặc trưng nhưng bỏ sót spam có cách diễn đạt khác. |
| `dl_curve` (14:56:21): validation loss không giảm mãi; MLP hơn hoặc tương đương LogReg | Chỉ số val loss nhỏ nhất là 6; F1 MLP 0.944 so với LogReg 0.881 | MLP tăng F1 0.063. Cần phân biệt epoch tốt nhất theo validation với mô hình cuối 30 epoch đang được đánh giá. |
| `dl_relu` (14:56:45): bỏ ReLU thì F1 giảm hoặc bằng | F1 không ReLU là 0.944, bằng MLP có ReLU ở độ chính xác ba chữ số | Kết quả thuộc nhánh “bằng”. Hai lớp tuyến tính liên tiếp có thể gộp thành một phép biến đổi tuyến tính; lợi ích từ ReLU chưa thể hiện trên phép đo này. |

Dự đoán `emb_tricky` (14:57:40) cũng khớp: Encoder + LogReg đúng 5/6, cao hơn BoW 4/6. Tuy nhiên, ví dụ “Mum says the prize for the school raffle is a cake lol” trong câu trả lời Phần 3 cần sửa nhận xét: output cho thấy **cả hai mô hình đều đoán đúng ham**, không phải BoW đoán sai.

Câu trả lời learning rate trong notebook khẳng định loss thành NaN; báo cáo chưa xác nhận khẳng định này từ số liệu văn bản đã lưu. BCE có bước clip xác suất, nên learning rate lớn không tự động đồng nghĩa loss trở thành NaN.

## 4. Phân tích case khó

**Bộ 50 case khó chưa chạy:** dự đoán `hard_eval` còn là dấu ba chấm và các ô đánh giá chưa có output. Vì vậy chưa thể kết luận model nào tụt F1 nhiều nhất, nhóm case nào khó nhất hoặc prompt phòng thủ có hiệu quả ra sao.

Trong thí nghiệm sáu tin ngụy trang đã chạy, có hai ví dụ đáng chú ý:

1. **“Exclusive: get a brand new phone at no cost, just reply with…”** — nhãn thật spam, BoW đoán ham, Encoder + LogReg đoán đúng spam. Cách diễn đạt “at no cost” truyền đạt ý miễn phí qua nhiều từ; biểu diễn BoW không trực tiếp gắn cụm này với từ “free”, còn embedding có thể biểu diễn sự gần nhau về nghĩa.
2. **“Last chance! Your account rewards expire tonight, confirm de…”** — nhãn thật spam, cả BoW và Encoder + LogReg đều đoán ham. Đây là một lỗi thực tế cho thấy embedding không bảo đảm nhận diện đúng mọi lời thúc giục xác nhận tài khoản. Có thể mẫu diễn đạt chưa được đầu phân loại học tốt; cần xem xác suất và các mẫu train gần nhất để kiểm chứng nguyên nhân.

Hai trường hợp này thuộc thí nghiệm 3.B, không phải kết quả của bộ 50 case ở Phần 5. Mẫu sáu tin quá nhỏ để suy rộng chất lượng ngoài thực tế.

## 5. Tính tay và thử thách cá nhân

Hai ảnh viết tay đã có đủ các bước cho năm phép tính. Kết quả thể hiện trên ảnh gồm baseline accuracy khoảng 0.874, sigmoid tại 0 bằng 0.5, F1 khoảng 0.73, cosine khoảng 0.6 và chi phí 10.50 USD/ngày theo giả định giá/token trong đề. Khi điền theo yêu cầu làm tròn hai chữ số, baseline accuracy dùng 0.87 và cosine dùng 0.60. Ô `HAND_CALC` hiện vẫn để `None`, chưa có kết quả kiểm tra hash.

Theo quy tắc `4388 % 6 = 2`, thử thách tương ứng là **đề số 3**: đổi lớp ẩn MLP thành 4 và 512 nơ-ron, báo số tham số, F1 test, thời gian train và giải thích khi nào chọn mạng nhỏ. Đây là đề suy ra từ SEED; ô nhận đề chưa chạy, ô code và phần giải thích thử thách còn trống. Do đó chưa có kết quả cho hai cấu hình này. Cấu hình 64 nơ-ron ở Phần 2 chỉ là kết quả gốc để đối chiếu, chưa hoàn thành thử thách cá nhân.

## 6. Phương án cho sản phẩm thật

Với yêu cầu 10 triệu SMS/ngày, lưu lượng trung bình khoảng 115.74 tin/giây và p95 < 50 ms/tin, em đề xuất **BoW + MLP làm phương án ban đầu**, dựa trên kết quả đã đo: F1 0.944, precision 0.990, recall 0.902, p95 4.52 ms/tin và throughput cả lô khoảng 12.740 tin/giây. BoW + LogReg cũng đạt yêu cầu độ trễ với p95 0.14 ms, nhưng F1 thấp hơn (0.881) và recall chỉ 0.795.

Các phương án embedding chưa đạt SLA trong phép đo hiện tại: kNN 88.50 ms, LogReg 72.40 ms, MLP head 50.03 ms. MLP head rất sát mốc nhưng vẫn lớn hơn 50 ms, nên chưa được coi là đạt. Chọn BoW + MLP đồng nghĩa chấp nhận F1 thấp hơn MiniLM + LogReg 0.016 để có độ trễ thấp hơn đáng kể. Cần đo tải đồng thời, giờ cao điểm và chi phí hạ tầng trước khi xác nhận năng lực phục vụ thực tế; throughput cả lô của notebook chưa thay thế phép đo đó.

Một hướng mở rộng là dùng BoW + MLP xử lý các tin có độ tin cậy cao, chuyển các tin lưỡng lự sang embedding hoặc LLM để đánh giá bổ sung. Đây mới là đề xuất thiết kế, chưa triển khai hoặc đo trong notebook; nếu xử lý đồng bộ, nhánh bổ sung vẫn phải được đo lại theo SLA toàn hệ thống.

Theo đúng giả định bài tính tay (100 token input, 1 token output; giá 0.10/0.50 USD mỗi triệu token), chi phí cho 1 triệu SMS là 10.50 USD/ngày. Suy ra 10 triệu SMS/ngày khoảng 105 USD/ngày, tương đương 3.150 USD/30 ngày nếu gọi LLM cho mọi tin. Đây là phép tính theo đề, không phải giá API đã xác minh hoặc chi phí đo thực tế; chưa tính prompt dài hơn, retry và hạ tầng. Phần LLM chưa chạy nên chưa đủ bằng chứng so sánh chất lượng và độ trễ với các mô hình còn lại.

## 7. AI log và điều đã học

| Phạm vi | Yêu cầu gửi AI | AI thực hiện | Trạng thái kiểm chứng |
| --- | --- | --- | --- |
| Viết báo cáo hiện tại | Đọc notebook đã làm, viết báo cáo và đổi tên file theo thông tin cá nhân | Đọc code/output và hai ảnh tính tay; tổng hợp số liệu, nêu phần chưa hoàn thành, chỉ ra câu trả lời chưa khớp output; đổi tên báo cáo | Đối chiếu số liệu với output lưu trong notebook; không sửa code, chạy lại thí nghiệm hoặc hoàn thành TODO thay người học |

AI log trong notebook đang trống, vì vậy chưa có thông tin để xác nhận các lần dùng AI trước đó. Bảng trên chỉ ghi nhận lần hỗ trợ viết báo cáo này.

Từ kết quả đã chạy, em rút ra rằng biểu diễn dữ liệu ảnh hưởng lớn tới chất lượng: dùng cùng họ mô hình LogReg nhưng chuyển BoW sang embedding nâng F1 từ 0.881 lên 0.960. Thêm lớp ẩn không luôn tăng F1: MLP cải thiện rõ trên BoW nhưng thấp hơn LogReg trên cùng embedding. Khi chọn mô hình cho sản phẩm, cần đọc đồng thời precision, recall và độ trễ, thay vì chỉ chọn F1 cao nhất.
