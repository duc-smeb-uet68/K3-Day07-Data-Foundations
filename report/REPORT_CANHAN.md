# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Chưa cung cấp
**Nhóm:** Chưa cung cấp
**Ngày:** 2026-08-03

> Báo cáo này được hoàn thiện theo code, dữ liệu và output chạy thực tế do cá nhân cung cấp. Phần benchmark đã chạy bằng local embedding và agent OpenRouter; kết quả, giới hạn của query có filter và một lỗi grounding ở query 5 được ghi rõ bên dưới.

**Cấu hình benchmark:** corpus `data/vinuni-course-registration-vi`, 13 chunk, `FixedSizeChunker(chunk_size=500, overlap=50)`, local embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, in-memory store và agent OpenRouter Chat Completions với model `openai/gpt-4o-mini`. Unit test/similarity prediction vẫn dùng `MockEmbedder`; không dùng mock để kết luận retrieval ngữ nghĩa.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Khi cosine similarity cao, hai vector embedding có hướng gần nhau; điều đó thường cho thấy hai câu có nội dung hoặc ý nghĩa gần nhau. Chỉ hướng của vector được so sánh nên độ dài tuyệt đối của embedding ít ảnh hưởng hơn.

**Ví dụ có độ tương tự CAO:**

- Câu A: “Sinh viên đăng ký học phần trên SIS.”
- Câu B: “Sinh viên thực hiện đăng ký môn học qua hệ thống SIS.”
- Tại sao tương đồng: Hai câu cùng nói về hành động đăng ký môn học của sinh viên trên SIS, chỉ khác cách diễn đạt.

**Ví dụ có độ tương tự THẤP:**

- Câu A: “Sinh viên đăng ký học phần trên SIS.”
- Câu B: “Thư viện cung cấp không gian học tập.”
- Tại sao khác: Một câu nói về đăng ký học phần, câu còn lại nói về dịch vụ thư viện; chủ đề và hành động chính khác nhau.

**Tại sao cosine similarity được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Embedding văn bản thường cần so sánh hướng biểu diễn ngữ nghĩa hơn là độ lớn vector. Cosine similarity giảm ảnh hưởng của việc một embedding có norm lớn hơn, trong khi khoảng cách Euclid có thể xem sự khác nhau về độ lớn là khác biệt ngữ nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

Với tài liệu dài 10.000 ký tự, `chunk_size=500`, `overlap=50`:

```text
số chunk = ceil((10.000 - 50) / (500 - 50))
         = ceil(9.950 / 450)
         = ceil(22,111...)
         = 23 chunk
```

**Đáp án:** 23 chunks.

Khi tăng overlap lên 100:

```text
số chunk = ceil((10.000 - 100) / (500 - 100))
         = ceil(9.900 / 400)
         = ceil(24,75)
         = 25 chunk
```

Số chunk tăng từ 23 lên 25 vì mỗi bước trượt chỉ còn 400 ký tự. Overlap lớn giúp giữ ngữ cảnh ở ranh giới giữa hai chunk, nhưng làm tăng số chunk, chi phí embedding/lưu trữ và mức lặp nội dung.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### `SentenceChunker.chunk`

Tôi dùng regex `(?<=[.!?])\s+|\n+` để phát hiện ranh giới sau `.`, `!`, `?` khi có whitespace và tại xuống dòng. Dấu câu được giữ lại trong câu, sau đó mỗi câu được `strip()` và các phần rỗng bị loại bỏ. Input rỗng hoặc chỉ có whitespace trả về `[]`; `max_sentences_per_chunk` được chặn tối thiểu ở 1 rồi nhóm tối đa số câu tương ứng vào mỗi chunk.

### `RecursiveChunker.chunk` / `_split`

Thuật toán thử separator theo thứ tự ưu tiên `['\n\n', '\n', '. ', ' ', '']`. Trường hợp cơ sở là text rỗng hoặc đã không dài hơn `chunk_size`; khi đó trả về một chunk. Nếu separator không xuất hiện, thuật toán chuyển sang separator kế tiếp; nếu hết separator hoặc gặp separator rỗng thì dùng `_hard_split()` theo ranh giới ký tự. Các phần nhỏ được gom vào buffer nếu vẫn nằm trong giới hạn, còn phần lớn được đệ quy xử lý với các separator thấp hơn.

### `EmbeddingStore.add_documents` + `search`

Mỗi `Document` được chuyển thành record gồm `id`, `content`, bản sao `metadata` và embedding tạo từ `content`. In-memory list là backend bắt buộc và cũng là nguồn dữ liệu chính; ChromaDB chỉ được dùng như mirror tùy chọn khi dependency có sẵn. Khi search, query được embed, tính dot product với từng embedding, sắp xếp score giảm dần và cắt còn tối đa `top_k` kết quả.

### `search_with_filter` + `delete_document`

`search_with_filter()` lọc metadata trước khi tính similarity. Một record chỉ được giữ lại khi mọi cặp key/value trong `metadata_filter` khớp chính xác. `delete_document()` xóa mọi chunk có `metadata['doc_id']` bằng `doc_id` yêu cầu; code cũng hỗ trợ xóa record có `id` trùng trực tiếp và trả `True` nếu có record bị xóa, ngược lại trả `False`.

### `KnowledgeBaseAgent.answer`

Agent gọi `store.search(question, top_k)`, ghép các chunk thành context có nhãn `[Chunk <id>]`, rồi tạo prompt gồm hướng dẫn chỉ dùng context, câu hỏi và phần context truy xuất. Nếu context không đủ, prompt yêu cầu LLM nói rõ thay vì đoán. Hàm `llm_fn` được inject từ bên ngoài nên có thể dùng LLM thật hoặc stub deterministic trong test; agent không tự sinh câu trả lời khi chưa gọi `llm_fn`.

### Chiến lược cá nhân và baseline

Chiến lược đã kiểm tra trong pipeline là fixed-size chunking với `chunk_size=500`, `overlap=50`, giữ toàn bộ metadata front matter trên từng chunk. Đây là cấu hình mặc định của `build_knowledge_base()` và phù hợp để kiểm tra pipeline end-to-end; chưa có đủ kết quả local embedding để khẳng định đây là chiến lược tốt nhất về ngữ nghĩa.

Baseline được chạy bằng `ChunkingStrategyComparator().compare(..., chunk_size=200)` trên 3 tài liệu đầu tiên của corpus VinUni:

| Tài liệu | Chiến lược | Số chunk | Độ dài trung bình | Nhận xét về ngữ cảnh |
|---|---:|---:|---:|---|
| `vinuni-course-registration-guide` | fixed_size | 7 | 185,00 | Kích thước ổn định nhưng có thể cắt giữa heading/list. |
|  | by_sentences | 10 | 127,80 | Giữ ranh giới câu tốt, nhưng tạo nhiều chunk ngắn hơn. |
|  | recursive | 11 | 116,09 | Ưu tiên đoạn/dòng/câu; mạch lạc hơn ở ranh giới cấu trúc nhưng kích thước biến thiên. |
| `vinuni-registrar-faqs-vi` | fixed_size | 5 | 191,00 | Gọn và đều, có nguy cơ cắt giữa các ý FAQ. |
|  | by_sentences | 7 | 133,29 | Giữ câu hoàn chỉnh. |
|  | recursive | 6 | 156,17 | Cân bằng tốt hơn giữa số lượng và độ dài trong tài liệu này. |
| `vinuni-spring-2026-important-notice` | fixed_size | 5 | 183,80 | Dễ kiểm soát giới hạn ký tự. |
|  | by_sentences | 6 | 150,67 | Giữ được các câu điều kiện/thời hạn. |
|  | recursive | 8 | 112,88 | Giữ cấu trúc đoạn nhưng tạo nhiều chunk nhỏ. |

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết quả kiểm thử

> Lệnh đã chạy:

```powershell
python -m pytest tests/ -v
```

Kết quả tóm tắt:

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                   [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                            [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                     [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                      [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                           [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                           [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                 [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                  [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                  [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                  [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                             [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                         [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                   [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                          [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                              [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                        [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                              [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                  [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                    [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                      [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                            [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                 [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                   [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                       [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                    [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                             [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                            [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                       [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                   [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                              [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                  [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                        [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                  [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                               [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                             [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                            [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                           [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                    [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                          [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                              [100%]

============================================================= 42 passed in 0.14s =============================================================
```

**Số lượng bài test vượt qua:** **42 / 42**.

Các nhóm test đã pass gồm chunking fixed-size/sentence/recursive, embedding store, top-k search, metadata filter, delete theo document, cosine similarity, comparator và agent.

### Kiểm tra ingest và agent

```powershell
$env:PYTHONUTF8='1'; python ingest.py
```

Kết quả:

```text
ingest self-check OK: parse được 4 khóa metadata, tạo 18 chunk (mỗi chunk giữ doc_id + metadata).
```

Smoke test với corpus VinUni nạp được **13 chunk**. `KnowledgeBaseAgent` với `llm_fn` stub trả về `STUB_OK`; prompt đã kiểm tra có cả câu hỏi và marker `Retrieved context:`.

**Lưu ý môi trường:** lệnh đang chạy bằng Python 3.13.1, trong khi chuẩn lab là Python 3.11. `.venv\Scripts\python.exe` hiện trỏ tới Python 3.11 tại đường dẫn không còn tồn tại, vì vậy chưa thể xác nhận lại bằng đúng interpreter 3.11 trong môi trường hiện tại. Đây là giới hạn môi trường, không phải test failure của code.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Các score dưới đây được tính bằng `compute_similarity(MockEmbedder()(A), MockEmbedder()(B))`. Theo hướng dẫn K3, các score mock chỉ dùng để kiểm tra pipeline/công thức, không dùng để kết luận hai câu tiếng Việt có thực sự gần nghĩa.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (mock) | Đúng? |
|---:|---|---|---|---:|---|
| 1 | Sinh viên đăng ký học phần trên SIS. | Sinh viên thực hiện đăng ký môn học qua hệ thống SIS. | cao | 0,057612 | Không thể kết luận semantic; đây là score cao nhất trong 5 cặp. |
| 2 | Hạn cuối để Drop môn là ngày 13/03/2026. | Cổng đăng ký mở từ 14:00 ngày 18/12/2025. | thấp | 0,026445 | Tương đối phù hợp với dự đoán thấp. |
| 3 | Sinh viên phải kiểm tra điều kiện tiên quyết. | Thư viện cung cấp không gian học tập. | thấp | -0,126321 | Phù hợp về hướng score, nhưng không phải bằng chứng semantic đáng tin. |
| 4 | Add và Drop chỉ được thực hiện trong thời gian chính thức. | Các thay đổi trong giai đoạn Add/Drop không ghi nhận trên bảng điểm. | cao | -0,087250 | Không phù hợp với dự đoán. |
| 5 | Sinh viên toàn thời gian thường đăng ký khoảng 30 tín chỉ mỗi năm. | Khối lượng toàn thời gian thông thường là 30 tín chỉ mỗi năm. | cao | -0,169578 | Bất ngờ nhất; câu gần như đồng nghĩa nhưng mock cho score thấp nhất. |

Kết quả bất ngờ nhất là cặp 5: hai câu diễn đạt gần như cùng một ý nhưng score âm và thấp nhất. Điều này xác nhận `MockEmbedder` sinh vector dựa trên chuỗi, gần như ngẫu nhiên theo text, nên không biểu diễn tốt quan hệ ngữ nghĩa tiếng Việt. Muốn kết luận về retrieval cần chạy lại cùng 5 cặp bằng local model `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy 5 câu hỏi đánh giá của nhóm trên mã nguồn cá nhân trong gói `src`. 5 câu hỏi này trùng với các thành viên cùng nhóm (xem `REPORT_NHOM.md`). Backend sử dụng local embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; agent sử dụng OpenRouter Chat Completions với model `openai/gpt-4o-mini`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---:|---|---|---:|---|---|
| 1 | Các bước đăng ký học phần trên SIS là gì? | `vinuni-course-registration-guide.md`: các bước vào SIS, chọn `Course Registration`, chọn lớp `Open`, `Add`, `Register`. | 0,790 | Có | Đăng nhập SIS → `Academics` → `Course Registration` → chọn học kỳ → chọn lớp `Open` → `Add` → `Register` → kiểm tra `Registered` và `Your Class Schedule`. |
| 2 | Hạn cuối để Add môn và Drop môn trong học kỳ Spring 2026 là khi nào? | `vinuni-spring-2026-registration-announcement.md`: lịch và hạn Add/Drop Spring 2026. | 0,788 | Có | Hạn cuối Add là 06/03/2026; hạn cuối Drop là 13/03/2026. |
| 3 | Khối lượng học tập toàn thời gian thông thường là bao nhiêu tín chỉ mỗi năm? | `vinuni-undergraduate-academic-regulations-vi.md`: khối lượng toàn thời gian thông thường. | 0,684 | Có | Khối lượng học tập toàn thời gian thông thường là 30 tín chỉ mỗi năm. |
| 4 | Nếu học phần đã đăng ký trên SIS nhưng không hiển thị trên Canvas thì phải làm gì? | `vinuni-spring-2026-important-notice.md`: hướng dẫn xử lý khi SIS và Canvas không đồng bộ. | 0,623 | Có | Cần báo sớm cho Phòng Quản lý Đào tạo. |
| 5 | Sau mốc nào yêu cầu rút môn Spring 2026 không còn được chấp nhận? | `vinuni-spring-2026-registration-announcement.md`: thông tin về hạn Drop; nguồn điều kiện 30% nằm ở top-2. | 0,700 | Có một phần; chunk đầy đủ ở top-2 | Agent trả lời sau hạn cuối 13/03/2026, nhưng chưa nêu điều kiện sau khi hoàn thành quá 30% thời lượng học tập. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5**

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> Local multilingual embedding cho kết quả semantic rõ ràng hơn mock; 5/5 câu hỏi có nguồn liên quan trong top-3 nhưng top-1 vẫn có thể lệch. Query 5 cho thấy agent có thể lấy đúng chủ đề nhưng bỏ sót điều kiện quan trọng, vì vậy cần đánh giá cả độ đầy đủ của câu trả lời chứ không chỉ nhìn score retrieval.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 — 4 câu đầy đủ, 1 câu agent trả lời thiếu điều kiện |
| **Tổng phần cá nhân** | **59 / 60** |

### Các thông tin cá nhân còn thiếu

- Họ tên sinh viên và tên nhóm chưa có trong workspace, cần bổ sung trước khi nộp.
- Nhóm vẫn cần xác nhận chính thức bộ 5 query/gold answer trong `REPORT_NHOM.md`.
- Cần chạy bổ sung Q2 với metadata filter và sửa/đánh giá lại Q5 nếu muốn agent trả lời đầy đủ điều kiện 30%.
