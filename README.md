# 🚀 LunarLander-v3 với DQN & Double DQN

## 📑 Mục lục
- [🚀 LunarLander-v3 với DQN \& Double DQN](#-lunarlander-v3-với-dqn--double-dqn)
  - [📑 Mục lục](#-mục-lục)
  - [🔍 Tổng quan dự án](#-tổng-quan-dự-án)
  - [🌌 Môi trường](#-môi-trường)
  - [🧠 Thuật toán](#-thuật-toán)
    - [✅ DQN (Deep Q-Network)](#-dqn-deep-q-network)
    - [✅ Double DQN](#-double-dqn)
  - [🧪 Kiểm tra phiên bản thư viện](#-kiểm-tra-phiên-bản-thư-viện)
  - [📊 Kết quả huấn luyện](#-kết-quả-huấn-luyện)
  - [📽️ Video mô phỏng mô hình (Double DQN)](#️-video-mô-phỏng-mô-hình-double-dqn)
  - [🔮 Cải tiến trong tương lai](#-cải-tiến-trong-tương-lai)
    - [🚀 1. Sử dụng thuật toán nâng cao hơn](#-1-sử-dụng-thuật-toán-nâng-cao-hơn)
    - [🧠 2. Cải thiện kiến trúc mạng học sâu](#-2-cải-thiện-kiến-trúc-mạng-học-sâu)
    - [🧮 3. Hỗ trợ huấn luyện song song hoặc phân tán](#-3-hỗ-trợ-huấn-luyện-song-song-hoặc-phân-tán)
    - [🎯 4. Mở rộng sang môi trường phức tạp hơn](#-4-mở-rộng-sang-môi-trường-phức-tạp-hơn)
    - [🛠 5. Cải thiện phần hiển thị và phân tích](#-5-cải-thiện-phần-hiển-thị-và-phân-tích)
  - [📚 Tài liệu tham khảo](#-tài-liệu-tham-khảo)
    - [📄 Bài báo khoa học \& lý thuyết](#-bài-báo-khoa-học--lý-thuyết)
    - [📚 Tài liệu kỹ thuật \& thư viện](#-tài-liệu-kỹ-thuật--thư-viện)
    - [🎥 Học thông qua thực hành \& nguồn mở](#-học-thông-qua-thực-hành--nguồn-mở)

---

## 🔍 Tổng quan dự án

Dự án này triển khai và so sánh hai thuật toán học tăng cường phổ biến: **DQN (Deep Q-Network)** và **Double DQN** trên môi trường **LunarLander-v3** từ OpenAI Gym. Mục tiêu là giúp agent điều khiển tàu đổ bộ một cách tối ưu và an toàn, đạt được điểm số cao nhất có thể.

---

## 🌌 Môi trường

Chúng tôi sử dụng **LunarLander-v3**, một môi trường mô phỏng việc đổ bộ tàu vũ trụ:

- Quan sát: vector 8 chiều (vị trí, vận tốc, góc, tiếp xúc mặt đất)
- Hành động: 4 hành động rời rạc (không làm gì, đẩy động cơ chính, trái, phải)
- Phần thưởng:
  - Thưởng cho hạ cánh mềm
  - Phạt nếu rơi mạnh hoặc rơi ngoài khu vực đích
- Kết thúc khi:
  - Agent hạ cánh thành công
  - Ra khỏi khung hình hoặc hết bước

---

## 🧠 Thuật toán

### ✅ DQN (Deep Q-Network)

- Sử dụng mạng neural để ước lượng hàm Q.
- Cập nhật giá trị Q thông qua Bellman Equation.
- Gặp vấn đề **overestimation bias** vì sử dụng cùng một mạng để chọn và đánh giá hành động.

### ✅ Double DQN

- Giảm thiểu overestimation bằng cách:
  - Dùng **mạng chính** để chọn hành động tối ưu.
  - Dùng **mạng mục tiêu** để đánh giá hành động đó.
- Thường ổn định và chính xác hơn DQN cơ bản.

## 🧪 Kiểm tra phiên bản thư viện

Để đảm bảo dự án hoạt động đúng cách ( tất cả có trong requirement.txt ), sử dụng chính xác các phiên bản thư viện đã được kiểm chứng bên dưới:

| 📦 Thư viện        | ✅ Phiên bản tương thích |
|--------------------|--------------------------|
| `gymnasium`        | `1.1.1`                  |
| `torch`            | `2.5.1+cu121`            |
| `matplotlib`       | `3.10.1`                 |
| `ipython`          | `9.1.0`                  |

---

## 📊 Kết quả huấn luyện
Với thuật toán DQN, tỷ lệ hạ cánh thành công của tàu vũ đạt mức khá cao với 95.6%, Double-DQN phiên bản cải tiến của thuật toán DQN đạt tỷ lệ thành công tới 99.7%. Với thuật toán DQN, tỷ lệ hạ cánh thành công của tàu vũ trụ đạt mức khá cao với 95.6%, trong khi Double-DQN – phiên bản cải tiến của DQN – đạt tỷ lệ thành công lên tới 99.7%. Điều này cho thấy rằng cả hai thuật toán đều có khả năng học chính sách hiệu quả trong môi trường LunarLander-v3. Tuy nhiên, Double DQN vượt trội hơn nhờ khả năng giảm thiểu hiện tượng quá ước lượng giá trị Q (Q-value overestimation), vốn là một điểm yếu cố hữu của DQN thuần túy.Double DQN không chỉ cải thiện hiệu suất mà còn đảm bảo sự ổn định lâu dài cho quá trình học trong môi trường giả lập hạ cánh này. Nó là lựa chọn ưu tiên khi triển khai các bài toán có không gian hành động rời rạc và yêu cầu học chính sách tối ưu ổn định đặc biệt với môi trường Lunar-Lander-v3.  

## 📽️ Video mô phỏng mô hình 
- 🎬 **Nội dung**: Video mô tả quá trình hạ cánh thành công của tàu vũ trụ. 
- 📺 **Xem video**:  
👉 [📽️ Nhấn để xem video](lunar_lander_video.mp4)

---

## 🔮 Cải tiến trong tương lai

Dù mô hình Double DQN đã đạt kết quả rất khả quan với môi trường LunarLander-v3, vẫn còn nhiều hướng phát triển để nâng cao hiệu suất, độ ổn định và khả năng mở rộng:

### 🚀 1. Sử dụng thuật toán nâng cao hơn
- ✅ **Dueling DQN**: Tách biệt giá trị trạng thái và giá trị hành động để đánh giá hiệu quả hơn.
- ✅ **Prioritized Experience Replay**: Ưu tiên các trải nghiệm có ảnh hưởng lớn đến quá trình học.
- ✅ **Noisy Nets**: Thêm nhiễu vào mạng Q để cải thiện chiến lược thăm dò (exploration).

### 🧠 2. Cải thiện kiến trúc mạng học sâu
- 📈 Tăng số lớp hoặc đơn vị ẩn để học các biểu diễn phức tạp hơn.
- 🧪 Thử nghiệm với các activation function khác như Swish, Mish thay cho ReLU.

### 🧮 3. Hỗ trợ huấn luyện song song hoặc phân tán
- 💻 Tận dụng nhiều GPU hoặc multi-threading để tăng tốc huấn luyện.
- ☁️ Kết hợp với nền tảng như Ray RLlib hoặc các thư viện phân tán.

### 🎯 4. Mở rộng sang môi trường phức tạp hơn
- 🌍 Áp dụng với các môi trường khác như:
  - `BipedalWalker-v3`
  - `CarRacing-v2`
  - Môi trường 3D hoặc điều khiển robot thực tế (sim2real)

### 🛠 5. Cải thiện phần hiển thị và phân tích
- 📊 Giao diện trực quan hóa các bước huấn luyện (tensorboard hoặc matplotlib nâng cao).
- 🎥 Tự động ghi lại quá trình huấn luyện thành video `.mp4` sau mỗi n episode.
- 📈 Biểu đồ so sánh DQN vs Double DQN theo reward, tốc độ hội tụ, v.v.

---
## 📚 Tài liệu tham khảo

Dưới đây là một số tài liệu và nguồn học thuật uy tín đã được tham khảo trong quá trình phát triển dự án:

### 📄 Bài báo khoa học & lý thuyết

- 📘 **Mnih et al. (2015)** — *"Human-level control through deep reinforcement learning"*  
  [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

- 📘 **Van Hasselt et al. (2016)** — *"Deep Reinforcement Learning with Double Q-learning"*  
  [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)

- 📘 **Schaul et al. (2016)** — *"Prioritized Experience Replay"*  
  [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)

### 📚 Tài liệu kỹ thuật & thư viện

- 🧠 **OpenAI Gymnasium Documentation**  
  [https://gymnasium.farama.org](https://gymnasium.farama.org)

- 🔧 **PyTorch Official Docs**  
  [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

- 📈 **Matplotlib for Animation**  
  [https://matplotlib.org/stable/api/animation_api.html](https://matplotlib.org/stable/api/animation_api.html)

### 🎥 Học thông qua thực hành & nguồn mở

- 📺 **Deep Reinforcement Learning Nanodegree (Udacity)**  
  [https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

- 💻 **GitHub: DQN & RL Repositories (chọn lọc)**  
  - https://github.com/higgsfield/RL-Adventure
  - https://github.com/dennybritz/reinforcement-learning

---

> 🔎 *Tất cả tài liệu trên đã được chọn lọc nhằm đảm bảo độ tin cậy và hỗ trợ tối ưu cho quá trình triển khai và cải tiến mô hình.*
