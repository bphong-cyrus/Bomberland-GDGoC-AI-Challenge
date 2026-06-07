# PPO agent cho Bomberland — phá camping, đẩy lên top

Tài liệu này giải thích **vì sao agent DQN cũ camping ở góc**, **cách tiếp cận PPO mới sửa tận gốc**, và **quy trình train → kiểm tra → nộp**.

---

## ⭐ CẬP NHẬT 2026-06-07 — "BC active prior + bounded survival shield" (ĐÃ KIỂM CHỨNG)

Chẩn đoán lại từ chính 2 log train (`ppo_trend.log`, `ppo_bc.log`):

- **PPO from-scratch (không BC)** → CAMPING: `cells=4.3, boxes=0, items=0`. `avg_rank` đẹp là *lạc quan giả* (đối thủ tự chết).
- **PPO + BC** → ACTIVE nhưng RECKLESS: `cells=24, boxes=3.5, items=1.9` nhưng `survive=4/12, chết@step100`, vs hunter `survive=1/12`.

→ BC **đã sửa xong phần khó** (chịu chơi: farm + đi map). Phần thiếu **duy nhất** là **SỐNG SÓT khi đang active**. Theo TrueSkill, sinh tồn áp đảo → bản BC chết sớm nên thua dù farm giỏi.

**Nguyên nhân chết sớm = shield inference chỉ chặn chết 1 bước.** Sửa: nâng shield thành **BFS sinh-tồn giới hạn `horizon` bước** (mặc định 6) — đủ mạnh để chặn bẫy bom 2-3 bước, đủ "thoáng" để không gây camping (chính GeniusRuleAgent BFS sâu mà vẫn active). Dùng *y hệt* khi train và khi nộp.

**Kết quả kiểm chứng — CÙNG model BC `ckpts_ppo_bc`, chỉ đổi shield h=1 → h=6:**

| | h=1 (cũ) vs MIX | h=6 (mới) vs MIX | h=1 vs hunter | h=6 vs hunter |
|---|---|---|---|---|
| sống đến cuối | 5/16 | **11/16** | 6/16 | **11/16** |
| sole-win | 1/16 | **5/16** | 5/16 | **8/16** |
| items | 3.6 | **6.1** | — | — |

→ Sinh tồn gấp đôi, sole-win 5x, **farm NHIỀU hơn** (không camping). **Bạn có thể resubmit ngay model BC cũ với `ppo.py` mới để thắng nhanh**, rồi train lại bản đầy đủ.

**Lệnh train khuyến nghị (mới):**
```bash
python -m agent.dqn_agent.train_ppo \
    --bc_pretrain 1 --bc_games 400 --bc_epochs 5 \
    --iters 1500 --epi_per_iter 8 \
    --lr 2.5e-4 --clip 0.2 --epochs 4 --minibatch 1024 \
    --ent_start 0.03 --ent_end 0.005 \
    --lam_adv 0.05 --lam_e 0.1 \
    --safe_mask 1 --shield_horizon 6 \
    --learner_prob 0.5 --self_play_after 200 --snapshot_every 100 \
    --eval_every 100 --save_dir ckpts_ppo
```
Đọc dòng `[eval]` mới có thêm **`vs MIX`** (hunter+genius+tactical = giống leaderboard nhất): mục tiêu **`survive` cao + `avg_rank` thấp + `items≥3`**. Test trước khi nộp:
```bash
python -m agent.dqn_agent.test_behavior_ppo --model ckpts_ppo/.../model.pt --opponents mix --games 20 --shield_horizon 6
```

> Phần dưới là bối cảnh lịch sử (vẫn đúng), nhưng lưu ý mục 2 nói "bỏ mask khi học" đã được **thay** bằng "shield giới hạn 6 bước dùng cả khi train" như trên — vì lúc viết mục 2 chưa có BC active prior.

---

## 1. Chẩn đoán: vì sao DQN cũ "đi loanh quanh 1 góc đặt bom"

Đây **không phải lỗi thuật toán** mà là bẫy **"Reward Hacking"** (tài liệu BTC trang 11 cảnh báo đúng hiện tượng này). 3 nguyên nhân gốc:

1. **Safety-mask cứng** (`model.safe_action`) chỉ cho phép hành động *chắc chắn 100% sống sót*. → Mạng nơ-ron **không cần học chơi** (sống sót do mask lo), và lúc thi đấu mask khiến agent **cực kỳ rụt rè**, chỉ ở gần nhà nơi biết rõ đường thoát → camping. Tín hiệu `death=-3` (quan trọng nhất) gần như không bao giờ kích hoạt.
2. **Lỗi shaping ghim-góc**: `target_potential` cũ thưởng theo *độ gần hộp gần nhất*. Map cho mỗi spawn một ô an toàn 2×2 **bao quanh toàn hộp** → lúc spawn agent đã sát hộp → potential đã max → **không có lực kéo ra ngoài** (đi ra giữa map còn làm xa hộp → shaping âm).
3. **Reward bị chi phối bởi proxy** (shaping/novelty/step), ít gắn với rank thật; **eval dùng lại đúng baseline đang train** → điểm eval lạc quan giả. Kết quả leaderboard: `avg_rank≈1.55 ≈ ngẫu nhiên`.

## 2. Cách PPO mới sửa tận gốc

| Vấn đề | Cách sửa trong PPO |
|---|---|
| Mask gánh hết → không học | **Bỏ safety-mask khi học.** PPO *sample* hành động, **nếm cái chết trực tiếp** (death=-3), `entropy bonus` giữ thám hiểm → không sụp về 1 hành vi cố định. Lúc thi đấu chỉ giữ **shield tối thiểu** (né ô nổ *ngay step này*), rẻ hơn & ít rụt rè hơn BFS-mask nhiều. |
| Shaping ghim góc | **`advance_potential`**: kéo về *tâm bàn* (góc spawn ≈ 0 → mỗi bước ra giữa = thưởng dương) → bứt agent khỏi góc. **`enemy_potential`**: kéo về địch, scale theo sức mạnh (farm sớm, đánh muộn). |
| Reward lệch rank | Reward bám sát rank: `death −3`, `sole_winner +5`, `kill +2`, `box +0.5`, `item +1.0`, `opp_died +0.3`, `bomb_target +0.1`, `bomb_waste −0.05`, anti-camp `−0.05`. |
| Đối thủ yếu / eval ảo | League **nặng Hunter** (3/8) + snapshot self-play; eval theo **hành vi** (cells/boxes/items/survival) thay vì chỉ win-rate. |

## 3. File mới (đều ở `agent/dqn_agent/`)

- **`ppo.py`** — `PPOActorCritic` (TorchScript-able, dùng lại encoder & danger-model của `model.py`), `physical_action_mask` (chặn nước đi vô nghĩa), `shielded_action` (suy luận lúc thi đấu).
- **`train_ppo.py`** — trainer PPO: GAE, clipped objective, entropy schedule, reward sửa gốc, self-play league, eval hành vi.
- **`test_behavior_ppo.py`** — kiểm tra hành vi + heatmap (active hay camping).
- **`agent.py`** (đã cập nhật) — **tự nhận diện** model là PPO (output tuple `(logits,value)`) hay DQN (1 tensor Q) → nộp được cả hai. DQN cũ vẫn nguyên làm **fallback**.

> Encoder, `compute_danger`, `model.py`, `train.py` (DQN) **không đổi** — pipeline DQN cũ vẫn chạy y nguyên.

## 4. Train

### 4.1. Chạy thử cục bộ (kiểm tra code, vài phút)
```bash
# từ THƯ MỤC GỐC repo (không phải trong agent/)
python -m agent.dqn_agent.train_ppo --iters 30 --epi_per_iter 6 --max_steps 300 --eval_every 10
```

### 4.2. Train thật trên Kaggle/Colab (GPU)
```bash
python -m agent.dqn_agent.train_ppo \
    --iters 1500 --epi_per_iter 8 \
    --lr 2.5e-4 --clip 0.2 --epochs 4 --minibatch 1024 \
    --ent_start 0.03 --ent_end 0.005 \
    --lam_adv 0.1 --lam_e 0.1 \
    --learner_prob 0.5 --self_play_after 200 --snapshot_every 100 \
    --eval_every 100 --save_dir ckpts_ppo
```
- `--iters 1500 × --epi_per_iter 8 ≈ 12 000 episode`. Tăng `--iters` nếu còn thời gian/GPU.
- Mỗi `eval_every` sẽ in dòng `[eval]` (vs genius **và** vs hunter) + lưu `it{N}.pth` và `model.pt`.
- **Resume** (đủ state): `--resume ckpts_ppo/ppo_selfplay_1500it/it1000.pth`
- **Warm start** (chỉ weight, optimizer mới): `--warm_from <model.pt|.pth>`

### 4.3. Đọc log `[eval]` để biết có học không
Đúng hướng nếu qua các mốc eval: **`cells` tăng (≥15)**, **`boxes` tăng (≥3-5)**, **`items` tăng (≥2)**, `survive` cao, `avg_rank` giảm về 0. Nếu vẫn `cells≈5, boxes≈0` → xem mục Tuning.

## 5. Kiểm tra hành vi trước khi nộp
```bash
python -m agent.dqn_agent.test_behavior_ppo --model ckpts_ppo/ppo_selfplay_1500it/model.pt --opponents genius --games 20
python -m agent.dqn_agent.test_behavior_ppo --model ckpts_ppo/ppo_selfplay_1500it/model.pt --opponents hunter --games 20
```
Mục tiêu **VERDICT: BALANCED** + heatmap **trải đều khắp bàn** (không phải cụm `#` quanh `S`).

## 6. Đóng gói nộp (zip PHẲNG, ≤20 file)
Zip phải **phẳng** (không thư mục con), gồm đúng 4 file:
```
agent.py        (bản đã cập nhật, auto-detect)
model.py        (encoder/danger/DQN/safety-mask — ppo.py import từ đây)
ppo.py          (BẮT BUỘC cho model PPO)
model.pt        (TorchScript actor-critic vừa train, đổi tên từ ckpts_ppo/.../model.pt)
```
> Lưu ý: server gọi `agent.py`. KHÔNG có `import tqdm`/`matplotlib` ở mức module trong 4 file này (đã đảm bảo) — venv của bạn đang lỗi `colorama`/tqdm, nếu lỡ import ở submission sẽ fail lúc load.

## 7. Tuning (nếu chưa đạt)
- **Vẫn camping** (`cells<8, boxes≈0`): tăng `--lam_adv 0.15~0.2`, tăng `--ent_start 0.04`, kéo dài `--ent_end 0.01`, train thêm iters.
- **Liều, chết sớm** (`survive` thấp, `died_at<200`): giảm `--lam_e 0.05`, giảm `--ent_end 0.003`, hoặc tăng `--iters` (học né bom thêm).
- **Học chậm**: tăng `--epi_per_iter 12~16` (nhiều dữ liệu/update) hoặc `--epochs 6`.
- **KL quá lớn** (postfix `kl` > 0.03 thường xuyên): giảm `--lr` hoặc `--clip 0.15`.

## 8. Kỳ vọng thật lòng
Top hiện tại (Mu≈117, hàng trăm trận) rất mạnh. PPO + sửa gốc **chắc chắn phá được camping** và đẩy hạng lên đáng kể, nhưng *lọt top-100 không đảm bảo chỉ bằng 1 lần train* — cần lặp: train → `test_behavior_ppo` → chỉnh `lam_adv/lam_e/ent` → train lại. Nếu PPO chưa vượt, **DQN cũ vẫn là fallback** (nộp `model.pt` DQN + `agent.py` mới vẫn chạy).
