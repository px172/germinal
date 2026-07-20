# Germinal — upstream 相容性問題與修補記錄

本文件記錄在此 fork 中發現並修補的 upstream 問題，作為 fork 內部參考
（不對 upstream 提交）。對應的程式修改見 branch `fix/upstream-compat`。

基準：commit `1e1c1a5`（Include ablang2 and ipsae dependencies in Dockerfile, #60）
安裝方式：完全照 `environment_setup.md` 步驟 1–6，Python 3.10.20、Linux、CUDA 12.8 (A6000)
發現日期：2026-07（版本解析結果見各 issue）

Issue 1–4 已在本 fork 修補；Issue 5（chai）僅診斷、以改用 protenix 繞過。

以下三個問題都是「照文件全新安裝就會踩到」的，因為 `environment_setup.md`
和 `Dockerfile` 都沒有 pin 相關版本，上游釋出新版後行為就變了。

---

## Issue 1: PyRosetta ≥2026 破壞 `score_interface`，最終過濾階段必定 crash

### 摘要

`InterfaceAnalyzerMover.set_interface()` 在新版 PyRosetta 改為只接受
`DockingPartners` 物件，不再接受字串。`germinal/filters/pyrosetta_utils.py:109`
傳的是字串 `"A_B"`，因此任何設計只要通過前面的 gate、進入最終過濾階段就會 `TypeError`。

### 環境

- PyRosetta `2026.28+release.188eabbe2c00638eb408e86cbd67b1fec355b9c7`
  （由 `pyrosetta_installer.install_pyrosetta()` 安裝，即 `environment_setup.md`
  步驟 3 與 `Dockerfile:70-71` 的預設行為，未指定版本）

### 重現

```python
import pyrosetta as pr
pr.init('-ignore_unrecognized_res -load_PDB_components false -mute all '
        '-dalphaball params/DAlphaBall.gcc')
from germinal.filters.pyrosetta_utils import score_interface
score_interface("some_complex.pdb", binder_chain="B", target_chain="A")
```

```
File "germinal/filters/pyrosetta_utils.py", line 109, in score_interface
    iam.set_interface("A_B")
TypeError: set_interface(): incompatible function arguments. The following argument types are supported:
    1. (self: ...InterfaceAnalyzerMover, interface: ...core.pose.DockingPartners) -> None

Invoked with: <...InterfaceAnalyzerMover object at 0x...>, 'A_B'
```

在完整 pipeline 中，這會在 trajectory 通過 hallucination 與 clash 檢查、
進入最終過濾階段時才觸發，因此可能跑了數十分鐘才失敗。

### 影響

`environment_setup.md` 未指定 PyRosetta 版本，現在照文件安裝一定拿到新版，
最終過濾階段 100% 失敗 → 永遠無法產出 accepted design。

### 建議修法

新版提供 `DockingPartners.docking_partners_from_string()`，可同時相容新舊版：

```python
iam = InterfaceAnalyzerMover()
# PyRosetta >=2026 takes a DockingPartners object; older releases take a string.
try:
    from pyrosetta.rosetta.core.pose import DockingPartners
    iam.set_interface(DockingPartners.docking_partners_from_string("A_B"))
except (ImportError, TypeError):
    iam.set_interface("A_B")
```

實測套用後 `score_interface` 正常回傳 16 項指標
（dSASA 2697.32、interface_nres 24、interface_sc 0.59、packstat 0.55 等）。

只有這一個 call site 受影響（`grep -rn "set_interface" germinal/` 僅一筆）。

---

## Issue 2: transformers 5.x 讓 IgLM tokenizer 靜默失效，過濾階段 assert 失敗

### 摘要

`transformers` 5.x 把 `BertTokenizerFast` 的 `vocab_file=` kwarg 改名為 `vocab=`，
且舊名**不再報錯、改為靜默忽略**。`iglm` 套件內部
（`iglm/model/IgLM.py:32`）仍用 `vocab_file=`，導致 tokenizer 只剩 5 個特殊 token，
所有胺基酸與 `[HEAVY]`/`[HUMAN]` 等 token 全部變成 `[UNK]`。

`germinal/filters/filter_utils.py:842` 的 `model.log_likelihood(...)` 因此 assert 失敗。

### 環境

- `transformers` 5.14.1（自動解析而來，見下方依賴鏈）
- `iglm` 0.1.0

依賴鏈：`colabdesign/setup.py:13` 依賴 `iglm` → `iglm` 宣告
`transformers (>=4.6.1)`（**無上限**）→ 解析到 5.x。
`environment_setup.md` 與 `Dockerfile` 皆未 pin `transformers`。

### 重現

```python
import transformers
from iglm.model.IgLM import VOCAB_FILE
for kw in ("vocab", "vocab_file"):
    t = transformers.BertTokenizerFast(**{kw: VOCAB_FILE}, do_lower_case=False)
    print(kw, t.vocab_size, t.convert_tokens_to_ids("[HEAVY]"))
```

transformers 5.14.1 的輸出：

```
vocab      33 31     <- 正確
vocab_file  5  1     <- 全部變 [UNK]，且不報錯
```

完整 pipeline 中的失敗（trajectory 已通過 hallucination + clash + cofolding）：

```
File "germinal/filters/filter_utils.py", line 842, in get_iglm_ll
    log_likelihood = model.log_likelihood(sequence, chain_token, species_token)
File ".../iglm/model/IgLM.py", line 151, in log_likelihood
    assert (token_seq != self.tokenizer.unk_token_id
AssertionError: Unrecognized token supplied in starting tokens
```

另一個早期徵兆是啟動時的警告（vocab 只剩 5 個 token 時出現）：

```
[transformers] Model config: bos_token_id must be `None` or an integer within
the vocabulary (between 0 and 32), got 50256.
```

### 值得注意：hallucination 路徑不受影響

`colabdesign/colabdesign/iglm/model.py:44-46` 已經有相容寫法，先試 `vocab=`
再回退 `vocab_file=`，所以 `CustomIgLM`（hallucination 的梯度導引）在
transformers 5.x 下是正常的，`lm_ll` 數值有效。

換句話說 repo 內已經存在這個相容 shim，只是 `iglm` 套件自己的 `IgLM` 類別沒有，
而過濾路徑走的正是後者。

### 建議修法

擇一：

1. **pin 版本**（我採用的方式，不動程式碼）：
   在 `environment_setup.md` / `Dockerfile` / `setup.py` 加上 `transformers<5`。
   實測 4.57.6 下兩條路徑都正常：
   - `get_iglm_ll(...)` → `-1.4909`
   - `CustomIgLM` → `vocab_size 33, chain_id 31, species_id 26`
   另外 `chai_lab` 完全沒有 import `transformers`，降版無連帶影響
   （降版會一併把 `huggingface-hub` 拉到 0.36.2，實測 chai 推論仍正常）。

2. **在 `get_iglm_ll` 內加 shim**，沿用 `colabdesign/iglm/model.py` 已有的寫法，
   建立 `IgLM()` 後覆寫其 tokenizer。好處是不限制 transformers 版本。

---

## Issue 3: `seq_entropy_threshold=0` 必定 crash，且失敗訊息方向相反

### 3a. `0` 應代表停用，實際會 TypeError

`germinal/design/design.py:154-157` 明確把 `0` 轉成 `None` 並印出
"Sequence entropy threshold set to 0, disabling filter"，但後續兩處比較未防 `None`：

- `design.py:286`：`af_model._tmp["best"]["mean_soft_pseudo"] < seq_entropy_threshold`
- `design.py:317`：`af_model._tmp["best"].get("mean_soft_pseudo", 1) >= seq_entropy_threshold`

重現：

```bash
python run_germinal.py run=vhh target=il3 seq_entropy_threshold=0
```

```
File "germinal/design/design.py", line 286, in germinal_design
    and af_model._tmp["best"]["mean_soft_pseudo"] < seq_entropy_threshold
TypeError: '<' not supported between instances of 'float' and 'NoneType'
```

值得一提的是 `colabdesign/colabdesign/af/design.py:464` 已經正確地用
`seq_entropy_threshold is not None` 做了防護，可見 `None` 確實是設計上的停用哨兵值，
只是 `germinal/design/design.py` 這兩處漏了。

建議修法：

```python
# line 286
if (
    clear_best
    and seq_entropy_threshold is not None
    and af_model._tmp["best"]["mean_soft_pseudo"] < seq_entropy_threshold
):

# line 317
and (
    seq_entropy_threshold is None
    or af_model._tmp["best"].get("mean_soft_pseudo", 1) >= seq_entropy_threshold
)
```

### 3b. 失敗訊息把條件方向講反了

`design.py:344`：

```python
"Softmax trajectory metrics too low or sequence entropy too high to continue: "
```

但 line 317 的通過條件是 `mean_soft_pseudo >= seq_entropy_threshold`，
也就是熵**低於**門檻才失敗。訊息說 "too high" 與實際相反。

實例：熵 0.0658、門檻 0.10 時失敗，訊息卻說 entropy too high。
建議改為 "sequence entropy too low"。

（附帶一提，`design.py:289` 的 "Clearing best model due to low sequence entropy"
方向是對的，所以同一份程式碼裡兩處敘述互相矛盾，容易誤導。）

---

## Issue 4: structure_model=protenix 不 dump full_data，pdockq2 恆為 None，永遠無法產出設計

### 摘要

`run_protenix()` 組出的 `protenix pred` 指令沒有帶 `--need_atom_confidence`
（預設 False）。Protenix 在此設定下只寫出 `*_summary_confidence_*.json` 與 CIF，
**不寫** 含 `token_pair_pae` 的 `*_full_data_*.json`。`extract_protenix_scores`
需要那個 PAE matrix 才能算 pdockq2；找不到檔案時 ipsae 為 None、pdockq2 為 None。
最終過濾器要求 `pdockq2 > 0.23`，於是每個 redesign 序列都失敗，永遠 0 accepted
——與 chai 路徑（Issue 5，見下）相同的終局，只是走另一條分支。

### 環境

- Protenix 2.0.0（`configs/run/vhh.yaml` 預設 `structure_model: "protenix"` 之一）
- 由 `germinal/filters/protenix.py` 透過 `conda run -n protenix protenix pred ...`
  子程序呼叫

### 根因

`runner/dumper.py`（protenix 套件）只有在 `need_atom_confidence=True` 時才 dump
full_data：

```python
# runner/dumper.py
if self.need_atom_confidence:
    output_fpath = f"{sample_name}_full_data_sample_{rank}.json"
    save_json(data["full_data"][idx], output_fpath, indent=None)
```

`germinal/filters/protenix.py` 的 Popen 指令未傳此 flag：

```python
run_cmds = [
    "conda", "run", "-n", conda_env, "--no-capture-output",
    "protenix", "pred",
    "-i", input_path, "-o", str(output_dir),
    "-s", seeds_str, "-n", model_name,
    "-e", str(n_samples), "-c", str(n_cycles), "-p", str(n_steps),
    "--use_msa", str(use_msa),
    # 缺 --need_atom_confidence True
]
```

### 症狀

`extract_protenix_scores` 找得到 summary/CIF、但推導出的 `_full_data_` 檔不存在：

```
ipSAE skipped: full_data exists=False, pae_matrix size=1
Warning: PAE matrix not available, using ipsae metrics for pDockQ2/LIS
```

在完整 pipeline 中，任何走到最終過濾器的設計都會失敗於 `pdockq2 > 0.23`。
本現象只在 trajectory 通過 hallucination + clash + initial cofolding 後才顯現，
因此可能跑數十分鐘後才發現。

### 影響

`structure_model=protenix` 是 `configs/run/vhh.yaml` 官方列出的三個選項之一，
但現況下用它跑完整 pipeline 100% 產不出 accepted design。

### 建議修法

在 Popen 指令加 `--need_atom_confidence True`。`get_clean_full_confidence` 只移除
`atom_coordinate` / `atom_is_polymer`，`token_pair_pae` 會保留：

```python
    "--use_msa", str(use_msa),
    "--need_atom_confidence", "True",
]
```

實測驗證（transformers 4.x 環境、Protenix 2.0.0）：
- 單獨呼叫加 flag 後 `_full_data_sample_0.json` 生成，含 `token_pair_pae`
  shape (243,243)。
- 完整 pipeline run 中 "ipSAE skipped" 警告消失，走到過濾器的 trajectory 記錄到
  真實 pdockq2（0.0087，與 ipsae_pdockq2 一致）而非 None。
- forward pass 成本不變，只多 dump confidence 檔。

### 附註：external validation 是有效的（非 bug）

修好 pdockq2 後，30×2 條 trajectory 仍 0 accepted，但這是設計品質問題不是 bug：
走到最終過濾器的候選其 protenix 獨立重折的 `external_iptm` 僅 0.12–0.15（AF2
hallucination 自評 0.7–0.83），pdockq2 因此正確地偏低。這正是 external
validation filter 設計要抓的「AF2 對自己優化序列過度樂觀」現象，代表 pipeline
運作正常。

---

## Issue 5: structure_model=chai 從不設定 ipsae，pdockq2 恆為 None（與 Issue 4 平行）

### 摘要

`run_structure_prediction`（`germinal/filters/filter_utils.py`）把 `ipsae`
初始化為 None，`af3` 與 `protenix` 分支都會賦值，但 **`chai` 分支從不賦值**。
由於 pdockq2 現在只來自 ipsae（見下），chai 路徑下 pdockq2 恆為 None，
最終過濾器 `pdockq2 > 0.23` 必定失敗 → 0 accepted。

### 根因

```python
# germinal/filters/filter_utils.py  (run_structure_prediction)
ipsae = None
if run_settings["structure_model"] == "af3":
    external_pdb, external_metrics, ipsae = af3.run_af3(...)      # 設 ipsae
elif run_settings["structure_model"] == "chai":
    external_pdb, external_metrics = chai.run_chai(...)           # 未設 ipsae
elif run_settings["structure_model"] == "protenix":
    external_pdb, external_metrics, ipsae = protenix.run_protenix(...)  # 設 ipsae
return external_pdb, external_metrics, ipsae
```

而 `compute_pdockq_and_lis` 現在讓 pdockq2 專屬來自 ipsae
（`filter_utils.py` 註解說明是為修正舊的 ≥3-chain per-chain 聚合錯誤）：

```python
"pDockQ2": ipsae["pdockq2"] if ipsae is not None else None,
```

### 症狀

chai 完整 run 中，最終過濾階段：

```
[FILTER ERROR] Metric 'pdockq2' is None — cannot evaluate filter (> 0.23).
FAILING this filter (was previously silently passed).
```

### 影響

`structure_model=chai` 也是官方三選項之一，現況下用它同樣 100% 產不出設計。

### 建議修法

chai 路徑需要從 chai 輸出構造 ipsae（如同 protenix 分支對 AF3-compatible JSON
呼叫 ipsae 工具），或至少讓 chai 分支回傳可用的 pDockQ2 來源。這比 Issue 4 複雜，
需要研究 chai 輸出能否餵給 ipsae 工具產生 pdockq2；此處僅回報問題，未提供修補。

（暫解：若只想讓 chai 路徑能跑完，可從 `configs/filter/final/*.yaml` 移除
pdockq2，但會少一項外部驗證指標。）

---

## 附註：非 bug，但文件可補充

`configs/config.yaml:25` 的 `af_params_dir` 預設為空字串，README 未說明應指向哪一層。
由 `colabdesign/colabdesign/af/alphafold/model/data.py:34` 可知它會 join `params/`：

```python
path = os.path.join(data_dir, 'params', f'params_{model_name}.npz')
```

所以應指向**包含** `params/` 的那一層（例如 AlphaFold 資料庫根目錄），
而不是 `params/` 本身。文件補一句可省去踩坑。
