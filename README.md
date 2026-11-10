# In-House Legal Intelligence System (CS781 Capstone Project)

**Author:** Ayush Mangal  
**Roll Number:** 240242  
**Email:** ayushn24@iitk.ac.in  
**Institution:** Indian Institute of Technology Kanpur (IIT Kanpur)  
**CodaBench Username:** `a_240242`  
**Repository:** [github.com/ayushyra222324/Legal-Intelligence-System](https://github.com/ayushyra222324/Legal-Intelligence-System)

---

## Executive Summary

This repository contains the complete implementation, pretraining scripts, adaptation notebooks, and evaluation pipelines for an **In-House Legal Intelligence System** constructed under the strict constraints of the CS781 Capstone Project. To maintain strict data privacy, prevent privilege leakage, and avoid reliance on commercial cloud APIs, all models were trained from scratch ($50\text{M}$--$300\text{M}$ parameter lock) on a 200M-token Indian Legal Corpus without external pretrained weights.

The system features four foundational Transformer backbones pretrained in Phase 1 and systematically transferred across six specialized downstream legal NLP tasks (Phases 2--7) using Parameter-Efficient Fine-Tuning (PEFT), bidirectional attention conversion, Dual-Anchor chunking, threshold optimization, and logit-level Knowledge Distillation (KD).

---

## CodaBench Official Leaderboard Results

The table below summarizes the final submission scores, primary evaluation metrics, leaderboard ranks, and submission counts recorded on **November 9, 2026** across all seven project phases (39 total valid submissions).

| Phase | Task | Winning Model & Adaptation | Primary Metric | Primary Score | Secondary Metric Scores | Rank | Valid Submissions |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **P1** | Legal LM Pretraining | Hybrid Prefix-LM (59M) | BERTScore F1 | **0.796455** | ROUGE-L: 0.108061 | 141 | 4 |
| **P2** | Legal NER | Modern Decoder + LORMA ($r=128$, Bidi) | Strict Macro-F1 | **0.373512** | Strict Prec: 0.348576, Strict Rec: 0.412869 | 109 | 7 |
| **P3** | Rhetorical Role Prediction | Modern Decoder + LORMA ($r=128$, Mean Pool) | Macro-F1 | **0.376948** | Macro-Prec: 0.395100, Macro-Rec: 0.381779 | 127 | 7 |
| **P4** | Court Judgment Prediction | Modern Decoder + LORMA ($r=16$, 1024 Tok) | Macro-F1 | **0.544039** | Macro-Prec: 0.544209, Macro-Rec: 0.573491 | 127 | 7 |
| **P5** | Legal Statute Identification | GQA Decoder + LORMA ($\tau=0.10$) | Macro-F1 | **0.177300** | Micro-F1: 0.318500, Subset Acc: 0.000000 | 133 | 4 |
| **P6** | Abstractive Summarization | Modern Decoder Student + KD (3-Layer) | BERTScore F1 | **0.821925** | ROUGE-1: 0.333589, ROUGE-L: 0.185996 | 117 | 5 |
| **P7** | Prior Case Retrieval (PCR) | Modern Decoder Teacher + LORMA (InfoNCE) | Micro-F1@7 | **0.154164** | Micro-Prec@7: 0.157324, Micro-Rec@7: 0.151129 | 130 | 5 |

---

## Repository Structure

```
Legal-Intelligence-System/
├── Phase1/
│   ├── Phase1_moderndecoder.ipynb       # 134M LLaMA-style Causal LM pretraining
│   ├── phase1_encdec.ipynb             # 106M T5-style Encoder-Decoder pretraining
│   ├── phase1_gqa.ipynb                # 64M Grouped-Query Attention Causal LM
│   └── phase1_hybridencoder.ipynb      # 59M Hybrid Prefix-LM pretraining (P1 Winner)
├── Phase2/
│   ├── phase2_encdec_lorma.ipynb       # Encoder-Decoder + LORMA token classification
│   ├── phase2_houlsby_gqa.ipynb        # GQA Decoder + Houlsby Bottleneck Adapters
│   ├── phase2_hybridenc_lora.ipynb     # Hybrid Encoder + Standard PEFT LoRA
│   ├── phase2_hybridenc_lorma.ipynb    # Hybrid Encoder + LORMA adapters
│   ├── phase2_lorma_gqa_crf.ipynb      # GQA Decoder + LORMA + CRF Sequence Head
│   ├── phase2_moderndec_lorma.ipynb    # Modern Decoder + Bidi LORMA (P2 Winner)
│   └── phase2_moderndec_lorma_distill.ipynb # Modern Decoder KD distillation
├── Phase3/
│   ├── phase3_encdec_lorma.ipynb       # Encoder-Decoder + LORMA sentence classifier
│   ├── phase3_gqa_lora.ipynb           # GQA Decoder + Standard LoRA
│   ├── phase3_gqa_lorma.ipynb          # GQA Decoder + LORMA adapters
│   ├── phase3_gqa_prefixtuning.ipynb   # GQA Decoder + Virtual Prefix Tuning (32 tok)
│   ├── phase3_hybridenc_lorma.ipynb    # Hybrid Encoder + LORMA adapters
│   ├── phase3_moderndec_lorma.ipynb    # Modern Decoder + Bidi LORMA (P3 Winner)
│   └── phase3_moderndec_teacher.ipynb  # Modern Decoder Teacher & Student Distillation
├── Phase4/
│   ├── phase4_encdec_lorma.ipynb       # Encoder-Decoder + LORMA Dual-Anchor classification
│   ├── phase4_gqa_bitfit.ipynb         # GQA Decoder + BitFit (Bias-Only tuning)
│   ├── phase4_gqa_lora.ipynb           # GQA Decoder + Standard LoRA
│   ├── phase4_gqa_lorma.ipynb          # GQA Decoder + LORMA adapters
│   ├── phase4_hybridencoder_lorma.ipynb# Hybrid Encoder + LORMA adapters
│   ├── phase4_moderndec_lorma.ipynb    # Modern Decoder + LORMA 1024-tok (P4 Winner)
│   └── phase4_moderndec_lorma_student.ipynb # 2-Layer GQA Student Distillation
├── Phase5/
│   ├── phase5_encdec_prompttuning.ipynb# Encoder-Decoder + Soft Prompt Tuning (16 tok)
│   ├── phase5_gqa_lorma.ipynb          # GQA Decoder + LORMA + Threshold Search (P5 Winner)
│   ├── phase5_hybridenc_qlora.ipynb    # Hybrid Encoder + 4-bit NF4 QLoRA
│   └── phase5_moderndec_lorma_distill.ipynb # Modern Decoder LORMA Distillation
├── Phase6/
│   ├── phase6_encdec_qlora.ipynb       # Encoder-Decoder + QLoRA Seq2Seq Summarization
│   ├── phase6_gqa_lorma.ipynb          # GQA Decoder + LORMA Summarization
│   ├── phase6_hybridenc_lora.ipynb     # Hybrid Encoder + Causal LoRA Summarization
│   └── phase6_moderndec_lorma_teacher.ipynb # Modern Decoder KD Distillation (P6 Winner)
├── Phase7/
│   ├── Phase7_GQA_Lorma.ipynb          # GQA Decoder + LORMA Metric Learning
│   ├── Phase7_Hybridenc_Prefixtuning.ipynb # Hybrid Encoder + Virtual Prefix Tuning
│   ├── Phase7_Moderndec_lorma_teacher.ipynb # Modern Decoder Teacher + KD (P7 Winner)
│   └── Phase7_encdec_qlora.ipynb       # Encoder-Decoder + QLoRA InfoNCE Retrieval
└── README.md                           # Detailed System & Methodology Documentation
```

---

## Detailed Methodology & Implementation by Phase

### Phase 1: Legal Language Model Pretraining
* **Task Goal:** Pretrain compact domain foundation language models from scratch on a 200M-token Indian legal corpus comprising Supreme Court rulings (45%), High Court judgments (35%), Statutory Acts (15%), and Tribunal orders (5%).
* **Tokenization:** Custom 32,000-token byte-level BPE tokenizer trained via `tokenizers`. Preserved statutory section symbols ($\S$) and citation formats (e.g., `AIR 1954 SC 282`) to prevent subword fragmenting.
* **Architectures Trained:**
  1. **Hybrid Prefix-LM (59.23M params) [P1 Winner]:** 8 layers, $d_{\text{model}}=768$, 12 heads. Bidirectional self-attention over prompt prefixes combined with causal masking for target generation. Val Loss: 4.4334, BERTScore F1: **0.796455**.
  2. **Grouped-Query Attention (GQA) Causal LM (63.91M params):** 8 decoder layers, $d_{\text{model}}=768$, $d_{\text{ff}}=2048$, 12 query heads grouped into 4 KV heads (3:1 ratio), RoPE, RMSNorm. Val Loss: 5.3007, BERTScore F1: **0.790354**.
  3. **Custom Legal Encoder-Decoder (105.71M params):** 6 encoder / 6 decoder layers, $d_{\text{model}}=768$, $d_{\text{ff}}=2048$, cross-attention. Val Loss: 3.9352, BERTScore F1: **0.783713**.
  4. **Modern Decoder (134.11M params):** 12 causal decoder layers, $d_{\text{model}}=768$, 12 heads, SwiGLU MLP activations, RMSNorm, RoPE. Val Loss: 4.2547, BERTScore F1: **0.768136**.
* **Notebooks:** `Phase1_moderndecoder.ipynb`, `phase1_encdec.ipynb`, `phase1_gqa.ipynb`, `phase1_hybridencoder.ipynb`.

---

### Phase 2: Legal Named Entity Recognition (NER)
* **Task Goal:** Sequence tagging over 12 legal entity categories (e.g., `COURT`, `STATUTE`, `JUDGE`, `PROVISION`) evaluated via Strict Macro-F1 with exact character boundary matching.
* **Key Innovations & Alignment:**
  * **Exact Offset Mapping:** Integrated `return_offsets_mapping=True` during subword tokenization, mapping tokens back to raw document character positions to prevent zero-score submission drift on CodaBench.
  * **Bidirectional Attention Conversion:** Converted causal decoders into full bidirectional encoders (`is_causal=False`), exposing left-and-right sequence context.
* **Notebooks & Submissions:**
  * `phase2_moderndec_lorma.ipynb` **(Winner - Strict F1: 0.3735)**: Modern Decoder + Rank-128 LORMA adapters + Bidirectional conversion.
  * `phase2_hybridenc_lorma.ipynb` **(Strict F1: 0.0981)**: Hybrid Encoder + Rank-128 LORMA.
  * `phase2_moderndec_lorma_distill.ipynb` **(Strict F1: 0.0828)**: 3-Layer distilled student model.
  * `phase2_lorma_gqa_crf.ipynb` **(Strict F1: 0.0755)**: GQA Decoder + LORMA + CRF sequence classification head.
  * `phase2_houlsby_gqa.ipynb` **(Strict F1: 0.0743)**: GQA Decoder + Houlsby Bottleneck Adapters.
  * `phase2_hybridenc_lora.ipynb` **(Strict F1: 0.0022)**: Standard PEFT LoRA (demonstrating LoRA boundary failure).
  * `phase2_encdec_lorma.ipynb` **(Strict F1: 0.0000)**: Seq2Seq token classification optimization failure.

---

### Phase 3: Rhetorical Role Prediction
* **Task Goal:** Sentence-level discourse classification into 13 structural categories (`Facts`, `Argument`, `Ratio`, `Precedent`, `Dissent`, etc.) across Corporate Law and IT judgments.
* **Key Innovations:**
  * **Sentence Pooling:** Sequence representations were length-normalized and mean-pooled across sentence boundaries.
  * **Class-Weighted Cross-Entropy:** Penalized dominant classes (`Facts` and `Arguments` $>62\%$) to support rare roles (`Dissent` $<1.4\%$).
* **Notebooks & Submissions:**
  * `phase3_moderndec_lorma.ipynb` **(Winner - Macro-F1: 0.3769)**: Modern Decoder + Bidirectional Attention + Rank-128 LORMA.
  * `phase3_encdec_lorma.ipynb` **(Macro-F1: 0.3135)**: Encoder-Decoder + LORMA adapters.
  * `phase3_hybridenc_lorma.ipynb` **(Macro-F1: 0.2975)**: Hybrid Encoder + Rank-16 LORMA.
  * `phase3_moderndec_teacher.ipynb` **(Macro-F1: 0.2953)**: Distilled 10.56M TinyStudent model.
  * `phase3_gqa_prefixtuning.ipynb` **(Macro-F1: 0.2714)**: GQA Decoder + 32-Token Virtual Prefix Tuning.
  * `phase3_gqa_lora.ipynb` **(Macro-F1: 0.2649)**: GQA Decoder + Standard LoRA ($r=16$).
  * `phase3_gqa_lorma.ipynb` **(Macro-F1: 0.2640)**: GQA Decoder + Rank-128 LORMA.

---

### Phase 4: Court Judgment Prediction & Explanation
* **Task Goal:** Binary decision outcome classification (0/1) over lengthy judicial rulings (averaging 4,200 words).
* **Key Innovations:**
  * **Dual-Anchor Chunking:** To fit within bounded context windows without losing critical information, the first 1024 tokens (facts/petitions) and final 1024 tokens (orders/verdicts) were concatenated into a 2048-token Dual-Anchor sequence.
* **Notebooks & Submissions:**
  * `phase4_moderndec_lorma.ipynb` **(Winner - Macro-F1: 0.5440)**: Modern Decoder + LORMA ($r=16, \alpha=32$, 1024 tokens).
  * `phase4_moderndec_lorma_student.ipynb` **(Macro-F1: 0.5307)**: 2-Layer GQA distilled student achieving high recall (**0.7677**).
  * `phase4_encdec_lorma.ipynb` **(Macro-F1: 0.5088)**: Encoder-Decoder + Rank-128 LORMA.
  * `phase4_gqa_lora.ipynb` **(Macro-F1: 0.4942)**: GQA Decoder + Standard LoRA ($r=16$).
  * `phase4_gqa_bitfit.ipynb` **(Macro-F1: 0.4691)**: BitFit tuning of only 53K bias parameters.
  * `phase4_gqa_lorma.ipynb` **(Macro-F1: 0.4687)**: GQA Decoder + Rank-16 LORMA (Dual-Anchor 2048 tokens).
  * `phase4_hybridencoder_lorma.ipynb` **(Macro-F1: 0.3429)**: Hybrid Encoder + Rank-16 LORMA.

---

### Phase 5: Legal Statute Identification (LSI)
* **Task Goal:** Multi-label classification across 100 statutory provisions from case facts.
* **Key Innovations:**
  * **Handling Target Sparsity (98.16% Zeros):** Unweighted Binary Cross-Entropy (BCE) pushed raw logits below -4.0, causing zero-label predictions under default $\tau=0.50$ thresholds.
  * **Post-Hoc Decision Threshold Search:** Lowering decision thresholds to $\tau=0.10$ restored positive statutory recall across all sparse targets.
* **Notebooks & Submissions:**
  * `phase5_gqa_lorma.ipynb` **(Winner - Macro-F1: 0.1773)**: GQA Decoder + LORMA ($r=16, \tau=0.10$, Micro-F1: 0.3185).
  * `phase5_encdec_prompttuning.ipynb` **(Macro-F1: 0.1007)**: Encoder-Decoder + Soft Prompt Tuning (16 tokens, 89K trainable params).
  * `phase5_moderndec_lorma_distill.ipynb` **(Macro-F1: 0.0621)**: Modern Decoder + LORMA distillation.
  * `phase5_hybridenc_qlora.ipynb` **(Macro-F1: 0.0471)**: Hybrid Encoder + 4-bit NF4 QLoRA.

---

### Phase 6: Abstractive Summarization
* **Task Goal:** Sequence-to-sequence legal summarization of 1536-token case texts evaluated via BERTScore F1, ROUGE-1, and ROUGE-L.
* **Key Innovations:**
  * **Extract-then-Abstract Pipeline:** TF-IDF sentence centrality and cue-word position scoring populated a 541-token prompt context.
  * **Logit-Level Knowledge Distillation:** Distilled fine-tuned 6-layer teachers into compact 3-layer students using soft-target KL-divergence ($T=2.0, \alpha=0.5$).
* **Notebooks & Submissions:**
  * `phase6_moderndec_lorma_teacher.ipynb` **(Winner - BERTScore F1: 0.821925)**: 3-Layer Modern Decoder Student via KD (ROUGE-1: **0.333589**, ROUGE-L: **0.185996**). Outperformed its 6-layer teacher (BERTScore F1: 0.814779).
  * `phase6_gqa_lorma.ipynb` **(BERTScore F1: 0.793868)**: GQA Decoder + LORMA ($r=16, \alpha=32$).
  * `phase6_encdec_qlora.ipynb` **(BERTScore F1: 0.782626)**: Encoder-Decoder + QLoRA ($r=16$, ROUGE-1: **0.292672**).
  * `phase6_hybridenc_lora.ipynb` **(BERTScore F1: 0.712070)**: Hybrid Encoder Causal LM (suffered N-gram repetition collapse, ROUGE-1: 0.0031).

---

### Phase 7: Prior Case Retrieval (PCR)
* **Task Goal:** Dense bi-encoder metric learning returning the top 7 relevant legal precedents for a query case from a candidate pool (Micro-F1@7).
* **Key Innovations:**
  * **Contrastive Metric Learning:** Bi-encoder trained with in-batch InfoNCE contrastive loss ($\tau=0.05$) on $L_2$-normalized document embeddings:
    $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\sum_{d \in \mathcal{D}} \exp(\text{sim}(q, d)/\tau)}$$
* **Notebooks & Submissions:**
  * `Phase7_Moderndec_lorma_teacher.ipynb` **(Winner - Micro-F1@7: 0.154164)**: 6-Layer Modern Decoder Teacher + LORMA ($r=16, \text{InfoNCE }\tau=0.05$).
  * `Phase7_encdec_qlora.ipynb` **(Micro-F1@7: 0.145304)**: Encoder-Decoder + QLoRA ($r=16$).
  * `Phase7_Moderndec_lorma_teacher.ipynb` *(Student Script)* **(Micro-F1@7: 0.144123)**: 3-Layer Distilled Student (retained 93.5% teacher score at 25.7% parameter size).
  * `Phase7_GQA_Lorma.ipynb` **(Micro-F1@7: 0.135854)**: GQA Decoder + LORMA.
  * `Phase7_Hybridenc_Prefixtuning.ipynb` **(Micro-F1@7: 0.011813)**: Hybrid Encoder + Virtual Prefix Tuning (demonstrating soft prefix metric learning collapse).

---

## Technical Ablations & Critical Findings

1. **Bidirectional Attention Conversion (+10.55 F1 Gain):** Disabling causal attention masks ($\text{is\_causal}=\text{False}$) during sequence tagging (P2) and discourse classification (P3) exposed full sequence context, boosting P2 Strict Macro-F1 from 0.0812 to **0.3735** and P3 Macro-F1 from 0.2714 to **0.3769**.
2. **LORMA vs. Standard LoRA (44.9$\times$ Performance Boost):** On token-level boundary tasks (P2 NER), Linear-Out Rank-Minimizing Adaptation (LORMA) achieved **0.0981** Strict F1 compared to standard PEFT LoRA (**0.0022**). Interleaved rank-minimizing projections preserve subword boundary transitions significantly better than parallel low-rank matrices.
3. **Distillation regularizes text generation:** In Phase 6 (Summarization), distilling soft teacher probabilities into a compact 3-layer student achieved higher BERTScore F1 (**0.8219**) than the teacher model (**0.8148**). KL-divergence distillation acts as a label-smoothing regularizer that prevents autoregressive hallucination loops.
4. **Virtual Prefixes fail in Metric Learning:** Prefix Tuning (P7 Retrieval) collapsed to **0.0118** Micro-F1@7 compared to QLoRA (**0.1453**). Virtual key-value prompt prefixes cannot deform a frozen representation space for dense contrastive retrieval.

---

## Computational Environment & Requirements

* **Hardware:** Dual NVIDIA Tesla T4 GPUs (16GB VRAM per GPU).
* **Software Stack:** PyTorch Native Automatic Mixed Precision (AMP FP16), `tokenizers`, `peft`, `bert-score`, `rouge-score`, `scikit-learn`.
* **Execution & Resiliency:** Pretraining runs completed in 5.5--10.4 hours; downstream fine-tuning runs completed in 0.5--1.9 hours per job queue. All notebooks implement 20-minute checkpointing and automated recovery from preemption.
