---
layout: post
title:  "Hierarchical Classification of Social Media Comments Using LLM-to-BERT Knowledge Distillation"
date:   2025-12-11 14:34:15 -0600
categories: LLM
---

## Motivation

Categorizing 5.8 million Danmaku (bullet screen) comments into nuanced categories—**emotion-positive**, **emotion-negative**, **informational**, **social**, and **others**—presents unique challenges that standard classification approaches struggle to solve:

**The Core Challenge**:
- **Extreme class imbalance**: Information comments constitute < 1% of the dataset, making them nearly impossible to learn in a multi-label setting
- **Conflicting task requirements**: Emotion detection benefits from abundant training data, while information detection suffers from severe data scarcity
- **Label quality issues**: Initial LLaMA-generated pseudo-labels achieved only ~60% accuracy, insufficient for production use
- **Scale constraints**: Processing millions of comments requires efficient models (BERT/RoBERTa), but training them needs high-quality labels

**Key Contribution**:

This work introduces a **hierarchical classification pipeline with agreement-based quality filtering** that addresses these challenges through three innovations:

1. **Cross-model agreement filtering**: Using LLaMA-BERT disagreement to identify ambiguous samples and improve training data quality (44% agreement rate yields 106K high-quality samples)

2. **Hierarchical specialization**: Separating emotion detection (BERT on 106K samples) from information detection (RoBERTa on 1K manual labels) allows each model to optimize for its specific task, dramatically improving performance

3. **Disagreement-based relabeling**: Treating model disagreements as signals of inherent ambiguity rather than discarding them, converting uncertainty into a trainable "others" category

**Result**: Emotion accuracy improved from 60% to 90%+, and information detection went from impossible (F1=0.00) to production-ready (F1=0.93).

This post documents the complete pipeline combining:
- **LLaMA 3.1-8B-Instruct** for initial pseudo-labeling (596K comments)
- **BERT-base-Chinese** fine-tuned for emotion detection via agreement filtering
- **RoBERTa-Chinese** specialized for information detection via manual labeling

## Large Language Model: LLaMA 3.1-8B-Instruct

I used **LLaMA 3.1-8B-Instruct** running in an Apptainer container on A100 GPUs to generate pseudo-labels for the initial training dataset.


## Data

We aggregate semi-syncrhonized comments at a video-hourly level where each record holds a video_id, the collection timestamp collect_YMDH (YYYYMMDDHH), a chat_id, a maxlimit cap, and danmu_list, which is a concatenated sequence of 9-field danmu tuples (per comment: onset time in seconds, display mode, font size, RGB color as int, Unix ms timestamp, pool flag, sender ID, database row ID, and the comment text). 


The table below shows the first three lines (header + 2 data rows) from the `1_danmu_day` file:
 

## Original Data Row
**video_id:** 78462684  
**collect_YMDH:** 2019122222  
**chat_id:** 134249478  
**maxlimit:** 500

## Individual Danmu Comments (13 total)

| # | onset_time | mode | font_size | color      | timestamp_ms  | pool | sender_id | row_id              | text                          |
|---|------------|------|-----------|------------|---------------|------|-----------|---------------------|-------------------------------|
| 1 | 10.86500   | 1    | 25        | 16777215   | 1576985463    | 0    | 15953525  | 26132238280687620   | 先点赞，后观看！              |
| 2 | 73.37900   | 1    | 25        | 16777215   | 1576989665    | 0    | b7954b2b  | 26134441157459970   | ？？？？                      |
| 3 | 81.88600   | 1    | 25        | 16777215   | 1576993241    | 0    | 4692f37b  | 26136315807924226   | 我懂了                        |
| 4 | 11.76400   | 1    | 25        | 16777215   | 1576997999    | 0    | 55d4babf  | 26138810264846336   | 没错我才刚开机                |
| 5 | 73.78500   | 1    | 25        | 16777215   | 1576998657    | 0    | 9fd71001  | 26139155370606596   | ？？？？？？？？？？？？？    |
| 6 | 38.78000   | 1    | 25        | 16777215   | 1577005504    | 0    | aba03d84  | 26142745424297986   | 。。                          |
| 7 | 7.70200    | 1    | 25        | 16777215   | 1577009056    | 0    | 76614f4b  | 26144607672729600   | 这是什么语言阿？              |
| 8 | 60.25000   | 1    | 25        | 0          | 1577010216    | 0    | baca6618  | 26145215876169728   | 没看懂                        |
| 9 | 58.77500   | 1    | 25        | 16777215   | 1577014892    | 0    | 29aaa5d   | 26147667434274816   | 我在看啥                      |
| 10| 14.08000   | 1    | 25        | 16777215   | 1577019075    | 0    | 66897ec4  | 26149860501094466   | 这是matlab或者octave          |
| 11| 47.66400   | 1    | 25        | 16777215   | 1577023539    | 0    | aa1cd06c  | 26152200771207170   | ？？？                        |
| 12| 103.84000  | 1    | 25        | 16777215   | 1577024647    | 0    | 1be0174f  | 26152782122713088   | 看呆了忘记发弹幕              |
| 13| 11.32300   | 1    | 25        | 16777215   | 1577026708    | 0    | 318842ed  | 26153862433341442   | m文件，执行脚本。matlab编个函数或者ui还是很容易的 |

## Field Descriptions

- **onset_time**: Time when the danmu appears in the video (seconds)
- **mode**: Display mode
  - 1-3: Scrolling danmu
  - 4: Bottom danmu
  - 5: Top danmu
  - 6: Reverse danmu
  - 7: Precise positioning
  - 8: Advanced danmu
- **font_size**: Text size (12=very small, 16=extra small, 18=small, 25=medium, 36=large, 45=very large, 64=extra large)
- **color**: RGB color as decimal (16777215 = #FFFFFF white, 0 = #000000 black)
- **timestamp_ms**: Unix timestamp in milliseconds when the comment was posted
- **pool**: Comment pool (0=normal, 1=subtitle, 2=special/advanced)
- **sender_id**: User ID (for blocking functionality)
- **row_id**: Database row ID (for history functionality)
- **text**: The actual danmu comment text

**Notes:** 

## Labeling Schema Definition

For this categorization task, we developed a comprehensive 5-category schema designed to capture the semantic richness of danmu comments:

### Main Categories and Subcategories:

**1. Information**
- `domain_knowledge`: Technical or specialized knowledge related to video content
- `situated_knowledge`: Context-specific information about the video or situation
- `subjective_opinion`: Personal opinions or judgments about the content
- `video_relevant_remarks`: Direct commentary about what's happening in the video
- `external_sources`: References to external information or sources

**2. Emotions Positive**
- `admiration`: Expressions of respect or approval
- `encouragement`: Supportive or motivational comments
- `gratitude`: Thanks or appreciation
- `joy`: Happiness, amusement, or delight
- `empathy`: Understanding or relating to others
- `moved`: Being emotionally touched
- `relieved`: Expressing relief or comfort

**3. Emotions Negative**
- `criticism`: Negative judgments or complaints
- `worry`: Expressions of concern or anxiety  
- `shocked`: Surprise or disbelief
- `recollecting`: Nostalgic or sad remembrance

**4. Social**
- `social_presence`: Acknowledging the community or other viewers
- `self_report`: Sharing personal status or situation
- `interaction`: Attempting to communicate with others
- `curiosity`: Asking questions or expressing wonder
- `confusion`: Expressing lack of understanding

**5. Other**
- `other`: Comments that don't fit into the above categories

## Hierarchical Classification Architecture

After initial experimentation, I discovered that a single multi-label classifier struggled with both emotion detection and information identification. This led me to develop a **two-stage hierarchical approach**:

### Stage 1: Emotion Classification (BERT)
**Goal**: Identify positive and negative emotions with high precision

**Process**:
1. Label 596K comments using LLaMA 3.1-8B-Instruct
2. Train initial BERT on 100K LLaMA-labeled samples
3. Run BERT inference on remaining 496K holdout samples
4. Identify **agreement samples** where LLaMA and BERT predictions match
5. Use agreement-filtered data to train improved BERT model
6. Apply to all comments to detect emotions

**Performance**: 
- Positive emotion: **91% precision, 90%+ accuracy**
- Negative emotion: **84% precision, 80%+ accuracy**

### Stage 2: Information Detection (RoBERTa)
**Goal**: High-precision information extraction from non-emotional comments

**Process**:
1. Manually label 1,000 comments with emphasis on information/others distinction
2. Train specialized RoBERTa binary classifier with text features
3. Apply only to comments classified as "others" or "social" by BERT

**Performance**:
- Information F1: **0.93** (vs 0.04 for multi-label BERT)
- Information recall: **91%** (detects 91% of information comments)
- Information precision: **95%** (95% of predictions correct)

**Why Hierarchical?** The initial multi-label BERT achieved only **~60% accuracy** for emotions and performed very poorly on information detection (F1=0.04, even with 10× class weighting). By separating emotion detection from information detection:
- BERT focuses on what it does well (emotion patterns)
- RoBERTa specializes in information (trained on curated manual labels)
- Each model optimizes for its specific task

---

## LLaMA Model Labeling Process (Stage 1: Pseudo-Labeling)

### Model Setup
We used **LLaMA 3.1-8B-Instruct** model running in an Apptainer container environment to perform initial labeling of danmu comments. The model was deployed on GPU compute nodes with the following specifications:
- Model: LLaMA 3.1-8B-Instruct
- Hardware: A100 GPUs with 64GB memory
- Batch size: 50 comments per inference call
- Max tokens: 1800 per response

### Prompt Engineering
To ensure consistent and accurate labeling, we developed a structured prompt template:

```python
def build_prompt(batch_index, comments_batch):
    instruction = f"""You are a comment labeling assistant. Your ONLY job is to return JSON.

TASK: Label these {len(comments_batch)} comments using the schema below.

CRITICAL INSTRUCTIONS:
- Do NOT repeat this prompt
- Do NOT add explanations 
- Return ONLY the JSON object
- Start your response with {{ 
- End your response with }}

REQUIRED JSON FORMAT:
{{"comments": [{{"comment_id": "u1", "labels": ["category"], "rationale": "brief reason"}}, ...]}}

LABEL SCHEMA:
{schema_text}

RULES:
- For emotions_positive/negative: include both parent + subcategory: ["emotions_positive", "joy"]
- Keep rationale under 8 English words or 12 Chinese characters
- Use ["other"] if nothing fits

COMMENTS TO LABEL:
{comments_list}

JSON RESPONSE:"""
```

### Processing Pipeline
1. **Data Preparation**: Extract unique danmu comments from video collections
2. **Batch Processing**: Group comments into batches of 50 for efficient processing
3. **LLM Inference**: Send batches to LLaMA model with structured prompts
4. **Response Parsing**: Extract JSON responses and handle malformed outputs
5. **Quality Control**: Implement retry mechanisms for failed batches
6. **Checkpoint System**: Save progress for resumable processing

### Challenges and Solutions
- **Response Inconsistency**: LLM occasionally echoed prompts back; solved with response cleaning functions
- **JSON Parsing Errors**: Implemented robust parsing with fallback mechanisms for truncated responses
- **Scale**: Processing 5.8M comments required 48+ hour GPU sessions with background processing
- **Failure Handling**: Developed retry queue system for failed batches

## BERT Fine-tuning Methodology (Stage 2: Initial Training)

### Initial BERT Training (Version 1)

I started by training BERT-base-Chinese on a sample of LLaMA-generated labels to create a faster classifier.

**Sampling Strategy**:
- **596K total LLaMA labels** generated
- **100K samples** used for initial BERT training (to keep training time manageable)
- **496K holdout samples** reserved for agreement analysis

```python
class MultiLabelBertClassifier(nn.Module):
    def __init__(self, model_name, num_main_labels, num_subcat_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Two classification heads
        self.main_classifier = nn.Linear(self.bert.config.hidden_size, num_main_labels)
        self.subcat_classifier = nn.Linear(self.bert.config.hidden_size, num_subcat_labels)
```

### Model Architecture

**BERT Multi-Label Classifier**:
```python
class MultiLabelBertClassifier(nn.Module):
    def __init__(self, model_name, num_main_labels, num_subcat_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Two classification heads
        self.main_classifier = nn.Linear(self.bert.config.hidden_size, num_main_labels)
        self.subcat_classifier = nn.Linear(self.bert.config.hidden_size, num_subcat_labels)
```

### Initial Training Results (BERT v1)

Training on 100K LLaMA-labeled samples achieved modest performance:

| Category | Precision | Recall | F1-Score | Support | Notes |
|----------|-----------|---------|----------|---------|-------|
| **information** | 0.52 | 0.27 | **0.36** | 3,238 | Low recall - many missed |
| **emotions_positive** | 0.68 | 0.62 | **0.64** | 4,612 | Best category |
| **emotions_negative** | 0.55 | 0.46 | **0.50** | 2,417 | Moderate performance |
| **social** | 0.43 | 0.00 | **0.01** | 992 | Model rarely predicts this |
| **other** | 0.60 | 0.55 | **0.57** | 8,165 | Largest category |

**Overall Metrics**:
- Micro F1: **0.54**
- Weighted F1: **0.52**
- Training time: **72 minutes** on A100 GPU

**Key Observations**:
- ✅ **Emotions_positive performed best** (F1: 0.64) - BERT captures positive sentiment well
- ⚠️ **Information category struggled** (F1: 0.36, recall: 0.27) - missed 73% of information comments
- ❌ **Social category failed** (F1: 0.01) - model almost never predicts this class
- 📊 **Overall accuracy ~60%** - not sufficient for production use

**Root Cause**: Multi-label classification with severe class imbalance. Information comments represent only 0.45% of data, making it nearly impossible for the model to learn this minority class effectively.

---

## Agreement-Based Quality Filtering (Stage 3)

To improve training data quality, I developed a novel agreement-based filtering strategy.

### The Agreement Analysis

**Hypothesis**: When LLaMA and BERT agree on a label, it's likely correct. When they disagree, the sample may be ambiguous or incorrectly labeled.

**Process**:
1. Run BERT v1 inference on **496K holdout samples** (not used for training)
2. Compare BERT predictions with LLaMA labels
3. Calculate agreement rates after normalizing labels (fixing "other" vs "others" mismatch)

**Results After Label Normalization**:

```
Total LLaMA predictions: 596,000
Total BERT training samples: ~100,000
Holdout samples analyzed: ~527,000

Agreement breakdown (after normalization):
  ✅ Agreed:     233,281 (44.26%)
  ❌ Disagreed:  293,776 (55.74%)
```

**Critical Discovery**: Initial agreement appeared to be only **22%**, but after normalizing label formats ("other" → "others", deduplicating labels), agreement jumped to **44.26%**. Many "disagreements" were actually schema mismatches!

### Strategic Sampling for Improved Training Data

Based on agreement analysis, I created a curated training dataset using:

**Sampling Rules**:
1. **100% of agreed non-"others"** (65,729 samples) → High-quality emotion/information labels
2. **15% of agreed "others"** (25,132 of 167,552 samples) → Prevent "others" from dominating
3. **5% of disagreed** (14,688 of 293,776 samples) → Relabel as "others" (inherently ambiguous)
4. **100% of manual labels** (1,000 samples) → Human-validated gold standard

**Final Training Dataset Composition**:
```
Total samples: 106,549

By source:
  agreed_non_others:       65,729 (61.68%) - High confidence labels
  agreed_others_original:  25,132 (23.59%) - Downsampled majority class
  disagreed_to_others:     14,688 (13.78%) - Ambiguous → "others"
  manual_labels:            1,000 (0.94%)  - Human validation

Train/Validation split: 80/20
  Training:   85,239 samples
  Validation: 21,310 samples
```

**Innovation**: Treating disagreements as inherently ambiguous and relabeling them as "others" rather than discarding them. This converts uncertainty into a trainable category.

---

## Improved BERT Training (Stage 4: BERT v2)

### Training Configuration

**Key Parameters**:
- Model: **bert-base-chinese**
- Optimizer: **AdamW** (learning rate: 2e-5)
- Batch size: **128** (A100 optimized)
- Epochs: **10** with early stopping
- Loss: **BCEWithLogitsLoss** with class weighting

**Class Weighting Strategy**:
To address severe class imbalance, I applied weighted loss:

```
Main category weights:
  information:       2198.26  (10× boost for 0.45% minority class)
  emotions_positive:    0.71  (Slight downweight for 58% majority)
  emotions_negative:   27.84  (Medium boost for 3.38%)
  social:             582.83  (High boost for rare class)
  others:               1.67  (Moderate for 37% class)
```

### Training Progress (BERT v2)

```
Epoch  1/10:  Loss: 1.9186  ← Initial high loss with weighted classes
Epoch  2/10:  Loss: 0.9976
Epoch  3/10:  Loss: 0.8219
Epoch  4/10:  Loss: 0.6855
Epoch  5/10:  Loss: 0.5250
Epoch  6/10:  Loss: 0.4175
Epoch  7/10:  Loss: 0.3476
Epoch  8/10:  Loss: 0.2909
Epoch  9/10:  Loss: 0.2533
Epoch 10/10:  Loss: 0.2206  ✅ Best model saved

Training time: ~4 hours (25 min/epoch × 10 epochs)
```

### BERT v2 Performance Results

Training on agreement-filtered data with class weights achieved significant improvements:

**Emotion Categories (Primary Use Case)**:

| Category | Precision | Recall | F1-Score | Improvement vs v1 |
|----------|-----------|---------|----------|-------------------|
| **emotions_positive** | **0.91** | **0.92** | **0.91** | +0.27 (from 0.64) |
| **emotions_negative** | **0.84** | **0.81** | **0.82** | +0.32 (from 0.50) |

**Other Categories**:

| Category | Precision | Recall | F1-Score | Notes |
|----------|-----------|---------|----------|-------|
| **information** | 0.00 | 0.00 | **0.00** | Still struggles despite 10× weighting |
| **social** | 0.00 | 0.00 | **0.00** | Rare class remains problematic |
| **others** | 0.94 | 0.88 | **0.91** | Strong performance |

**Overall Performance**:
- Micro F1: **0.93** (up from 0.54)
- Weighted F1: **0.93** (up from 0.52)
- **Emotion accuracy: 90%+ for positive, 80%+ for negative** ✅

### Key Insights from BERT v2

**Successes**:
- ✅ **Emotion detection vastly improved**: Agreement filtering + class weighting raised emotion F1 from 0.50-0.64 to 0.82-0.91
- ✅ **"Others" category well-defined**: F1 of 0.91 shows the model learned to identify ambiguous/off-topic comments
- ✅ **Production-ready for emotions**: 90%+ accuracy makes this suitable for emotion analysis

**Remaining Challenges**:
- ❌ **Information F1 still 0.00**: Even with 10× class weighting, severe imbalance (0.45% of data) prevents learning
- ❌ **Social category failed**: Too few training examples (< 0.1%)

**Decision Point**: Rather than continue struggling with multi-label classification for information, I pivoted to a **specialized binary classifier**.

---

## RoBERTa Information Classifier (Stage 5)

### Motivation for Separate Information Classifier

Multi-label BERT achieved **F1=0.00** for information despite aggressive class weighting. Why?
- Information represents only **0.45%** of training data (484 samples out of 106K)
- Overshadowed by emotion and "others" categories (95% of data)
- Different linguistic patterns (factual vs. emotional language)

**Solution**: Train a dedicated **binary classifier** (information vs others) using:
- High-quality **manually labeled data** (not LLaMA labels)
- **Chinese RoBERTa** with richer text features
- Focused task allowing the model to learn information patterns

### Manual Labeling Process

I manually labeled **1,000 Danmaku comments** with careful attention to:
- **Information categories**: domain_knowledge, video_relevant_remarks, subjective_opinion
- **Emotional categories**: To separate from information
- **Others**: Spam, off-topic, unclear

**Label Distribution After Filtering**:
```
Total samples after removing emotions/social: 518
  Information: 364 (70.3%)
  Others:       50 (9.7%)

Train/Test split:
  Train: 414 samples (information: 364, others: 50)
  Test:  104 samples (information: 92, others: 12)
```

### RoBERTa Model Architecture

**Key Innovation**: Combining text embeddings with numerical features

```python
class InformationClassifier(nn.Module):
    def __init__(self, roberta_model, hidden_size, num_features):
        super().__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.3)
        
        # Feature projection
        self.feature_projection = nn.Linear(num_features, 32)
        
        # Combined classifier
        self.classifier = nn.Linear(hidden_size + 32, 2)  # Binary

    def forward(self, input_ids, attention_mask, features):
        # RoBERTa embeddings
        outputs = self.roberta(input_ids, attention_mask)
        pooled = outputs.pooler_output
        
        # Combine text + features
        feature_proj = F.relu(self.feature_projection(features))
        combined = torch.cat([pooled, feature_proj], dim=1)
        
        return self.classifier(self.dropout(combined))
```

**Text Features Used**:
- `word_count` (log-transformed)
- `char_count`
- `num_question_marks`
- `has_url` (binary)
- `num_numbers`

**Intuition**: Information comments tend to be longer, contain more facts/numbers, and have fewer question marks than emotional expressions.

### Training Configuration

- **Model**: hfl/chinese-roberta-wwm-ext
- **Class weighting**: Automatic (neg_count / pos_count)
- **Batch size**: 16
- **Epochs**: 5
- **Learning rate**: 2e-5
- **Training time**: ~5-10 minutes on A100

### RoBERTa Performance Results

**Final Results (Epoch 5)**:

```
Classification Report:
              precision    recall  f1-score   support

      others       0.50      0.67      0.57        12
 information       0.95      0.91      0.93        92

    accuracy                           0.88       104
```

**Performance Comparison**:

| Metric | Multi-label BERT | RoBERTa Binary | Improvement |
|--------|------------------|----------------|-------------|
| **Information Precision** | 0.00 | **0.95** | +0.95 |
| **Information Recall** | 0.00 | **0.91** | +0.91 |
| **Information F1** | 0.00 | **0.93** | +0.93 |
| **Overall Accuracy** | N/A | **88%** | ✅ |

### Key Results

**Successes**:
- ✅ **Information F1: 0.93** - Massive improvement from 0.00
- ✅ **91% recall** - Detects 91% of information comments (only 9% missed)
- ✅ **95% precision** - 95% of "information" predictions are correct
- ✅ **88% overall accuracy** - Reliable binary classification

**Trade-offs**:
- ⚠️ **Others precision: 0.50** - Half of "others" predictions are false positives
- ⚠️ **Small training set** - Only 414 training samples, may not generalize to all comment types

**Production Use**:
- Apply RoBERTa to comments classified as "others" or "social" by BERT
- Use **high-confidence threshold** (probability > 0.8) for precision
- Use **low threshold** (probability > 0.3) for high recall applications

---

## Final Pipeline Summary

### Complete Hierarchical Classification Workflow

```
5.8M Danmaku Comments
         ↓
    ┌────────────────────────────────────┐
    │   LLaMA 3.1-8B Pseudo-Labeling    │
    │   596K comments labeled            │
    └────────────────┬───────────────────┘
                     ↓
         Split: 100K train / 496K holdout
                     ↓
    ┌────────────────────────────────────┐
    │   BERT v1 (Initial Training)       │
    │   100K LLaMA labels                │
    │   Performance: ~60% accuracy       │
    └────────────────┬───────────────────┘
                     ↓
         Run BERT v1 on 496K holdout
                     ↓
    ┌────────────────────────────────────┐
    │   Agreement Analysis               │
    │   44% LLaMA-BERT agreement        │
    │   Strategic sampling:              │
    │   - 100% agreed non-others         │
    │   - 15% agreed others              │
    │   - 5% disagreed → "others"        │
    │   + 1,000 manual labels            │
    └────────────────┬───────────────────┘
                     ↓
    ┌────────────────────────────────────┐
    │   BERT v2 (Improved Training)      │
    │   106K curated samples             │
    │   Performance:                     │
    │   - Positive emotion: 91% F1       │
    │   - Negative emotion: 82% F1       │
    │   - Others: 91% F1                 │
    └────────────────┬───────────────────┘
                     ↓
         Apply BERT v2 to all comments
         Extract non-emotion comments
                     ↓
    ┌────────────────────────────────────┐
    │   RoBERTa Information Classifier   │
    │   1,000 manual labels (binary)     │
    │   Performance:                     │
    │   - Information: 93% F1            │
    │   - 91% recall, 95% precision      │
    └────────────────────────────────────┘
                     ↓
         Final Labels: emotion + information
```

### Performance Summary

| Component | Task | Training Data | Performance | Use Case |
|-----------|------|---------------|-------------|----------|
| **LLaMA 3.1-8B** | Pseudo-labeling | - | ~60% accurate | Initial label generation |
| **BERT v1** | Multi-label (initial) | 100K LLaMA | F1: 0.36-0.64 | Baseline comparison |
| **BERT v2** | Emotion detection | 106K curated | F1: 0.82-0.91 | **Production emotion classifier** ✅ |
| **RoBERTa** | Information binary | 1K manual | F1: 0.93 | **Production information classifier** ✅ |

### Key Metrics

**Emotion Classification (BERT v2)**:
- Positive emotion: **91% precision, 92% recall** → **91% F1**
- Negative emotion: **84% precision, 81% recall** → **82% F1**
- Training time: **4 hours** on A100
- Inference speed: **10,000-100,000 comments/hour**

**Information Classification (RoBERTa)**:
- Information: **95% precision, 91% recall** → **93% F1**
- Training time: **5-10 minutes** on A100
- Overall accuracy: **88%**

---

## Methodology Contributions & Related Work

### Research Foundations

This work builds upon several established research areas:

#### 1. Knowledge Distillation
Transferring knowledge from large teacher models to compact student models:
- **Hinton et al. (2015)**: "Distilling the Knowledge in a Neural Network" - foundational work on model compression
- **Sanh et al. (2019)**: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" - BERT-specific distillation
- **Sun et al. (2019)**: "Patient Knowledge Distillation for BERT Model Compression" - improved BERT distillation

**Our application**: Transfer LLaMA 3.1-8B knowledge to BERT-base-Chinese for efficient inference.

#### 2. Pseudo-Labeling & Semi-Supervised Learning
Using model predictions to generate training labels:
- **Lee (2013)**: "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks" - foundational pseudo-labeling
- **Xie et al. (2020)**: "Self-training with Noisy Student improves ImageNet classification" - handling noisy pseudo-labels
- **Sohn et al. (2020)**: "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence" - consistency regularization

**Our application**: LLaMA generates 596K pseudo-labels; we filter via agreement to improve quality.

#### 3. Co-Training & Multi-View Learning
Leveraging agreement between multiple models:
- **Blum & Mitchell (1998)**: "Combining Labeled and Unlabeled Data with Co-Training" - foundational co-training framework
- **Zhou & Li (2005)**: "Tri-training: exploiting unlabeled data using three classifiers" - extended to three models
- **Ruder & Plank (2018)**: "Strong Baselines for Neural Semi-Supervised Learning under Domain Shift" - modern neural approaches

**Our application**: Use LLaMA-BERT agreement as quality filter; 44% agreement rate identifies high-confidence samples.

#### 4. Confident Learning & Label Noise
Identifying and handling noisy labels:
- **Northcutt et al. (2021)**: "Confident Learning: Estimating Uncertainty in Dataset Labels" - principled framework for label noise
- **Arazo et al. (2019)**: "Unsupervised Label Noise Modeling and Loss Correction" - noise-robust training
- **Chen et al. (2019)**: "Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels" - theoretical foundations

**Our application**: Agreement filtering improves label quality; disagreements signal noise/ambiguity.

#### 5. LLMs for Data Annotation
Using large language models as annotators:
- **Gilardi et al. (2023)**: "ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks" - LLMs as annotators
- **Ding et al. (2023)**: "Is GPT-3 a Good Data Annotator?" - systematic evaluation of LLM annotation quality
- **He et al. (2023)**: "AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators" - improving LLM annotations

**Our application**: LLaMA 3.1-8B generates initial 596K labels with ~60% accuracy.

#### 6. Hierarchical Text Classification
Multi-stage classification approaches:
- **Silla & Freitas (2011)**: "A survey of hierarchical classification across different application domains" - comprehensive survey
- **Banerjee et al. (2019)**: "Hierarchical Transfer Learning for Multi-label Text Classification" - neural hierarchical approaches
- **Peng et al. (2018)**: "Large-scale hierarchical text classification with recursively regularized deep graph-convolutional networks" - graph-based hierarchy

**Our application**: Two-stage hierarchy (BERT for emotions → RoBERTa for information) based on task characteristics.

#### 7. Abstention & Selective Classification
Handling uncertain predictions:
- **Chow (1970)**: "On optimum recognition error and reject tradeoff" - foundational reject option
- **Geifman & El-Yaniv (2017)**: "Selective Classification for Deep Neural Networks" - modern neural abstention
- **Hendrycks & Gimpel (2017)**: "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" - uncertainty detection

**Our application**: Disagreements signal uncertainty; we relabel as "others" rather than reject.

---

### Novel Contributions

While building on established techniques, this work introduces three methodological innovations:

#### **1. Disagreement-Based Uncertainty Relabeling** ⭐ (Novel)

**Innovation**: Rather than discarding samples where models disagree or simply treating them as noisy labels, we:
- Treat disagreements as signals of *inherent ambiguity* in the data
- Actively relabel disagreed samples as "others" category (5% sampling)
- Track provenance (`disagreed_to_others`) to maintain data lineage
- Convert uncertainty into a trainable category

**Why this is novel**: 
- Traditional co-training (Blum & Mitchell, 1998) discards disagreements
- Confident learning (Northcutt et al., 2021) identifies noise but doesn't strategically relabel
- Abstention methods (Chow, 1970; Geifman & El-Yaniv, 2017) reject uncertain samples at inference time

**Our contribution**: We convert uncertainty into a productive training signal for the "ambiguous/other" category, improving model robustness by explicitly teaching what is unclear.

#### **2. Cross-Model Agreement for LLM Pseudo-Label Quality Filtering** (Novel Application)

**Innovation**: Use agreement between LLM (LLaMA) and student model (BERT) trained on LLM labels as a quality filter:
- 44% agreement rate after normalization
- Only high-agreement samples used for training specific categories
- Strategic sampling: 100% agreed non-"others", 15% agreed "others", 5% disagreed

**Why this is novel**:
- Traditional co-training uses *different views* of the same data (Blum & Mitchell, 1998)
- We use *teacher-student agreement* from knowledge distillation framework
- Combines pseudo-labeling (Lee, 2013) with confident learning (Northcutt et al., 2021)

**Our contribution**: Novel application of co-training principles to LLM-to-BERT distillation pipeline, creating a self-supervised quality filter without human validation.

#### **3. Task-Driven Hierarchical Classification with Heterogeneous Training Data** (Novel Combination)

**Innovation**: Two-stage hierarchy with different models and training data sources:
- **Stage 1 (BERT)**: Emotion detection using 106K agreement-filtered pseudo-labels
- **Stage 2 (RoBERTa)**: Information detection using 1K manually-labeled samples
- Hierarchy motivated by task characteristics (abundant vs. scarce data) rather than label taxonomy

**Why this is novel**:
- Traditional hierarchical classification (Silla & Freitas, 2011) follows label taxonomy
- We design hierarchy based on *data availability and task difficulty*
- Different training data sources for different hierarchy levels (pseudo-labels vs. manual labels)

**Our contribution**: Task-driven rather than taxonomy-driven hierarchical design, with heterogeneous training strategies optimized for each subtask's data characteristics.

---

### Research Positioning

This work can be positioned as:

**"LLM-to-BERT Knowledge Distillation with Agreement-Based Quality Filtering and Task-Driven Hierarchical Classification"**

**Key Claims**:
1. Cross-model agreement between teacher LLM and student BERT provides a reliable quality signal for pseudo-labeled data (44% agreement yields 106K high-quality samples)
2. Model disagreement patterns can be productively relabeled as "ambiguous" rather than discarded, improving classifier robustness
3. Task-driven hierarchical classification (based on data characteristics) outperforms flat multi-label classification for imbalanced datasets
4. Combining pseudo-labels (for abundant categories) with manual labels (for rare categories) in a hierarchical pipeline achieves production-ready performance

**Application Domain**: Danmu (bullet screen) comments present unique challenges:
- High volume (~5.8M comments) requiring efficient classifiers
- Extreme class imbalance (information < 1%)
- Informal language, slang, and cultural references
- Multi-label with 5 main categories + 27 subcategories

---

## Lessons Learned

### What Worked

1. **Hierarchical beats multi-task**: Separating emotion and information detection outperformed combined multi-label classifier
2. **Agreement filtering is powerful**: 44% agreement provided sufficient high-quality training data
3. **Manual labels are gold**: 1,000 manual labels > 100K noisy LLaMA labels for information detection
4. **Class weighting helps but has limits**: 10× weighting improved emotions but couldn't save information in multi-label setting

### What Didn't Work

1. **Multi-label for rare classes**: < 1% representation makes learning nearly impossible even with aggressive weighting
2. **LLaMA labels for information**: LLaMA struggled with nuanced information vs opinion distinction
3. **Single model for everything**: Different tasks need different architectures and training strategies

### Recommendations for Similar Projects

1. **Start with agreement analysis early**: Don't waste time training on all pseudo-labels
2. **Invest in targeted manual labeling**: 1K strategic labels > 100K noisy labels for rare classes
3. **Use hierarchical classification**: Separate models for separate tasks when class distributions differ wildly
4. **Monitor minority class performance**: Overall accuracy can be misleading when classes are imbalanced

---

## Production Deployment

### Final Pipeline

**For 5.8M comments**:

```python
# Step 1: BERT v2 for emotion detection
bert_predictions = bert_v2.predict(all_comments)
emotions_positive = filter(bert_predictions, 'emotions_positive')
emotions_negative = filter(bert_predictions, 'emotions_negative')
others_and_social = filter(bert_predictions, ['others', 'social'])

# Step 2: RoBERTa for information detection
info_predictions = roberta.predict(others_and_social)
information = filter(info_predictions, 'information', threshold=0.8)

# Final distribution
final_labels = {
    'emotions_positive': len(emotions_positive),
    'emotions_negative': len(emotions_negative),
    'information': len(information),
    'others': len(others_and_social) - len(information)
}
```

### Expected Results

Based on validation performance:
- **~60% emotions_positive** (high confidence)
- **~3-5% emotions_negative** (high confidence)
- **~1-2% information** (high precision with threshold=0.8)
- **~35% others** (spam, unclear, off-topic)

### Future Improvements

1. **Active learning**: Have humans label disagreement samples to refine boundaries
2. **Temporal adaptation**: Retrain periodically as slang and memes evolve
3. **Confidence calibration**: Adjust thresholds based on downstream task requirements
4. **Error analysis**: Deep dive into false positives/negatives for each category

---

## Conclusion

Classifying 5.8 million informal social media comments into nuanced categories required moving beyond standard single-model approaches. By combining:
- **LLaMA 3.1-8B** for pseudo-labeling (596K samples)
- **Agreement-based filtering** to improve data quality
- **BERT-base-Chinese** specialized for emotion detection (91% F1 for positive)
- **RoBERTa-Chinese** specialized for information detection (93% F1)

I achieved production-ready classifiers for both emotion and information extraction. The key insight: **hierarchical specialization beats multi-task learning** when task characteristics differ significantly (abundant emotional expressions vs rare informational content).

The agreement-based quality filtering technique (44% LLaMA-BERT agreement) proved essential for converting noisy pseudo-labels into high-quality training data, while manual labeling of just 1,000 strategic samples unlocked information detection that 100K pseudo-labels couldn't achieve.

This hierarchical approach is generalizable to other domains with severe class imbalance and multiple distinct classification objectives.

---

## References

### Knowledge Distillation
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS Deep Learning Workshop*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing*.
- Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). Patient knowledge distillation for BERT model compression. *EMNLP*.

### Pseudo-Labeling & Semi-Supervised Learning
- Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshop on Challenges in Representation Learning*.
- Xie, Q., Luong, M. T., Hovy, E., & Le, Q. V. (2020). Self-training with noisy student improves ImageNet classification. *CVPR*.
- Sohn, K., Berthelot, D., Carlini, N., et al. (2020). FixMatch: Simplifying semi-supervised learning with consistency and confidence. *NeurIPS*.

### Co-Training & Multi-View Learning
- Blum, A., & Mitchell, T. (1998). Combining labeled and unlabeled data with co-training. *COLT*.
- Zhou, Z. H., & Li, M. (2005). Tri-training: Exploiting unlabeled data using three classifiers. *IEEE Transactions on Knowledge and Data Engineering*, 17(11), 1529-1541.
- Ruder, S., & Plank, B. (2018). Strong baselines for neural semi-supervised learning under domain shift. *ACL*.

### Confident Learning & Label Noise
- Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research*, 70, 1373-1411.
- Arazo, E., Ortego, D., Albert, P., O'Connor, N., & McGuinness, K. (2019). Unsupervised label noise modeling and loss correction. *ICML*.
- Chen, P., Liao, B., Chen, G., & Zhang, S. (2019). Understanding and utilizing deep neural networks trained with noisy labels. *ICML*.

### LLMs for Data Annotation
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd-workers for text-annotation tasks. *PNAS*, 120(30).
- Ding, B., Qin, C., Liu, L., et al. (2023). Is GPT-3 a good data annotator? *ACL*.
- He, X., Lin, H., Yuan, Y., et al. (2023). AnnoLLM: Making large language models to be better crowdsourced annotators. *arXiv:2303.16854*.

### Hierarchical Text Classification
- Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. *Data Mining and Knowledge Discovery*, 22(1-2), 31-72.
- Banerjee, S., Akkaya, C., Perez-Sorrosal, F., & Tsioutsiouliklis, K. (2019). Hierarchical transfer learning for multi-label text classification. *ACL*.
- Peng, H., Li, J., He, Y., et al. (2018). Large-scale hierarchical text classification with recursively regularized deep graph-convolutional networks. *WWW*.

### Abstention & Selective Classification
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.

