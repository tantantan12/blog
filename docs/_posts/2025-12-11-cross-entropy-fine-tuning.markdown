---
layout: post
title:  "Using LLM for Categorization Tasks"
date:   2025-12-11 14:34:15 -0600
categories: LLM
---

## Motivation
We've learned how to (1) use embeddings to categorize text and use XGBoost to perform the text categorization (2) fine-tune pre-trained Bert model to perform text categorization. Today, we learn how to fine-tune an LLM based on cross-entropy to perform text categorization with subtle information cues.


## Large Language Model
We use Lamma 3.2


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

## LLaMA Model Labeling Process

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

## BERT Fine-tuning Methodology

### Model Architecture
We fine-tuned **BERT-base-Chinese** for multi-label classification using the LLaMA-generated labels as training data:

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

### Training Data Preparation
From the LLaMA-labeled dataset (595K examples), we:
1. **Sampled 100K examples** for efficient training (1-day target on A100)
2. **Created binary label vectors** for both main categories and subcategories
3. **Split data**: 80% training, 20% validation
4. **Tokenized text** with maximum length 512 tokens

### Training Configuration
- **Model**: bert-base-chinese
- **Optimizer**: AdamW with learning rate 2e-5
- **Batch size**: 64 (optimized for A100)
- **Epochs**: 3
- **Loss function**: BCEWithLogitsLoss for multi-label classification
- **Scheduler**: Linear warmup with 10% warmup steps

### Training Process
```python
# Multi-task loss combining main and subcategory classification
main_loss = criterion(main_logits, main_labels)
subcat_loss = criterion(subcat_logits, subcat_labels)
total_loss = main_loss + subcat_loss
```

### Inference Optimization
For fast inference on remaining unlabeled comments:
- **Batch size**: 256 for maximum GPU utilization
- **Processing speed**: ~10,000-100,000 comments/hour vs. 900/hour with LLaMA
- **Memory efficiency**: Process 5.25M remaining comments in hours instead of months

## Performance Comparison

### BERT Fine-tuning Results

#### Training Progress
The BERT model was successfully fine-tuned over 3 epochs with consistent loss reduction:

| Epoch | Training Loss | Training Time | Notes |
|-------|---------------|---------------|-------|
| 1     | 0.5954       | ~24 minutes   | Initial baseline |
| 2     | 0.4580       | ~24 minutes   | 23% improvement |
| 3     | 0.4202       | ~24 minutes   | **Best model** |

**Total Training Time**: ~72 minutes on A100 GPU  
**Final Test Loss**: 0.4672

#### Classification Performance by Category

| Category | Precision | Recall | F1-Score | Support | Performance Notes |
|----------|-----------|---------|----------|---------|-------------------|
| **information** | 0.52 | 0.27 | 0.36 | 3,238 | Moderate precision, low recall |
| **emotions_positive** | 0.68 | 0.62 | 0.64 | 4,612 | **Best performing category** |
| **emotions_negative** | 0.55 | 0.46 | 0.50 | 2,417 | Balanced but moderate |
| **social** | 0.43 | 0.00 | 0.01 | 992 | **Challenging category** |
| **other** | 0.60 | 0.55 | 0.57 | 8,165 | Largest category, stable |

#### Overall Metrics

| Metric Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| **Micro Average** | 0.61 | 0.48 | 0.54 |
| **Macro Average** | 0.56 | 0.38 | 0.42 |
| **Weighted Average** | 0.59 | 0.48 | 0.52 |
| **Samples Average** | 0.46 | 0.46 | 0.46 |

### Key Insights

**Strengths:**
- **Emotions_positive** category achieved the highest performance (F1: 0.64), indicating BERT effectively captures positive sentiment expressions
- **Training efficiency**: Model converged in just 72 minutes on A100, making it highly practical for production use
- **Consistent improvement**: Training loss decreased smoothly from 0.60 → 0.42, showing stable learning

**Challenges:**
- **Social category** shows near-zero recall (0.00), suggesting this category may need better definition or more training examples
- **Information category** has low recall (0.27), indicating difficulty in identifying informational content
- **Class imbalance**: The "other" category dominates with 8,165 samples vs. "social" with only 992

**Speed Comparison:**
- **LLaMA inference**: ~900 comments/hour (current bottleneck)
- **BERT inference**: ~10,000-100,000 comments/hour (100x faster)
- **Production impact**: Can process remaining 5.25M comments in days instead of months

### Next Steps
1. **Address social category**: Investigate low recall through error analysis
2. **Data balancing**: Consider upsampling underrepresented categories
3. **Production deployment**: Use BERT for remaining 5.25M unlabeled comments
4. **Quality validation**: Sample and manually verify BERT predictions vs. LLaMA labels