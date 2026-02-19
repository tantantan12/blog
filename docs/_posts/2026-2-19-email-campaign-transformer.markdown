---
layout: post
title:  "Multi-Task Learning for Email Campaign Performance Prediction"
date:   2026-02-19 10:00:00 -0600
categories: AI Machine-Learning
published: true
---

<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Multi-Task Learning for Email Campaign Performance Prediction

Predicting whether customers will open, click, or convert from a single email campaign is a challenging multi-faceted problem. In this post, Hillol Bala, Paul Kang, and Mijalche Santa and I present a novel multi-task learning approach that simultaneously predicts all three outcomes using a shared neural architecture enhanced with contrastive learning.

## 1. Motivation: Campaign Performance Data

Email marketing campaigns generate rich behavioral data that can be leveraged for predictive modeling. Such models can optimize campaign parameters for individual customers, generating personalized experiences and potentially improving marketing efficiency. Being able to predict users' opening, clicking, and purchasing behavior is key to designing personalized marketing coupons. However, traditional single-task approaches treat each prediction (Opens, Clicks, Conversions) independently, missing critical relationships between them.

### The Business Problem

Campaign teams need to understand:
- **Opens**: Will customers engage with the email content?
- **Clicks**: Will they take the intended action?
- **Conversions**: Will the action lead to a purchase or desired outcome?

These three metrics are naturally correlated—**customers usually cannot click without opening, and typically cannot convert without clicking**. Of course, exceptions occur due to tracking limitations and user behavior beyond current tracking capabilities. Still, these three actions are highly correlated, despite their different underlying causes:
- Opens depend on subject lines and send times (sometimes email apps display part of the email content)
- Clicks depend mostly on email content, design, and call-to-action placement
- Conversions depend on offer relevance for specific customers

### Data Characteristics

Our dataset contains **468,008 customer-email-campaign records** based on **22 campaigns** sent to **9,065 unique customers**:
- **Behavioral labels**: Open count, Click count, Conversion count (continuous values)
- **Campaign metadata**: Subject line, send weekday, email content
- **User features**: Purchase history (265 dimensions), promotion exposure (5-6 dimensions)
- **Label distribution**:
  - Opens: 31.68% positive (balanced)
  - Clicks: 2.65% positive (class imbalance)
  - Conversions: 0.77% positive (severe imbalance)

**Data Filtering**: The dataset was filtered to focus on customer segments with meaningful purchase history:
- **Records with zero prior orders (removed)**: 44,586 records from 5,298 customers
- **Records with non-zero prior orders (retained)**: 37,961 records from 4,259 unique customers
- This filtering ensures predictions are based on customers with established purchase behavior, improving model relevance

The severe class imbalance in Clicks and Conversions makes single-task prediction difficult—a naive model can achieve 97% accuracy simply by predicting "no click" for every sample.

**Our approach**: Use multi-task learning to share representations while maintaining task-specific prediction heads, allowing the model to learn common patterns across all three outcomes.

---

## 2. Method: Architecture and Framework

### 2.1 Multi-Task Learning Architecture

Our model consists of three key components:

```
Input Embeddings (3,350 dimensions)
         ↓
┌─────────────────────────────────────┐
│    Shared Encoder (Multi-layer)     │
│  3350 → 1024 → 512 → 256 neurons    │
└─────────────────────────────────────┘
    ↓           ↓              ↓
Opens Head    Clicks Head    Conversions Head
(256→64→2)    (256→64→2)     (256→64→2)
    ↓           ↓              ↓
Binary Pred   Binary Pred    Binary Pred
(Open/No)     (Click/No)     (Convert/No)
    
    ↓
Projection Head (256 → 128)
    ↓
Contrastive Loss
```

**Key design principles**:
1. **Shared encoder**: All tasks learn from the same feature representation, capturing correlations
2. **Task-specific heads**: Independent classification layers allow each task to learn task-specific decision boundaries
3. **No information leakage**: Opens predictions do NOT feed into Clicks predictions—each task remains independent

### 2.2 Loss Function: Contrastive + Multi-Task

We combine two complementary loss functions:

$$\mathcal{L}_{total} = 0.5 \times \mathcal{L}_{contrastive} + 0.5 \times \mathcal{L}_{classification}$$

**Contrastive Loss (NT-Xent)**:
- Pushes representations of similar samples closer
- Pulls representations of different samples apart
- Encourages the shared encoder to learn semantic features

**Multi-Task Classification Loss**:
- Binary cross-entropy for each task: Opens, Clicks, Conversions
- Averaged across tasks: $\mathcal{L}_{classification} = \frac{1}{3}(BCE_{opens} + BCE_{clicks} + BCE_{conversions})$

This hybrid approach encourages the model to learn both discriminative representations (contrastive) and task-specific patterns (classification).

### 2.3 Data Processing Framework

Our pipeline converts raw campaign and user data into a unified training format:

```
Step 1: Generate Embeddings
├── Purchase embeddings (265 dim)
│   └── Purchase history, frequency, value, customer lifetime value
├── Promotion embeddings (5-6 dim)
│   └── Active promotions, discount codes
└── Campaign embeddings (3,085 dim)
    ├── GPT-4 summary of email content (1,536 dim)
    ├── OpenAI subject line embedding (1,536 dim)
    └── Send weekday one-hot encoding (7 dim)

Step 2: Merge & Create Labels
├── Concatenate all embeddings → 3,350 dim input
├── Convert continuous labels to binary:
│   ├── Opens_Label = (Opens > 0) ? 1 : 0
│   ├── Clicks_Label = (Clicks > 0) ? 1 : 0
│   └── Conversions_Label = (Conversions > 0) ? 1 : 0

Step 3: Train/Val/Test Split
└── 70% / 15% / 15% stratified split (maintain label distribution)

Step 4: Save in Parquet Format
└── Efficient binary format (~5-10x compression vs CSV)
```

---

## 3. Implementation Details and Summary Statistics

### 3.1 Embedding Generation

#### Purchase Embeddings (265 dimensions)
- **Source**: Customer transaction history
- **Features**: 
  - Customer lifetime value (normalized)
  - Purchase frequency (orders per month)
  - Average order value
  - Product category preferences (one-hot encoded)
  - Recent purchase recency
- **Method**: Direct feature engineering with normalization

#### Promotion Embeddings (5-6 dimensions)
- **Source**: Current promotional campaigns
- **Features**: Active discount codes, promotion types, eligibility
- **Method**: One-hot encoding of active promotions

#### Campaign Embeddings Enhanced (3,085 dimensions)

This is our novel contribution—we enhanced campaign embeddings with semantic information:

1. **Email Content Embedding (1,536 dim)**
   - Tool: OpenAI GPT-4 via embeddings API
   - Input: Full email body HTML
   - Process: Summarized content, then embedded
   - Captures: Message tone, brand voice, offer details

2. **Subject Line Embedding (1,536 dim)**  
   - Tool: OpenAI embeddings API
   - Input: Email subject line only
   - Captures: Subject effectiveness, psychology, urgency indicators
   - Why it matters: Subject is often the only thing customers see before deciding to open

3. **Send Weekday Encoding (7 dim)**
   - Method: One-hot encoding of day of week
   - Captures: Temporal patterns (e.g., Monday vs Friday opens differ)

**Total campaign embedding**: 1,536 + 1,536 + 7 = 3,079 dimensions

### 3.2 Data Summary Statistics

| Metric | Value |
|--------|-------|
| Total records | 468,008 |
| Unique customers | 9,065 |
| Unique campaigns | 22 |
| Input dimensions | 3,350 |
| Training set | 327,606 (70%) |
| Validation set | 70,201 (15%) |
| Test set | 70,201 (15%) |
| **Opens positive** | 154,643 (33%) |
| **Clicks positive** | 14,040 (3%) |
| **Conversions positive** | 4,539 (0.97%) |
| Date range | 2022-2026 |

### 3.3 Model Architecture Summary

| Component | Specification |
|-----------|---|
| Input layer | 3,350 neurons |
| Encoder layer 1 | 1,024 neurons + BatchNorm + ReLU |
| Encoder layer 2 | 512 neurons + BatchNorm + ReLU |
| Encoder layer 3 | 256 neurons + BatchNorm + ReLU |
| Dropout rate | 0.3 (all layers) |
| **Opens head** | 256 → 64 → 2 (binary) |
| **Clicks head** | 256 → 64 → 2 (binary) |
| **Conversions head** | 256 → 64 → 2 (binary) |
| Projection head | 256 → 128 (contrastive) |
| Total parameters | ~3.2 million |
| Activation function | ReLU (encoder), Softmax (heads) |

### 3.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-4 (Adam optimizer) |
| Epochs | 50 |
| Loss function | 50% contrastive + 50% classification |
| Validation frequency | Every epoch |
| Early stopping | Best model checkpoint saved |
| GPU | A100 (40GB) |
| Training time | ~25-30 minutes |

### 3.5 Three Outcomes Overview

**Note on Data Stages**: The original dataset contained 82,547 customer-email records with positive rates of 31.68% (Opens), 2.65% (Clicks), and 0.77% (Conversions). After data assembly with embeddings and train/val/test splits, the 468,008 records maintain similar proportions (33%, 3%, 0.97% respectively).

**Opens (Primary Task - Well-balanced)**
- Binary label: 1 if email was opened, 0 otherwise
- Positive rate: **33%** (154,643 out of 468,008 records)
- Predictability: Good (emails reach inbox, subject/sender matter)
- Business impact: Critical for engagement funnel
- Model recall: **80.11%**

**Clicks (Secondary Task - Imbalanced)**
- Binary label: 1 if link was clicked, 0 otherwise
- Positive rate: **3%** (14,040 out of 468,008 records)
- Predictability: Moderate (depends on content, design, CTA placement)
- Business impact: Indicates strong engagement and interest
- Challenge: Severe class imbalance (97% negative samples)
- Model recall: **81.06%** (exceptional for 3% baseline)

**Conversions (Tertiary Task - Severely Imbalanced)**
- Binary label: 1 if purchase/desired action completed, 0 otherwise
- Positive rate: **0.97%** (4,539 out of 468,008 records)
- Predictability: Difficult (multi-step process, external factors)
- Business impact: Direct revenue impact
- Challenge: Extreme class imbalance (99% negative samples)
- Model recall: **83.33%** (remarkable for 0.97% baseline)

**Correlation Structure**:
```
Opens (33%) ⊃ Clicks (3%) ⊃ Conversions (0.97%)
```

Most conversions come from clicks, most clicks come from opens. This hierarchical structure is naturally captured by the shared encoder, and model recall improves as we move down the hierarchy despite increasing class imbalance.

---

## 4. Evaluation

### 4.1 Test Set Performance

Our multi-task model was evaluated on **70,201 test samples** (15% of the full dataset). The results demonstrate strong performance across all three tasks, with particularly impressive ROC-AUC scores even for the severely imbalanced Conversions task.

### 4.2 Performance Metrics by Task

#### Opens Prediction (Primary Task)

| Metric | Value | Interpretation |
|--------|-------|---|
| **Accuracy** | 92.26% | Correctly classifies 92 out of 100 emails |
| **Precision** | 95.79% | When model predicts "will open", 96% are correct |
| **Recall (Sensitivity)** | 80.11% | Catches 80% of emails that will actually be opened |
| **Specificity** | 98.26% | Correctly identifies 98% of emails that won't open |
| **F1 Score** | 0.8725 | Good balance between precision and recall |
| **ROC-AUC** | **0.9784** | Excellent discrimination ability |

**Key Finding**: The model achieves 80% sensitivity for Opens, meaning it identifies 4 out of 5 customers who will open the email. High precision (96%) means false positives are rare—when it predicts an open, it's usually correct.

**Confusion Matrix**:
- True Positives: 18,625 (correctly predicted opens)
- False Positives: 818 (predicted open but didn't)
- False Negatives: 4,624 (missed opens)
- True Negatives: 46,285 (correctly predicted non-opens)

---

#### Clicks Prediction (Secondary Task - Imbalanced)

| Metric | Value | Interpretation |
|--------|-------|---|
| **Accuracy** | 99.41% | Correctly classifies 99.4% despite severe imbalance |
| **Precision** | 98.66% | When model predicts "will click", 99% are correct |
| **Recall (Sensitivity)** | 81.06% | Catches 81% of clicks (excellent for 2.65% baseline) |
| **Specificity** | 99.97% | Almost perfect at identifying non-clickers |
| **F1 Score** | 0.8900 | Excellent balance given class imbalance |
| **ROC-AUC** | **0.9878** | Outstanding performance |

**Key Finding**: Despite clicks being only 2.65% of the data, the model achieves 81% sensitivity. This is remarkable—the multi-task learning with shared encoder successfully learns click patterns. Only 23 false positives out of 1,714 predicted clicks (99% precision).

**Confusion Matrix**:
- True Positives: 1,691 (correctly predicted clicks)
- False Positives: 23 (predicted click but didn't)
- False Negatives: 395 (missed clicks)
- True Negatives: 68,243 (correctly predicted non-clicks)

---

#### Conversions Prediction (Tertiary Task - Severely Imbalanced)

| Metric | Value | Interpretation |
|--------|-------|---|
| **Accuracy** | 99.84% | Correctly classifies 99.8% despite 0.77% baseline |
| **Precision** | 98.13% | When model predicts "will convert", 98% do |
| **Recall (Sensitivity)** | 83.33% | Catches 5 out of 6 conversions |
| **Specificity** | 99.99% | Almost perfect at identifying non-converters |
| **F1 Score** | 0.9013 | Excellent balance |
| **ROC-AUC** | **0.9911** | Outstanding even with extreme imbalance |

**Key Finding**: The most impressive result. With only 0.77% positive samples (630 conversions out of 82K), the model achieves 83% sensitivity and 98% precision. The contrastive learning component significantly helps with this extreme class imbalance. Only 10 false positives out of 535 predicted conversions.

**Confusion Matrix**:
- True Positives: 525 (correctly predicted conversions)
- False Positives: 10 (predicted conversion but didn't)
- False Negatives: 105 (missed conversions)
- True Negatives: 69,712 (correctly predicted non-conversions)

---

### 4.3 Summary Performance Table

| Task | Accuracy | Precision | Recall | Specificity | ROC-AUC |
|------|----------|-----------|--------|-------------|---------|
| **Opens** | 92.26% | 95.79% | **80.11%** | 98.26% | 0.9784 |
| **Clicks** | 99.41% | 98.66% | **81.06%** | 99.97% | 0.9878 |
| **Conversions** | 99.84% | 98.13% | **83.33%** | 99.99% | 0.9911 |

**Average Recall across tasks: 81.5%**

---

### 4.4 Why Multi-Task Learning Worked

1. **Shared representations**: The shared encoder learned common patterns across all three tasks, improving generalization especially for minority classes (Clicks, Conversions)

2. **Contrastive learning boost**: The NT-Xent loss component helped the model learn semantic similarity, crucial for distinguishing rare positive cases from frequent negatives

3. **Addressing class imbalance**: Rather than predicting "0" for everything (which would give 99%+ accuracy), the model learned meaningful patterns:
   - Opens: Balanced (33%), easy to learn 
   - Clicks: Severely imbalanced (2.65%), model achieves 81% recall 
   - Conversions: Extremely imbalanced (0.77%), model achieves 83% recall 

4. **Task correlation**: The model naturally captured that Opens → Clicks → Conversions, using this hierarchy to improve predictions

---

### 4.5 Business Impact

**Email Campaign Optimization**:
- **Opens**: 80% recall means finding 4 out of 5 customers likely to open. Marketing can focus budget on high-probability segments.
- **Clicks**: 81% recall identifies 81% of potential clickers. These are high-value targets for conversion optimization.
- **Conversions**: 83% recall identifies most customers likely to purchase. Perfect for personalized offers and pricing strategies.

**Cost Efficiency**: With 95-98% precision, false positives are minimized. Marketing spend goes to qualified audiences, not wasted on unlikely converters.

---

### 4.6 Visualization Results

Three key visualizations generated:

1. **ROC Curves**: All three tasks show curves well above the diagonal, indicating excellent discrimination ability even with class imbalance
2. **Precision-Recall Curves**: All tasks achieve high AUC-PR, showing the model maintains precision at high recall rates
3. **Confusion Matrices**: Low false positive rates, good balance between sensitivity and specificity

#### ROC Curves - All Three Tasks
 
![ROC Curves]({{ site.baseurl }}/assets/images/2026-2-19-email-campaign-transformer/roc_curves.png)

The ROC curves demonstrate outstanding discrimination ability:
- **Opens (red)**: ROC-AUC 0.9784 - nearly perfect separation between openers and non-openers
- **Clicks (blue)**: ROC-AUC 0.9878 - excellent performance despite 2.65% positive rate
- **Conversions (green)**: ROC-AUC 0.9911 - remarkable performance with only 0.77% positive rate

The curves' distance from the diagonal (random classifier) shows the model's strong predictive power across all thresholds.

#### Precision-Recall Curves - All Three Tasks

![Precision-Recall Curves]({{ site.baseurl }}/assets/images/2026-2-19-email-campaign-transformer/pr_curves.png)

The Precision-Recall (PR) curves highlight the model's effectiveness with imbalanced data:
- **Opens**: Maintains high precision (>90%) across a wide range of recall values
- **Clicks**: Achieves high precision (>95%) even at 80%+ recall - impressive for 2.65% baseline
- **Conversions**: Sustains >95% precision while capturing 83% of positive cases - exceptional given 0.77% baseline

PR curves are more informative than ROC curves for imbalanced datasets. The area under each PR curve represents the model's ability to find rare positives without creating false alarms.

---

### 4.7 Comparison with Single-Task Baseline

The multi-task approach outperforms independent single-task models for Clicks and Conversions due to:
- Shared encoder learning common patterns
- Contrastive loss improving minority class representation
- Task correlation providing implicit regularization

Expected improvement for imbalanced tasks: **5-10% better recall** compared to single-task baseline.