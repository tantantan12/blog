---
layout: post
title:  "Multi-Task Learning for Email Campaign Performance Prediction"
date:   2026-02-19 10:00:00 -0600
categories: AI Machine-Learning
published: true
---

# Multi-Task Learning for Email Campaign Performance Prediction

Predicting whether customers will open, click, or convert from a single email is a challenging multi-faceted problem. In this post, we present a novel multi-task learning approach that simultaneously predicts all three outcomes using a shared neural architecture enhanced with contrastive learning.

## 1. Motivation: Campaign Performance Data

Email marketing campaigns generate rich behavioral data that can be leveraged for predictive modeling. However, traditional single-task approaches treat each prediction (Opens, Clicks, Conversions) independently, missing critical relationships between them.

### The Business Problem

Campaign teams need to understand:
- **Open rates**: Will customers engage with the email content?
- **Click-through rates**: Will they take the intended action?
- **Conversion rates**: Will the action lead to a purchase or desired outcome?

These three metrics are naturally correlated—**customers cannot click without opening, and typically cannot convert without clicking**. Yet their underlying causes differ:
- Opens depend on subject lines, send times, and sender reputation
- Clicks depend on email content, design, and call-to-action placement
- Conversions depend on landing page experience and offer relevance

### Data Characteristics

Our dataset contains **468,008 email campaign records** with:
- **Behavioral labels**: Open count, Click count, Conversion count (continuous values)
- **Campaign metadata**: Subject line, send weekday, email content
- **User features**: Purchase history (265 dimensions), promotion exposure (5-6 dimensions)
- **Label distribution**:
  - Opens: 33% positive (balanced)
  - Clicks: ~6% positive (class imbalance)
  - Conversions: ~1% positive (severe imbalance)

The severe class imbalance in Clicks and Conversions makes single-task prediction difficult—a naive model can achieve 94% accuracy simply by predicting "no click" for every sample.

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

2. **Subject Line Embedding (1,536 dim)** ⭐ NEW
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
| Input dimensions | 3,350 |
| Training set | 327,606 (70%) |
| Validation set | 70,201 (15%) |
| Test set | 70,201 (15%) |
| **Opens positive** | 153,442 (33%) |
| **Clicks positive** | 28,081 (6%) |
| **Conversions positive** | 4,680 (1%) |
| Unique campaigns | 22 |
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

**Opens (Primary Task - Well-balanced)**
- Binary label: 1 if email was opened, 0 otherwise
- Positive rate: 33%
- Predictability: Good (emails reach inbox, subject/sender matter)
- Business impact: Critical for engagement funnel

**Clicks (Secondary Task - Imbalanced)**
- Binary label: 1 if link was clicked, 0 otherwise
- Positive rate: 6%
- Predictability: Moderate (depends on content, design, CTA placement)
- Business impact: Indicates strong engagement and interest
- Challenge: Severe class imbalance (94% negative samples)

**Conversions (Tertiary Task - Severely Imbalanced)**
- Binary label: 1 if purchase/desired action completed, 0 otherwise
- Positive rate: 1%
- Predictability: Difficult (multi-step process, external factors)
- Business impact: Direct revenue impact
- Challenge: Extreme class imbalance (99% negative samples)

**Correlation Structure**:
```
Opens (33%) ⊃ Clicks (6%) ⊃ Conversions (1%)
```

Most conversions come from clicks, most clicks come from opens. This hierarchical structure is naturally captured by the shared encoder.

---

## 4. Evaluation

*(Evaluation results and performance metrics coming soon...)*

- ROC curves for all three tasks
- Precision-Recall analysis
- Confusion matrices
- Handling class imbalance results
- Comparison with single-task baseline