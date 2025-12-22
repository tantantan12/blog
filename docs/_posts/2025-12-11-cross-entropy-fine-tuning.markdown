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
 

## Comparison across Methods