# HCC Prediction System - Quick Reference Card

## Quick Start
1. Go to **Start Prediction**
2. Enter patient data (Age, AFP, Tumor Size required)
3. Click **Predict Recurrence Risk**
4. Review results and SHAP explanation

## Risk Levels at a Glance

| Risk   | Probability | Action                     |
| ------ | ----------- | -------------------------- |
| LOW    | <40%        | Annual surveillance        |
| MEDIUM | 40-70%      | Quarterly imaging          |
| HIGH   | >70%        | Urgent review, monthly AFP |

## Key Risk Factors

| High Risk If...   | Why                     |
| ----------------- | ----------------------- |
| AFP > 400 ng/mL   | Aggressive tumor marker |
| Tumor > 5 cm      | Higher invasion risk    |
| Cirrhosis present | Field effect            |
| Age > 65          | Reduced hepatic reserve |

## Reading SHAP Charts

- Red bars = Increases risk
- Green bars = Decreases risk
- Longer bars = Stronger impact
- Base value = Average patient risk

## Remember

- This is a decision support tool
- Use with clinical judgment
- Consider all patient factors
- Not a replacement for diagnosis