# Endovascular-Stroke-ML: Stroke Outcome Prediction Tool

这是一个基于机器学习的临床辅助工具，旨在预测缺血性中风患者在接受血管内治疗（EVT）后的功能预后（mRS 评分）。

## 📝 项目简介
本项目通过分析患者的临床特征（如年龄、血压、NIHSS 评分等），利用集成学习模型（Voting Classifier）提供个性化的预后概率评估。

- **应用目标**：评估 mRS (modified Rankin Scale) 良好预后的概率。
- **技术栈**：Python, Streamlit, Scikit-learn, SHAP.
- **在线访问**：[在这里插入你的 Streamlit Cloud 部署链接]

## 🚀 核心功能
- **风险预测**：输入患者临床指标，实时计算良好预后概率。
- **风险分层**：根据概率自动进行绿/黄/红三级风险预警。
- **模型可解释性**：集成 SHAP 特征重要性分析，透明展示各指标对预测结果的影响。

## 📊 模型表现
项目包含了详细的模型评估指标：
- **ROC 曲线**：展示模型的分类效能。
- **SHAP Summary**：展示前 7 大核心影响因素（如 Baseline NIHSS, Age 等）。

## 🛠 如何在本地运行
如果你想在本地环境运行此应用：

1. 克隆仓库：
   ```bash
   git clone [https://github.com/lonzoohner-bot/Endovascular-Stroke-ML.git](https://github.com/lonzoohner-bot/Endovascular-Stroke-ML.git)
