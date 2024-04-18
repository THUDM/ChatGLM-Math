# ChatGLM-Math：Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline

## 基本信息/Introduction

尽管当前语言模型在语言能力表现出色，但其解决数学问题的能力在现实应用中仍然面临挑战。虽然研究者开发了许多策略和数据集以增强LLMs的数学能力，但在部署的LLM系统中同时保持和提高语言和数学能力仍然是一个挑战。在这项工作中，我们定制了自我批评（Self-Critique）流程，该流程在LLM的对齐阶段解决了这一挑战。我们首先从LLM本身训练一个通用的Math-Critique模型以提供反馈信号。然后，我们依次对LLM自己的生成结果进行拒绝采样微调和直接偏好优化。基于ChatGLM3-32B，我们在学术数据集和我们新创建的挑战性数据集MathUserEval上进行了一系列实验。结果显示，我们的流程显著增强了LLM的数学问题解决能力，同时还提高了其语言能力，性能超过了可能是其两倍大的LLM。

Large language models (LLMs) have shown excellent mastering of human language, but still struggle in real-world applications that require mathematical problem-solving. While many strategies and datasets to enhance LLMs' mathematics are developed, it remains a challenge to simultaneously maintain and improve both language and mathematical capabilities in deployed LLM systems. In this work, we tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment.  We first train a general Math-Critique model from the LLM itself to provide feedback signals. Then, we sequentially employ rejective fine-tuning and direct preference optimization over the LLM's own generations for data collection. Based on ChatGLM3-32B, we conduct a series of experiments on both academic and our newly created challenging dataset, \textsc{MathUserEval}. Results show that our pipeline significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger.



PaperLink: [2404.02893.pdf (arxiv.org)](https://arxiv.org/pdf/2404.02893.pdf)


---

## MathUserEval测试集/MathUserEval Test Set

MathUserEval是一个为真实使用场景设计的测试集，针对用户关心的问题和更具挑战性的数学问题。我们的一些数据来源于大学考试题目，另一些则来源于模拟对话。对于后者，我们指派了一系列标注人员，他们根据日常使用大型模型的经验和观察向我们的系统提出数学相关的问题。

根据收集的数据分布，我们将测试集分为两个主要类别：基础数学问题和高级数学问题，并且有八个子类别。各个类别中的问题数量如下表所示。所有问题都以开放式格式提出。可能的答案包括单个数字、多个数字或数学表达式。所有的Overall分数采用的都是Macro-Average。

MathUserEval is a test set designed for real-world usage scenarios, focusing on questions that users care about and more challenging mathematical problems. Some of our data comes from university exam questions, while others are derived from simulated conversations. For the latter, we assigned a group of annotators who, based on their experience and observations using large models in daily applications, posed math-related questions to our system.

Based on the distribution of collected data, we divided the test set into two main categories: Elementary Math Problems and Advanced Math Problems, with eight subcategories in total. The number of questions in each category is as follows in the table. All questions are presented in an open-ended format. Possible answers include single numbers, multiple numbers, or mathematical expressions. All Overall scores use Macro-Average.

| Category       | Sub-Category               | Size |
| -------------- | -------------------------- | ---- |
| **Elementary** | Calculate（基础计算）      | 75   |
|                | Algebra（代数方程）        | 113  |
|                | Geometry（几何学）         | 81   |
|                | Trigonometry（三角学）     | 73   |
| **Advanced**   | Discrete Math（离散数学）  | 45   |
|                | Probability（概率统计）    | 46   |
|                | Linear Algebra（线性代数） | 58   |
|                | Calculus（微积分）         | 54   |

MathUserEval 总共包含 545 道高质量数学问题，以及22道补充交叉学科数学问题。每个样本都包含一个高质量，由标注员精心撰写的参考答案，以及在我们的分类体系中对应的类别。数据保存在`data/math-user-eval.jsonl`中，每一行都以`json`格式包含一个样本。

MathUserEval contains a total of 545 high-quality mathematics questions, along with 22 additional supplementary interdisciplinary mathematics questions. Each sample includes a high-quality reference answer carefully written by annotators, as well as the corresponding category within our classification system. The data is stored in `data/math-user-eval.jsonl`, with each line containing a sample in `json` format.

以下是一个例子：

Here is an example:

```json
{	"question_id": 163,
 	"question": "求函数f(x)=x+x^2+x^3的原函数",
 	"reference": "函数f(x)=x+x^2+x^3的原函数为F(x)=1\\/2*x^2+1\\/3*x^3+1\\/4*x^4+C（C为常数）。\n",
 	"category": "高等数学",
 	"subcategory": "calculus"}
```

---

## 评价方法/Metric

为了有效评估响应的质量，MathUserEval 目前采用 GPT-4-1106-Preview 来分析并随后对响应进行评分。评测方法与AlignBench的逻辑推理类题目保持一致。

To effectively evaluate the quality of responses, MathUserEval currently utilizes GPT-4-1106-Preview to analyze and subsequently score the responses. The evaluation method is consistent with the logic reasoning type questions of AlignBench.


---

## 如何使用MathUserEval评测模型/How to use MathUserEval

整个评估过程包含三个步骤：获取待评测模型的生成结果、调用评价模型获取分析和打分，最终计算结果。相应的脚本保存在`scripts`中，可以修改其中参数之后调用。

The entire evaluation process consists of three steps: obtaining the generation results from the model being evaluated, calling the evaluation model for analysis and scoring, and finally calculating the outcomes. The corresponding scripts are saved in `scripts` and can be called after modifying the parameters. 

1. **步骤一** 获取待评测模型的生成结果

   首先，您需要获得待评测模型的 API 来生成结果，如果是开源模型，您需要自己部署成可以调用获得回复的 API。（此部分不包含在此仓库中）。

   其次，在`inference/api_models`中实现您自己的 API 调用类，`do_nothing`类可以作为一个示例。（此类主要用于调用 API，注意 API 类名应与文件名相同）

   第三，修改参数并运行以下脚本以获得待评测模型的生成结果。

   **Step One:** Obtaining the Generation Results of the Model Being Evaluated

   First, you need to obtain the API of the model being evaluated to generate results. If it is an open-source model, you need to deploy it yourself to call and receive replies. (This part is not included in this repository).

   Secondly, implement your own API calling class in `inference/api_models`, where the `do_nothing` class can serve as an example. (This class is mainly used for calling APIs, note the API class name should match the file name).

   Thirdly, modify the parameters and run the following script to obtain the generation results of the model being evaluated.

   ```bash
   MODEL=do_nothing # TODO: Modify the model name (same as your API calling class)
   
   python get_answers.py \
       --model do_nothing \
       --workers 1 \
       --question-file data/math-user-eval.jsonl \
       --save-dir data/model_answer
   ```

   待评测模型的回复将被保存在`data/model_answer`中，以备下一步的评测。

   The replies from the model being evaluated will be saved in `data/model_answer` for the next step of evaluation.

2. **步骤二** 调用评价模型获取分析和打分

   目前我们使用 `gpt-4-1106-preview` 作为评测模型，之后为了方便中文社区，我们计划以 API 的形式开放 `Math-Critique` 作为`gpt-4-1106-preview`  的替代评测模型给研究人员使用。

   首先，在`config/mathusereval.json`中填写您的 OpenAI API 密钥。

   然后，修改并运行以下脚本以获得评价模型的评测结果。

   **Step Two:** Calling the Evaluation Model for Analysis and Scoring

   Currently, we use `gpt-4-1106-preview`  as the evaluation model. Later, for the convenience of the Chinese community, we plan to offer`Math-Critique`  as an alternative to ``gpt-4-1106-preview` ` for researchers to use in the form of an API.

   First, fill in your OpenAI API key in `config/mathusereval.json`.

   Then, modify and run the following script to obtain the evaluation results from the evaluation model.

   ```bash
   MODEL=do_nothing # TODO: Modify the model name (same as your API calling class)
   
   python judge.py \
       --config-path config/mathusereval.json \
       --model-name $MODEL \
       --parallel 1 \
   ```

   评测结果将保存在`data/judgment`

   The evaluation results will be saved in `data/judgment`.

3. **步骤三** 最终计算结果

   运行以下脚本以获取保存在`data/judgment`中的所有模型的最终结果。

   **Step Three:** Final Calculation of Results
   
   Run the following script to obtain the final results of all models saved in `data/judgment`.
   
   ```bash
   python show_result.py \
       --input-dir data/judgment \
       --ques-file data/data_release.jsonl \
       --save-file data/results/results.xlsx
   ```
   

---

## 排行榜/Leaderboard

| Model                           | Overall  | Elementary |             |               |          |          | Advanced |              |              |             |           |
| ------------------------------- | -------- | ---------- | ----------- | ------------- | -------- | -------- | -------- | ------------ | ------------ | ----------- | --------- |
|                                 |          | **Avg**    | **algebra** | **calculate** | **geo.** | **tri.** | **Avg**  | **calculus** | **discrete** | **linear.** | **Prob.** |
| GPT-4-0125-Preview              | **5.79** | **5.26**   | **5.04**    | **7.63**      | **3.98** | 4.59     | 6.71     | 7.26         | 6.62         | **5.48**    | 7.72      |
| GPT-4-1106-Preview              | 5.73     | 5.07       | 4.96        | 7.00          | 3.78     | 4.71     | **6.81** | **7.39**     | **6.96**     | 5.29        | **7.91**  |
| GLM-4                           | 5.11     | 4.86       | 4.47        | 6.56          | 3.95     | **4.74** | 5.43     | 6.00         | 5.67         | 4.26        | 6.02      |
| ChatGLM3-32B-SFT-2312 + RFT&DPO | 4.23     | 4.01       | 3.88        | 5.41          | 2.90     | 3.99     | 4.59     | 5.22         | 4.76         | 3.38        | 5.20      |
| GPT-4-0613                      | 4.14     | 3.34       | 2.88        | 4.76          | 3.17     | 2.78     | 5.33     | 5.57         | 5.49         | 4.26        | 6.22      |
| ChatGLM3-32B-SFT-2312 + RFT     | 4.01     | 3.86       | 3.84        | 5.37          | 2.57     | 3.77     | 4.26     | 4.72         | 4.69         | 2.98        | 4.89      |
| Qwen-72B-Chat                   | 3.87     | 3.99       | 3.96        | 4.81          | 3.83     | 3.34     | 3.67     | 4.54         | 3.71         | 2.84        | 3.65      |
| GPT-3.5-Turbo-0613              | 3.42     | 3.04       | 2.81        | 4.07          | 2.23     | 3.26     | 4.07     | 4.83         | 4.38         | 3.26        | 3.91      |
| ChatGLM3-32B-SFT-2312           | 3.39     | 3.35       | 3.35        | 4.51          | 2.51     | 3.11     | 3.44     | 4.04         | 4.38         | 2.41        | 3.13      |
| Claude-2                        | 3.29     | 2.63       | 2.35        | 3.63          | 2.20     | 2.53     | 4.35     | 4.56         | 4.53         | 3.29        | 5.28      |
| DeepSeek-Chat-67B               | 3.24     | 2.76       | 2.21        | 4.73          | 2.12     | 2.30     | 3.84     | 4.41         | 4.82         | 2.79        | 3.52      |
| Yi-34B-Chat                     | 2.64     | 2.49       | 2.04        | 3.61          | 2.25     | 2.27     | 2.87     | 2.80         | 3.47         | 2.03        | 3.41      |

## Citation

If you find our work helpful, please consider citing the following papers.

```
@misc{xu2024chatglmmath,
      title={ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline}, 
      author={Yifan Xu and Xiao Liu and Xinghan Liu and Zhenyu Hou and Yueyan Li and Xiaohan Zhang and Zihan Wang and Aohan Zeng and Zhengxiao Du and Wenyi Zhao and Jie Tang and Yuxiao Dong},
      year={2024},
      eprint={2404.02893},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

