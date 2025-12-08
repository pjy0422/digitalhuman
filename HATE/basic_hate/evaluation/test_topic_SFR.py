"""
基于SFR-Embedding-Mistral的主题距离分析器
使用sentence_transformers库和Salesforce/SFR-Embedding-Mistral模型计算主题句与段落之间的余弦相似度距离

依赖安装：
pip install sentence-transformers scikit-learn matplotlib seaborn numpy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Union, Tuple
import warnings
import torch
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

def generate_trend_summary(correlation: float, slope: float, significant: bool, strength: str) -> str:
    """生成趋势分析的文字总结"""
    direction_map = {
        'increasing': '上升',
        'decreasing': '下降', 
        'flat': '平稳'
    }
    
    strength_map = {
        'very_strong': '非常强',
        'strong': '强',
        'moderate': '中等',
        'weak': '弱',
        'very_weak': '非常弱'
    }
    
    if not significant:
        return f"未检测到显著趋势 (相关系数: {correlation:.3f}, 不显著)"
    
    direction = '上升' if slope > 0 else '下降' if slope < 0 else '平稳'
    strength_zh = strength_map.get(strength, strength)
    
    return f"检测到{strength_zh}的{direction}趋势 (相关系数: {correlation:.3f}, 斜率: {slope:.4f})"


def analyze_list_trend(data: List[Union[int, float]], 
                      show_plot: bool = True, 
                      plot_title: str = "List Trend Analysis") -> Dict[str, Union[float, str]]:
    """
    分析列表元素随索引变化的趋势
    
    Parameters:
    -----------
    data : List[Union[int, float]]
        要分析的数据列表
    show_plot : bool, optional
        是否显示可视化图表，默认True
    plot_title : str, optional
        图表标题
        
    Returns:
    --------
    Dict[str, Union[float, str]]
        包含各种趋势分析指标的字典
    """
    if len(data) < 2:
        raise ValueError("列表长度必须至少为2")
    
    # 创建索引数组
    indices = np.array(range(len(data)))
    values = np.array(data)
    
    # 1. 皮尔逊相关系数
    correlation, p_value = stats.pearsonr(indices, values)
    
    # 2. 线性回归分析
    X = indices.reshape(-1, 1)
    reg = LinearRegression().fit(X, values)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r_squared = reg.score(X, values)
    
    # 3. 斯皮尔曼等级相关系数（对非线性趋势更敏感）
    spearman_corr, spearman_p = stats.spearmanr(indices, values)
    
    # 4. 趋势强度分类
    def classify_trend_strength(corr_coef: float) -> str:
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    # 5. 趋势方向
    def get_trend_direction(slope_val: float, significance: bool) -> str:
        if not significance:
            return "no_significant_trend"
        elif slope_val > 0:
            return "increasing"
        elif slope_val < 0:
            return "decreasing"
        else:
            return "flat"
    
    # 6. 统计显著性
    is_significant = p_value < 0.05
    
    # 7. 计算变化率
    if len(data) > 1:
        total_change = values[-1] - values[0]
        avg_change_per_step = total_change / (len(data) - 1)
        percent_change = (total_change / abs(values[0])) * 100 if values[0] != 0 else float('inf')
    else:
        total_change = 0
        avg_change_per_step = 0
        percent_change = 0
    
    # 8. 单调性检查
    def check_monotonicity(arr: np.ndarray) -> str:
        diffs = np.diff(arr)
        if np.all(diffs >= 0):
            return "monotonic_increasing"
        elif np.all(diffs <= 0):
            return "monotonic_decreasing"
        else:
            return "non_monotonic"
    
    monotonicity = check_monotonicity(values)
    
    # 9. 趋势稳定性（基于残差分析）
    predicted_values = reg.predict(X)
    residuals = values - predicted_values
    trend_stability = 1 - (np.std(residuals) / np.std(values)) if np.std(values) != 0 else 1
    
    # 构建结果字典
    result = {
        # 基本统计
        'length': len(data),
        'start_value': float(values[0]),
        'end_value': float(values[-1]),
        'min_value': float(np.min(values)),
        'max_value': float(np.max(values)),
        'mean_value': float(np.mean(values)),
        'std_value': float(np.std(values)),
        
        # 趋势分析
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        
        # 回归分析
        'linear_slope': slope,
        'linear_intercept': intercept,
        'r_squared': r_squared,
        
        # 变化分析
        'total_change': total_change,
        'avg_change_per_step': avg_change_per_step,
        'percent_change': percent_change,
        
        # 趋势特征
        'trend_direction': get_trend_direction(slope, is_significant),
        'trend_strength': classify_trend_strength(correlation),
        'is_significant': int(is_significant),
        'monotonicity': monotonicity,
        'trend_stability': trend_stability,
        
        # 解释性文本
        'summary': generate_trend_summary(correlation, slope, is_significant, classify_trend_strength(correlation))
    }
    
    # 可视化
    # if show_plot:
    #     create_trend_plot(indices, values, reg, result, plot_title)
    # print(result)
    return result


class SFRTopicDistanceAnalyzer:
    """
    基于SFR-Embedding-Mistral的主题距离分析器
    使用sentence_transformers库和Salesforce/SFR-Embedding-Mistral模型提取文本embedding，计算余弦相似度作为主题距离
    """
    
    def __init__(self, model_name: str = "Salesforce/SFR-Embedding-Mistral"):
        """
        初始化分析器
        
        Parameters:
        -----------
        model_name : str
            SFR模型名称，默认使用Salesforce/SFR-Embedding-Mistral
        """
        self.model_name = model_name
        
        print(f"加载SFR-Embedding-Mistral模型: {model_name}")
        
        # 使用sentence_transformers加载模型
        self.model = SentenceTransformer(model_name, cache_folder='')
        if torch.cuda.is_available():
            self.model.half()
        print("模型加载完成!")
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        构造SFR模型需要的详细指令格式
        
        Parameters:
        -----------
        task_description : str
            任务描述
        query : str
            查询文本
            
        Returns:
        --------
        str
            格式化的指令
        """
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def calculate_topic_distances(self, topic_sentence: str, paragraphs: List[str]) -> Dict[str, List[float]]:
        """
        计算主题句与各段落之间的距离
        
        Parameters:
        -----------
        topic_sentence : str
            主题句
        paragraphs : List[str]
            段落列表
            
        Returns:
        --------
        Dict[str, List[float]]
            包含余弦相似度和余弦距离的字典
        """
        print("正在计算主题句与段落之间的相似度...")
        
        # 为主题句添加指令格式
        task_description = "Retrieve passages that are semantically similar to the given topic sentence"
        formatted_topic = self.get_detailed_instruct(task_description, topic_sentence)
        
        # 段落不需要添加指令格式
        queries = [formatted_topic]
        passages = paragraphs
        
        # 使用sentence_transformers计算embeddings
        embeddings = self.model.encode(queries + passages)
        
        # 计算余弦相似度，乘以100转换为百分比形式，然后除以100恢复到0-1范围
        scores = util.cos_sim(embeddings[:1], embeddings[1:])
        cosine_similarities = scores[0].tolist()  # 获取第一行（主题句与所有段落的相似度）
        
        # 计算余弦距离
        cosine_distances = [1 - sim for sim in cosine_similarities]
        
        return {
            'cosine_similarity': cosine_similarities,
            'cosine_distance': cosine_distances
        }
    
    def visualize_distances(self, topic_sentence: str, paragraphs: List[str], 
                          paragraph_labels: List[str] = None, 
                          figsize: Tuple[int, int] = (15, 10), outputname: str='') -> Dict[str, List[float]]:
        """
        可视化主题距离
        
        Parameters:
        -----------
        topic_sentence : str
            主题句
        paragraphs : List[str]
            段落列表
        paragraph_labels : List[str], optional
            段落标签
        figsize : Tuple[int, int]
            图形大小
        outputname : str
            输出文件名
            
        Returns:
        --------
        Dict[str, List[float]]
            距离计算结果
        """
        if paragraph_labels is None:
            paragraph_labels = [f"段落 {i+1}" for i in range(len(paragraphs))]
        
        # 计算距离
        results = self.calculate_topic_distances(topic_sentence, paragraphs)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'SFR Topic Similarity Analysis: "{topic_sentence[:60]}..."', 
                    fontsize=14, fontweight='bold')
        
        # 1. 余弦相似度条形图
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(results['cosine_similarity'])), 
                        results['cosine_similarity'], 
                        color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Cosine Similarity', fontweight='bold')
        ax1.set_xlabel('Paragraph')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_xticks(range(len(paragraph_labels)))
        ax1.set_xticklabels(paragraph_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 余弦距离条形图
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(results['cosine_distance'])), 
                        results['cosine_distance'], 
                        color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_title('Cosine Distance', fontweight='bold')
        ax2.set_xlabel('Paragraph')
        ax2.set_ylabel('Cosine Distance')
        ax2.set_xticks(range(len(paragraph_labels)))
        ax2.set_xticklabels(paragraph_labels, rotation=45, ha='right')
        ax2.set_ylim(0, 2)
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 相似度排序图
        ax3 = axes[1, 0]
        sorted_indices = np.argsort(results['cosine_similarity'])[::-1]  # 从高到低排序
        sorted_similarities = [results['cosine_similarity'][i] for i in sorted_indices]
        sorted_labels = [paragraph_labels[i] for i in sorted_indices]
        
        bars3 = ax3.barh(range(len(sorted_similarities)), sorted_similarities, 
                        color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_title('Similarity Ranking', fontweight='bold')
        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Paragraph')
        ax3.set_yticks(range(len(sorted_labels)))
        ax3.set_yticklabels(sorted_labels)
        ax3.set_xlim(0, 1)
        
        # 添加数值标签
        for i, bar in enumerate(bars3):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 4. 相似度热力图
        ax4 = axes[1, 1]
        
        # 创建相似度矩阵用于热力图显示
        similarity_matrix = np.array(results['cosine_similarity']).reshape(1, -1)
        
        im = ax4.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax4.set_title('Similarity Heatmap', fontweight='bold')
        ax4.set_xlabel('Paragraph')
        ax4.set_ylabel('Topic')
        ax4.set_xticks(range(len(paragraph_labels)))
        ax4.set_xticklabels(paragraph_labels, rotation=45, ha='right')
        ax4.set_yticks([0])
        ax4.set_yticklabels(['Topic'])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4, shrink=0.6)
        cbar.set_label('Cosine Similarity')
        
        # 在热力图上显示数值
        for i in range(len(paragraph_labels)):
            ax4.text(i, 0, f'{results["cosine_similarity"][i]:.3f}', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if outputname:
            plt.savefig(outputname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存为: {outputname}")
        
        plt.close()  # 关闭图形以释放内存
        
        return results
    
    def print_analysis_report(self, topic_sentence: str, paragraphs: List[str],
                            paragraph_labels: List[str] = None):
        """
        打印详细的分析报告
        
        Parameters:
        -----------
        topic_sentence : str
            主题句
        paragraphs : List[str]
            段落列表
        paragraph_labels : List[str], optional
            段落标签
        """
        if paragraph_labels is None:
            paragraph_labels = [f"段落 {i+1}" for i in range(len(paragraphs))]
        
        results = self.calculate_topic_distances(topic_sentence, paragraphs)
        
        # print("=" * 80)
        # print("基于SFR-Embedding-Mistral的主题距离分析报告")
        print("=" * 80)
        print(f"模型: {self.model_name}")
        print(f"主题句: {topic_sentence}")
        print(f"段落数量: {len(paragraphs)}")
        print()
        
        print("相似度分析结果:")
        print("-" * 50)
        
        # 按相似度排序
        similarity_data = list(zip(
            paragraph_labels, 
            results['cosine_similarity'], 
            results['cosine_distance']
        ))
        similarity_data.sort(key=lambda x: x[1], reverse=True)  # 按相似度从高到低排序
        
        print("排序结果 (按相似度从高到低):")
        print(f"{'排名':<4} {'段落':<15} {'余弦相似度':<12} {'余弦距离':<12}")
        print("-" * 50)
        
        for rank, (label, similarity, distance) in enumerate(similarity_data, 1):
            print(f"{rank:<4} {label:<15} {similarity:<12.4f} {distance:<12.4f}")
        
        print()
        print("=" * 80)

def analyze_topic_similarity(topic_sentence: str, paragraphs: List[str], 
                           paragraph_labels: List[str] = None, analyzer = None,
                           model_name: str = "Salesforce/SFR-Embedding-Mistral",
                           outputname: str = '') -> Dict[str, List[float]]:
    """
    简化的主题相似度分析函数
    
    Parameters:
    -----------
    topic_sentence : str
        主题句
    paragraphs : List[str]
        段落列表
    paragraph_labels : List[str], optional
        段落标签
    model_name : str
        SFR模型名称
    outputname : str
        输出文件名
        
    Returns:
    --------
    Dict[str, List[float]]
        相似度和距离结果
    """
    # 创建分析器
    # analyzer = SFRTopicDistanceAnalyzer(model_name=model_name)
    
    # 生成分析报告
    analyzer.print_analysis_report(topic_sentence, paragraphs, paragraph_labels)
    
    # 可视化结果
    results = analyzer.visualize_distances(topic_sentence, paragraphs, paragraph_labels, (15,10), outputname)
    print('returned: ', results)
    analyze_list_trend(results['cosine_similarity'])
    return results

def main(path, analyzer):
    """
    主函数，处理JSONL格式的输入文件
    
    Parameters:
    -----------
    path : str
        输入文件路径
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    topic_sentence = data[0]['input']
    test_paragraphs = data[0]['decision_making_process']
    paragraph_labels = []
    for i in range(len(test_paragraphs)):
        paragraph_labels.append(str(i) + '-' + test_paragraphs[i].split(']')[0])
    
    results = analyze_topic_similarity(
        topic_sentence=topic_sentence,
        paragraphs=test_paragraphs,
        paragraph_labels=paragraph_labels,
        analyzer = analyzer,
        model_name="Salesforce/SFR-Embedding-Mistral",
        outputname=path.replace('.jsonl', '_sfr_topic.png')
    )
    return results

def check_topic(path, analyzer, data=None):
    if not data:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        topic_sentence = data[0]['input']
        test_paragraphs = data[0]['decision_making_process']
    else:
        topic_sentence = data['input']
        test_paragraphs = data['decision_making_process']
    paragraph_labels = []
    for i in range(len(test_paragraphs)):
        paragraph_labels.append(str(i) + '-' + test_paragraphs[i].split(']')[0])
    topic_sentence = topic_sentence.replace('compose a brief argumentative essay on:' ,'').strip()
    distances = analyzer.calculate_topic_distances(topic_sentence, test_paragraphs)
    results = analyze_list_trend(distances['cosine_similarity'])
    results['cosine_similarity'] = distances['cosine_similarity']
    print(results)
    return results

# 演示程序
if __name__ == "__main__":
    # 处理多个文件的示例
    analyzer = SFRTopicDistanceAnalyzer(model_name='Salesforce/SFR-Embedding-Mistral')
    for id in range(0, 1):
        try:
            main(f'/results/tasksolving/brainstorming/results_en_{id}_critera.jsonl', analyzer)
        except FileNotFoundError:
            print(f"文件 results_en_{id}_critera.jsonl 不存在，跳过...")
        except Exception as e:
            print(f"处理文件 results_en_{id}_critera.jsonl 时出错: {e}")
    
    # 演示用例（基于您提供的例子）
    # print("\n" + "="*60)
    # print("运行演示用例...")
    # print("="*60)
    
    # # 使用您例子中的数据进行测试
    # model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    
    # def get_detailed_instruct(task_description: str, query: str) -> str:
    #     return f'Instruct: {task_description}\nQuery: {query}'
    
    # # 简单验证例子
    # task = 'Given a web search query, retrieve relevant passages that answer the query'
    # queries = [
    #     get_detailed_instruct(task, 'How to bake a chocolate cake'),
    #     get_detailed_instruct(task, 'Symptoms of the flu')
    # ]
    
    # passages = [
    #     "To bake a delicious chocolate cake, you'll need the following ingredients: all-purpose flour, sugar, cocoa powder, baking powder, baking soda, salt, eggs, milk, vegetable oil, and vanilla extract. Start by preheating your oven to 350°F (175°C). In a mixing bowl, combine the dry ingredients (flour, sugar, cocoa powder, baking powder, baking soda, and salt). In a separate bowl, whisk together the wet ingredients (eggs, milk, vegetable oil, and vanilla extract). Gradually add the wet mixture to the dry ingredients, stirring until well combined. Pour the batter into a greased cake pan and bake for 30-35 minutes. Let it cool before frosting with your favorite chocolate frosting. Enjoy your homemade chocolate cake!",
    #     "The flu, or influenza, is an illness caused by influenza viruses. Common symptoms of the flu include a high fever, chills, cough, sore throat, runny or stuffy nose, body aches, headache, fatigue, and sometimes nausea and vomiting. These symptoms can come on suddenly and are usually more severe than the common cold. It's important to get plenty of rest, stay hydrated, and consult a healthcare professional if you suspect you have the flu. In some cases, antiviral medications can help alleviate symptoms and reduce the duration of the illness."
    # ]
    
    # embeddings = model.encode(queries + passages)
    # scores = util.cos_sim(embeddings[:2], embeddings[2:]) * 100
    # print("验证例子结果:")
    # print(scores.tolist())
    # print("期望结果: [[86.71537780761719, 36.645721435546875], [35.00497055053711, 82.07388305664062]]")
    
    # # 运行完整分析
    # topic_sentence = "Climate change is one of the most pressing global challenges of our time"
    
    # test_paragraphs = [
    #     "Global warming has led to rising sea levels and more frequent extreme weather events. Scientists worldwide are studying the impacts of greenhouse gas emissions on our planet's climate system.",
        
    #     "The latest smartphone features include advanced camera technology, 5G connectivity, and artificial intelligence capabilities. Tech companies are constantly innovating to provide better user experiences.",
        
    #     "Renewable energy sources such as solar and wind power are becoming increasingly important in the fight against climate change. Many countries are investing heavily in clean energy infrastructure.",
        
    #     "Cooking traditional Italian pasta requires selecting the right type of noodles and preparing a flavorful sauce. The secret to perfect pasta lies in timing and using high-quality ingredients.",
        
    #     "Environmental protection efforts include reforestation, wildlife conservation, and reducing carbon emissions. International cooperation is essential for addressing global environmental challenges."
    # ]
    
    # paragraph_labels = [
    #     "Climate Science",
    #     "Smartphone Tech", 
    #     "Renewable Energy",
    #     "Italian Cooking",
    #     "Environmental Protection"
    # ]
    
    # print("\n开始基于SFR-Embedding-Mistral的主题距离分析...")
    
    # # 运行分析
    # results = analyze_topic_similarity(
    #     topic_sentence=topic_sentence,
    #     paragraphs=test_paragraphs,
    #     paragraph_labels=paragraph_labels,
    #     model_name="Salesforce/SFR-Embedding-Mistral",
    #     outputname="sfr_demo_analysis.png"
    # )
    
    # print("\n分析完成!")
    # print("注意：相似度越高表示与主题句越相关，距离越小表示越相似")
    # print("使用sentence_transformers库，无需手动处理tokenization、池化和归一化")
