import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple
from scipy import stats

# 设置中文字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CosineSimilarityAnalyzer:
    def __init__(self, model_count: int):
        """
        初始化分析器
        
        Args:
            model_count (int): 模型数量 m
        """
        self.model_count = model_count
        self.data = []
        self.matrices = []
    
    def load_jsonl_data(self, jsonl_string: str) -> None:
        """
        加载JSONL格式的数据
        
        Args:
            jsonl_string (str): 已解析的数据列表
        """
        lines = jsonl_string  # 已经是解析后的数据列表
        self.data = []
        self.matrices = []
        
        for i, line in enumerate(lines):
            # 从数据结构中提取cosine_similarity
            cosine_values = line['topic']['cosine_similarity']
            
            # 计算轮次数
            n = len(cosine_values) // self.model_count
            
            # 重构为 [n, m] 矩阵：n轮对话，每轮m个模型
            # 然后转置为 [m, n] 便于后续分析：m个模型，n轮对话
            matrix = np.array(cosine_values).reshape(n, self.model_count).T
            print(f"样本 {i+1}: {self.model_count} 个模型, {n} 轮对话")
            if n > 1:
                self.data.append(line)
                self.matrices.append(matrix)

    def load_jsonl_file(self, filename: str) -> None:
        """
        从文件加载JSONL数据
        
        Args:
            filename (str): 文件路径
        """
        with open(filename, 'r', encoding='utf-8') as f:
            content = []
            for line in f:
                content.append(json.loads(line))
        self.load_jsonl_data(content)
    
    def analyze_round_trends(self) -> Dict:
        """
        分析轮次趋势：
        1. m个模型的平均余弦相似度随轮次的变化
        2. 每个模型各自的余弦相似度随轮次的变化
        
        Returns:
            dict: 包含趋势分析结果的字典
        """
        if len(self.matrices) == 0:
            print("没有数据可供分析")
            return {}
        
        results = {
            'overall_trend': {  # 所有模型的平均趋势
                'round_averages': [],  # 每轮的所有模型平均值
                'correlation': 0,
                'p_value': 0,
                'slope': 0,
                'r_squared': 0,
                'interpretation': ''
            },
            'model_trends': {},  # 每个模型的趋势
            'sample_analysis': []  # 每个样本的分析
        }
        
        # 用于汇总所有样本的数据
        all_round_averages = []  # 所有样本的每轮平均值
        all_round_ids = []       # 对应的轮次ID
        
        model_all_values = {i: {'values': [], 'round_ids': []} for i in range(self.model_count)}
        
        # 逐个分析每个样本
        for sample_idx, matrix in enumerate(self.matrices):
            n_rounds = matrix.shape[1]
            round_ids = list(range(1, n_rounds + 1))
            
            # 计算每轮的所有模型平均值
            round_averages = np.mean(matrix, axis=0)  # 每轮m个模型的平均
            
            # 分析该样本的整体趋势
            sample_corr, sample_p = stats.pearsonr(round_ids, round_averages)
            sample_slope, sample_intercept, sample_r, _, _ = stats.linregress(round_ids, round_averages)
            
            sample_analysis = {
                'sample_id': sample_idx + 1,
                'round_averages': round_averages.tolist(),
                'correlation': sample_corr,
                'p_value': sample_p,
                'slope': sample_slope,
                'r_squared': sample_r ** 2,
                'is_significant': sample_p < 0.05,
                'interpretation': self._interpret_correlation(sample_corr, sample_p)
            }
            results['sample_analysis'].append(sample_analysis)
            
            # 汇总到总体数据
            all_round_averages.extend(round_averages)
            all_round_ids.extend(round_ids)
            
            # 按模型汇总数据
            for model_id in range(self.model_count):
                model_values = matrix[model_id]  # 该模型在所有轮次的值
                model_all_values[model_id]['values'].extend(model_values)
                model_all_values[model_id]['round_ids'].extend(round_ids)
        
        # 分析整体趋势（所有样本的所有轮次平均值）
        if all_round_ids and all_round_averages:
            overall_corr, overall_p = stats.pearsonr(all_round_ids, all_round_averages)
            overall_slope, overall_intercept, overall_r, _, _ = stats.linregress(all_round_ids, all_round_averages)
            
            results['overall_trend'] = {
                'round_averages': all_round_averages,
                'round_ids': all_round_ids,
                'correlation': overall_corr,
                'p_value': overall_p,
                'slope': overall_slope,
                'intercept': overall_intercept,
                'r_squared': overall_r ** 2,
                'is_significant': overall_p < 0.05,
                'interpretation': self._interpret_correlation(overall_corr, overall_p)
            }
        
        # 分析每个模型的趋势
        for model_id in range(self.model_count):
            model_data = model_all_values[model_id]
            model_values = model_data['values']
            model_round_ids = model_data['round_ids']
            
            if model_values and model_round_ids:
                model_corr, model_p = stats.pearsonr(model_round_ids, model_values)
                model_slope, model_intercept, model_r, _, _ = stats.linregress(model_round_ids, model_values)
                
                results['model_trends'][f'模型{model_id + 1}'] = {
                    'values': model_values,
                    'round_ids': model_round_ids,
                    'correlation': model_corr,
                    'p_value': model_p,
                    'slope': model_slope,
                    'intercept': model_intercept,
                    'r_squared': model_r ** 2,
                    'is_significant': model_p < 0.05,
                    'interpretation': self._interpret_correlation(model_corr, model_p)
                }
        
        return results
    
    def _interpret_correlation(self, corr: float, p_value: float) -> str:
        """
        解释相关性结果
        """
        if p_value >= 0.05:
            return "无显著相关性"
        
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "强"
        elif abs_corr >= 0.5:
            strength = "中等"
        elif abs_corr >= 0.3:
            strength = "弱"
        else:
            strength = "很弱"
        
        direction = "负" if corr < 0 else "正"
        return f"{strength}{direction}相关"
    
    def plot_trend_analysis(self, figsize: Tuple[int, int] = (18, 12)) -> None:
        """
        绘制趋势分析图
        """
        trend_results = self.analyze_round_trends()
        if not trend_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 整体平均趋势（所有样本）
        ax1 = axes[0, 0]
        overall_trend = trend_results['overall_trend']
        
        if overall_trend['round_ids']:
            # 绘制散点图
            ax1.scatter(overall_trend['round_ids'], overall_trend['round_averages'], 
                       alpha=0.6, color='blue', s=20)
            
            # 绘制趋势线
            x_trend = np.array(overall_trend['round_ids'])
            y_trend = overall_trend['slope'] * x_trend + overall_trend['intercept']
            ax1.plot(x_trend, y_trend, 'r-', linewidth=2, 
                    label=f"趋势线 (斜率={overall_trend['slope']:.5f})")
            
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('所有模型平均余弦相似度')
            ax1.set_title(f"整体平均趋势\n相关系数: {overall_trend['correlation']:.4f}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 各样本的平均趋势对比
        ax2 = axes[0, 1]
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.matrices)))
        
        for sample_info in trend_results['sample_analysis']:
            sample_id = sample_info['sample_id']
            round_averages = sample_info['round_averages']
            n_rounds = len(round_averages)
            round_ids = list(range(1, n_rounds + 1))
            
            ax2.plot(round_ids, round_averages, 'o-', alpha=0.7,
                    label=f"样本{sample_id} (r={sample_info['correlation']:.3f})",
                    color=colors[(sample_id-1) % len(colors)])
        
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('模型平均余弦相似度')
        ax2.set_title('各样本的平均趋势对比')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 各模型的整体趋势对比
        ax3 = axes[0, 2]
        model_trends = trend_results['model_trends']
        model_colors = plt.cm.tab10(np.linspace(0, 1, self.model_count))
        
        for i, (model_name, model_data) in enumerate(model_trends.items()):
            # 为了可视化，我们对数据进行分组平均
            round_ids = np.array(model_data['round_ids'])
            values = np.array(model_data['values'])
            
            # 按轮次分组计算平均值
            unique_rounds = np.unique(round_ids)
            avg_values = [np.mean(values[round_ids == r]) for r in unique_rounds]
            
            ax3.plot(unique_rounds, avg_values, 'o-', alpha=0.8,
                    label=f"{model_name} (r={model_data['correlation']:.3f})",
                    color=model_colors[i])
        
        ax3.set_xlabel('轮次')
        ax3.set_ylabel('余弦相似度')
        ax3.set_title('各模型趋势对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 相关系数比较
        ax4 = axes[1, 0]
        model_names = list(model_trends.keys())
        model_correlations = [model_trends[name]['correlation'] for name in model_names]
        
        bars = ax4.bar(model_names, model_correlations,
                      color=model_colors[:len(model_names)], alpha=0.7)
        ax4.set_ylabel('相关系数')
        ax4.set_title('各模型轮次相关系数比较')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 添加数值标签
        for bar, corr in zip(bars, model_correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 衰减率比较
        ax5 = axes[1, 1]
        model_slopes = [model_trends[name]['slope'] for name in model_names]
        
        bars = ax5.bar(model_names, model_slopes,
                      color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(model_names))), alpha=0.7)
        ax5.set_ylabel('衰减率 (斜率)')
        ax5.set_title('各模型衰减率比较')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 添加数值标签
        for bar, slope in zip(bars, model_slopes):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{slope:.5f}', ha='center', va='bottom', fontsize=8)
        
        # 6. 模型排名（按衰减率）
        ax6 = axes[1, 2]
        
        # 按衰减率排序
        model_slope_pairs = list(zip(model_names, model_slopes))
        model_slope_pairs.sort(key=lambda x: x[1])  # 按衰减率排序
        sorted_names, sorted_slopes = zip(*model_slope_pairs)
        
        bars = ax6.barh(range(len(sorted_names)), sorted_slopes,
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names))))
        ax6.set_yticks(range(len(sorted_names)))
        ax6.set_yticklabels(sorted_names)
        ax6.set_xlabel('衰减率 (每轮变化)')
        ax6.set_title('模型衰减速度排名\n(负值越大衰减越快)')
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def print_trend_analysis(self) -> None:
        """
        打印详细的趋势分析报告
        """
        results = self.analyze_round_trends()
        if not results:
            return
        
        print("\n" + "="*80)
        print("余弦相似度轮次趋势分析报告")
        print("="*80)
        
        # 整体趋势分析
        overall = results['overall_trend']
        print(f"\n1. 整体趋势分析（所有模型的平均值）:")
        print(f"   相关系数: {overall['correlation']:.4f}")
        print(f"   p值: {overall['p_value']:.6f}")
        print(f"   统计显著性: {'是' if overall['is_significant'] else '否'} (α=0.05)")
        print(f"   R²: {overall['r_squared']:.4f}")
        print(f"   斜率: {overall['slope']:.6f} (每轮变化量)")
        print(f"   解释: {overall['interpretation']}")
        
        if overall['correlation'] < -0.1 and overall['is_significant']:
            print(f"   ✓ 结论: 随着轮次增加，所有模型的平均余弦相似度呈显著下降趋势")
        elif overall['correlation'] > 0.1 and overall['is_significant']:
            print(f"   ✓ 结论: 随着轮次增加，所有模型的平均余弦相似度呈显著上升趋势")
        else:
            print(f"   ✗ 结论: 轮次与所有模型的平均余弦相似度无显著线性关系")
        
        # 各模型趋势分析
        print(f"\n2. 各模型趋势分析:")
        model_trends = results['model_trends']
        
        # 按相关系数排序
        sorted_models = sorted(model_trends.items(), key=lambda x: x[1]['correlation'])
        
        for model_name, model_data in sorted_models:
            print(f"\n   {model_name}:")
            print(f"     相关系数: {model_data['correlation']:.4f}")
            print(f"     p值: {model_data['p_value']:.6f}")
            print(f"     统计显著性: {'是' if model_data['is_significant'] else '否'}")
            print(f"     R²: {model_data['r_squared']:.4f}")
            print(f"     斜率: {model_data['slope']:.6f} (每轮变化量)")
            print(f"     解释: {model_data['interpretation']}")
        
        # 模型衰减速度排名
        print(f"\n3. 模型衰减速度排名:")
        slope_ranking = [(name, data['slope']) for name, data in model_trends.items()]
        slope_ranking.sort(key=lambda x: x[1])  # 按斜率排序，负数越小排名越靠前
        
        print("   (负斜率表示衰减，负值越大衰减越快)")
        for rank, (model_name, slope) in enumerate(slope_ranking, 1):
            trend_desc = "下降" if slope < 0 else "上升" if slope > 0 else "平稳"
            print(f"     第{rank}名: {model_name} (斜率: {slope:.6f}, {trend_desc}趋势)")
        
        # 各样本分析汇总
        print(f"\n4. 各样本分析汇总:")
        sample_analyses = results['sample_analysis']
        significant_samples = [s for s in sample_analyses if s['is_significant']]
        declining_samples = [s for s in sample_analyses if s['correlation'] < -0.1 and s['is_significant']]
        
        print(f"   总样本数: {len(sample_analyses)}")
        print(f"   显著趋势样本数: {len(significant_samples)}，占比 {len(significant_samples)/len(sample_analyses)*100:.2f}%")
        print(f"   显著下降趋势样本数: {len(declining_samples)}，占比 {len(declining_samples)/len(sample_analyses)*100:.2f}%")
        
        # for sample_info in sample_analyses:
        #     sig_mark = "*" if sample_info['is_significant'] else " "
        #     print(f"   样本{sample_info['sample_id']}{sig_mark}: "
        #           f"r={sample_info['correlation']:.3f}, "
        #           f"p={sample_info['p_value']:.4f}, "
        #           f"斜率={sample_info['slope']:.5f}")
        
        print(f"\n{'*' * 80}")
        print("注: * 表示统计显著 (p < 0.05)")
        print(f"{'*' * 80}")


def analyze_cosine_similarity_from_file(jsonl_file_path: str, model_count: int):
    """
    从JSONL文件分析余弦相似度趋势的主函数
    
    Args:
        jsonl_file_path (str): JSONL文件路径
        model_count (int): 模型数量 m
    """
    print(f"正在分析文件: {jsonl_file_path}")
    print(f"模型数量: {model_count}")
    
    # 创建分析器
    analyzer = CosineSimilarityAnalyzer(model_count)
    
    try:
        # 加载数据
        analyzer.load_jsonl_file(jsonl_file_path)
        
        if len(analyzer.data) == 0:
            print("错误：没有成功加载任何数据")
            return analyzer
        
        print(f"成功加载 {len(analyzer.data)} 个样本")
        
        # 执行趋势分析
        print(f"\n{'='*80}")
        print("执行轮次趋势分析")
        print(f"{'='*80}")
        
        # 1. 打印详细的趋势分析报告
        analyzer.print_trend_analysis()
        
        # # 2. 绘制趋势分析图
        # print("\n绘制趋势分析图...")
        # analyzer.plot_trend_analysis()
        
        # print(f"\n{'='*80}")
        # print("分析完成！")
        # print(f"{'='*80}")
        
        return analyzer
        
    except FileNotFoundError:
        print(f"错误：文件 {jsonl_file_path} 不存在")
        return None
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 使用示例
if __name__ == "__main__":
# 2. 设置模型数量
    # model_count = 10
    model_count = 4
    # 3. 执行分析
    analyzer = analyze_cosine_similarity_from_file(jsonl_file_path, model_count)
    
    # 如果需要后续操作，可以使用返回的analyzer对象
    if analyzer is not None:
        print("分析器已准备就绪，可以进行进一步分析")
        
        # 例如：获取分析结果
        # results = analyzer.analyze_round_trends()
        # 或者重新绘图
        # analyzer.plot_trend_analysis()
