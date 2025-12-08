from collections import Counter
import re

from collections import Counter

def analyze_questionnaire_results(responses):
    """
    分析问卷结果并生成统计报告
    
    参数:
        responses: list of str, 每个元素是一份问卷的回答
    
    返回:
        dict: 包含所有统计结果
    """
    
    # 初始化统计容器
    q1_choices = []
    q2_choices = []
    q3_choices = []
    q4_choices = []
    
    for response in responses:
        lines = [line.strip() for line in response.strip().split('\n')]
        
        for line in lines:
            # 问题1
            if line.startswith('1.'):
                choice = line.split('.')[1].strip().lower()
                if choice:  # 接受任何非空选项
                    q1_choices.append(choice)
            
            # 问题2
            elif line.startswith('2.'):
                choice = line.split('.')[1].strip().lower()
                if choice:
                    q2_choices.append(choice)
            
            # 问题3
            elif line.startswith('3.'):
                choice = line.split('.')[1].strip().lower()
                if choice:
                    q3_choices.append(choice)
            
            # 问题4
            elif line.startswith('4.'):
                choice = line.split('.')[1].strip().lower()
                if choice:
                    q4_choices.append(choice)
    
    # 计算频率
    total = len(responses)
    
    results = {
        'total_responses': total,
        'q1_distribution': dict(Counter(q1_choices)),
        'q2_distribution': dict(Counter(q2_choices)),
        'q3_distribution': dict(Counter(q3_choices)),
        'q4_distribution': dict(Counter(q4_choices))
    }
    
    # 添加百分比
    for key in ['q1_distribution', 'q2_distribution', 'q3_distribution', 'q4_distribution']:
        if results[key]:
            results[key + '_percent'] = {
                k: f"{v/total*100:.1f}%" for k, v in results[key].items()
            }
    
    return results


def print_results(results):
    """打印格式化的统计结果"""
    
    print(f"总回答数: {results['total_responses']}\n")
    
    questions = [
        ('q1_distribution', '问题1'),
        ('q2_distribution', '问题2'), 
        ('q3_distribution', '问题3'),
        ('q4_distribution', '问题4')
    ]
    
    for dist_key, title in questions:
        print("=" * 50)
        print(f"{title}")
        print("-" * 50)
        
        if results[dist_key]:
            for choice, count in sorted(results[dist_key].items()):
                pct = results[dist_key + '_percent'][choice]
                print(f"  {choice.upper()}: {count} ({pct})")
        else:
            print("  无有效数据")
        print()


# # 使用示例
# if __name__ == "__main__":
#     # 示例数据
#     sample_responses = [
#         """1. a
# 2. b
# 3. c
# 4. a""",
#         """1. b
# 2. a
# 3. c
# 4. b""",
#         """1. a
# 2. b
# 3. a
# 4. a"""
#     ]
    
#     results = analyze_questionnaire_results(sample_responses)
#     print_results(results)

# 使用示例
if __name__ == "__main__":
    # 示例数据
    sample_responses = [
    ]    
    path = ''
    with open(path, 'r') as f:
        for line in f:
            sample_responses.append(eval(line)[1])

    results = analyze_questionnaire_results(sample_responses)
    print_results(results)
