from collections import Counter
import re

def analyze_questionnaire_results(responses):
    """
    分析问卷结果并生成统计报告
    
    参数:
        responses: list of str, 每个元素是一份问卷的回答
    
    返回:
        dict: 包含所有统计结果
    """
    
    # 初始化统计容器
    q1_1_choices = []
    q1_2_agents = []
    q3_choices = []
    q2_1_words = []
    q2_2_words = []
    
    for response in responses:
        lines = [line.strip() for line in response.strip().split('\n')]
        
        for line in lines:
            # 问题1-1: a-c
            if line.startswith('1-1.'):
                choice = line.split('.')[1].strip().lower()
                if choice in ['a', 'b', 'c']:
                    q1_1_choices.append(choice)
            
            # 问题1-2: Agent提名
            elif line.startswith('1-2.'):
                agent = line.split('.')[1].strip()
                if agent:  # 只要不为空就记录
                    q1_2_agents.append(agent)
            
            # 问题2-1: 填空（正面描述）
            elif line.startswith('2-1.'):
                text = line.split('.', 1)[1].strip()
                # 分词并清理
                words = [w.strip().lower() for w in re.split(r'[,，;；]', text) if w.strip()]
                q2_1_words.extend(words)
            
            # 问题2-2: 填空（负面描述）
            elif line.startswith('2-2.'):
                text = line.split('.', 1)[1].strip()
                words = [w.strip().lower() for w in re.split(r'[,，;；]', text) if w.strip()]
                q2_2_words.extend(words)
            
            # 问题3: a-b
            elif line.startswith('3.'):
                choice = line.split('.')[1].strip().lower()
                if choice in ['a', 'b']:
                    q3_choices.append(choice)
    
    # 计算频率
    total = len(responses)
    
    results = {
        'total_responses': total,
        'q1_1_distribution': dict(Counter(q1_1_choices)),
        'q1_2_agent_frequency': dict(Counter(q1_2_agents)),
        'q3_distribution': dict(Counter(q3_choices)),
        'q2_1_word_frequency': dict(Counter(q2_1_words).most_common(20)),
        'q2_2_word_frequency': dict(Counter(q2_2_words).most_common(20))
    }
    
    # 添加百分比
    for key in ['q1_1_distribution', 'q3_distribution']:
        if results[key]:
            results[key + '_percent'] = {
                k: f"{v/total*100:.1f}%" for k, v in results[key].items()
            }
    
    # Agent提名的百分比（可能超过100%因为有些人可能没提名）
    if results['q1_2_agent_frequency']:
        results['q1_2_agent_frequency_percent'] = {
            k: f"{v/total*100:.1f}%" for k, v in results['q1_2_agent_frequency'].items()
        }
    
    return results


def print_results(results):
    """打印格式化的统计结果"""
    
    print(f"总回答数: {results['total_responses']}\n")
    
    print("=" * 50)
    print("问题1-1 - 成功归因 (a-c)")
    print("-" * 50)
    for choice, count in sorted(results['q1_1_distribution'].items()):
        pct = results['q1_1_distribution_percent'][choice]
        print(f"  {choice.upper()}: {count} ({pct})")
    
    print("\n" + "=" * 50)
    print("问题1-2 - Agent提名频率")
    print("-" * 50)
    # 按频率排序显示
    sorted_agents = sorted(results['q1_2_agent_frequency'].items(), key=lambda x: x[1], reverse=True)
    for agent, count in sorted_agents:
        pct = results['q1_2_agent_frequency_percent'][agent]
        print(f"  {agent}: {count} ({pct})")
    
    print("\n" + "=" * 50)
    print("问题3 - 贡献认可 (a-b)")
    print("-" * 50)
    for choice, count in sorted(results['q3_distribution'].items()):
        pct = results['q3_distribution_percent'][choice]
        print(f"  {choice.upper()}: {count} ({pct})")
    
    print("\n" + "=" * 50)
    print("问题2-1 - 正面行为词频 (Top 20)")
    print("-" * 50)
    for word, count in list(results['q2_1_word_frequency'].items())[:20]:
        print(f"  {word}: {count}")
    
    print("\n" + "=" * 50)
    print("问题2-2 - 负面行为词频 (Top 20)")
    print("-" * 50)
    for word, count in list(results['q2_2_word_frequency'].items())[:20]:
        print(f"  {word}: {count}")


# 使用示例
if __name__ == "__main__":
    # 示例数据
#     sample_responses = [
#         """1-1. a
# 1-2. Agent5
# 2-1. factual,respectful,avoided ad-hominem attacks,highly strategic
# 2-2. purposely competitive,opportunistic,openly zero-sum, thinking of "winning" above collaboration
# 3. b""",
#         """1-1. b
# 1-2. Agent3
# 2-1. strategic, professional, evidence-based
# 2-2. aggressive, dismissive, combative language
# 3. a""",
#         """1-1. a
# 1-2. Agent5
# 2-1. systematic, thorough, analytical
# 2-2. ruthless, zero-sum mindset
# 3. b"""
#     ]
    # 示例数据
    sample_responses = [
    ]
    path = ''
    with open(path, 'r') as f:
        for line in f:
            sample_responses.append(eval(line)[1])
    results = analyze_questionnaire_results(sample_responses)
    print_results(results)
