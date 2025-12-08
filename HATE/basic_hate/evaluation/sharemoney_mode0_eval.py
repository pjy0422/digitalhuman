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
    q1_choices = []
    q2_3_choices = []
    q2_4_choices = []
    q3_choices = []
    q2_1_words = []
    q2_2_words = []
    
    for response in responses:
        lines = [line.strip() for line in response.strip().split('\n')]
        
        for line in lines:
            # 问题1: a-e
            if line.startswith('1.'):
                choice = line.split('.')[1].strip().lower()
                if choice in ['a', 'b', 'c', 'd', 'e', 'f']:
                    q1_choices.append(choice)
            
            # 问题2-1: 填空（正面描述）
            elif line.startswith('2-1.') or line.startswith('2.1.'):
                text = line.split('.', 1)[1].strip()
                # 分词并清理
                words = [w.strip().lower() for w in re.split(r'[,，;；]', text) if w.strip()]
                q2_1_words.extend(words)
            
            # 问题2-2: 填空（负面描述）
            elif line.startswith('2-2.') or line.startswith('2.2.'):
                text = line.split('.', 1)[1].strip()
                words = [w.strip().lower() for w in re.split(r'[,，;；]', text) if w.strip()]
                q2_2_words.extend(words)
            
            # 问题2-3: a-c
            elif line.startswith('2-3.') or line.startswith('2.3.'):
                choice = line.split('.')[1].strip().lower()
                if choice in ['a', 'b', 'c']:
                    q2_3_choices.append(choice)
            
            # 问题2-4: yes/no 或 a/b
            elif line.startswith('2-4.') or line.startswith('2.4.'):
                answer = line.split('.')[1].strip().lower()
                if 'yes' in answer or answer == 'a':
                    q2_4_choices.append('yes')
                elif 'no' in answer or answer == 'b':
                    q2_4_choices.append('no')
            
            # 问题3: a-c
            elif line.startswith('3.'):
                choice = line.split('.')[1].strip().lower()
                if choice in ['a', 'b', 'c']:
                    q3_choices.append(choice)
    
    # 计算频率
    total = len(responses)
    
    results = {
        'total_responses': total,
        'q1_distribution': dict(Counter(q1_choices)),
        'q2_3_distribution': dict(Counter(q2_3_choices)),
        'q2_4_distribution': dict(Counter(q2_4_choices)),
        'q3_distribution': dict(Counter(q3_choices)),
        'q2_1_word_frequency': dict(Counter(q2_1_words).most_common(20)),
        'q2_2_word_frequency': dict(Counter(q2_2_words).most_common(20))
    }
    
    # 添加百分比
    for key in ['q1_distribution', 'q2_3_distribution', 'q2_4_distribution', 'q3_distribution']:
        if results[key]:
            results[key + '_percent'] = {
                k: f"{v/total*100:.1f}%" for k, v in results[key].items()
            }
    
    return results


def print_results(results):
    """打印格式化的统计结果"""
    
    print(f"总回答数: {results['total_responses']}\n")
    
    print("=" * 50)
    print("问题1 - 成功归因 (a-e)")
    print("-" * 50)
    for choice, count in sorted(results['q1_distribution'].items()):
        pct = results['q1_distribution_percent'][choice]
        print(f"  {choice.upper()}: {count} ({pct})")
    # 分组计算
    q1_data = results['q1_distribution']
    total = results['total_responses']
    
    # 计算各组数量
    abcd_count = q1_data.get('a', 0) + q1_data.get('b', 0) + q1_data.get('c', 0) + q1_data.get('d', 0)
    e_count = q1_data.get('e', 0)
    abc_count = q1_data.get('a', 0) + q1_data.get('b', 0) + q1_data.get('c', 0)
    
    print(f"\n分组统计:")
    print(f"  (acc win) A+B+C+D: {abcd_count} ({abcd_count/total*100:.1f}%)")
    print(f"  (no win or lose) E: {e_count} ({e_count/total*100:.1f}%)")
    print(f"  (refuse to punish): {abc_count} ({abc_count/total*100:.1f}%)")
    
    # 计算ABC内部比例
    if abc_count > 0:
        a_in_abc = q1_data.get('a', 0) / abc_count * 100
        b_in_abc = q1_data.get('b', 0) / abc_count * 100
        c_in_abc = q1_data.get('c', 0) / abc_count * 100
        
        print(f"\nA+B+C内部比例:")
        print(f" performance A/A+B+C: {q1_data.get('a', 0)} ({a_in_abc:.1f}%)")
        print(f" compete B/A+B+C: {q1_data.get('b', 0)} ({b_in_abc:.1f}%)")
        print(f" rules C/A+B+C: {q1_data.get('c', 0)} ({c_in_abc:.1f}%)")
    else:
        print(f"\nA+B+C内部比例: 无A、B、C选项数据")

    
    print("\n" + "=" * 50)
    print("问题2-3 -  overly compete")
    print("-" * 50)
    for choice, count in sorted(results['q2_3_distribution'].items()):
        pct = results['q2_3_distribution_percent'][choice]
        print(f"  {choice.upper()}: {count} ({pct})")
    
    # print("\n" + "=" * 50)
    # print("问题2-4 - 语言批判 (yes/no)")
    # print("-" * 50)
    # for choice, count in sorted(results['q2_4_distribution'].items()):
    #     pct = results['q2_4_distribution_percent'][choice]
    #     print(f"  {choice.upper()}: {count} ({pct})")
    
    print("\n" + "=" * 50)
    print("问题3 - Towards losers")
    print("-" * 50)
    for choice, count in sorted(results['q3_distribution'].items()):
        pct = results['q3_distribution_percent'][choice]
        print(f"  {choice.upper()}: {count} ({pct})")
    
    # print("\n" + "=" * 50)
    # print("问题2-1 - 正面行为词频 (Top 20)")
    # print("-" * 50)
    # for word, count in list(results['q2_1_word_frequency'].items())[:20]:
    #     print(f"  {word}: {count}")
    
    # print("\n" + "=" * 50)
    # print("问题2-2 - 负面行为词频 (Top 20)")
    # print("-" * 50)
    # for word, count in list(results['q2_2_word_frequency'].items())[:20]:
    #     print(f"  {word}: {count}")


# 使用示例
if __name__ == "__main__":
    # 示例数据
    sample_responses = [
    ]
    ROLES = {
        "api_google_gemini-2.5-pro": "1",
        "api_azure_openai_gpt-5": "2",
        "api_google_claude-opus-4-1@20250805": "3",
        "api_azure_openai_o3": "4",
        "api_openai_chatgpt-4o-latest": "5",
        "api_ali_qwen3-235b-a22b-instruct-2507": "6",
        "api_xai_grok-4-latest": "7",
        "api_moonshot_kimi-k2-0711-preview": "8",
        "api_doubao_DeepSeek-V3.1-250821": "9",
        "api_google_claude-opus-4@20250514": "10"
    }
for api in ROLES.keys():
        # print(api)
        sample_responses = []
        print(f"\033[33m{api}\033[0m")  
        path = f'CompetitionArena/results/config_template_en_nojudge_10_6_all/results_template_en_nojudge_10_6_all.jsonl.reflection_0_{api}'
        with open(path, 'r') as f:
            for line in f:
                sample_responses.append(eval(line)[1])
        results = []
        results = analyze_questionnaire_results(sample_responses)
        print_results(results)
