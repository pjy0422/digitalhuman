import json 
import argparse
from tqdm import tqdm
from utils import setup_logger
from agent import Agent
import random
import os
from utils import get_environment_prompt, get_nsp_prompt, get_character_prompt
from utils import get_response_json, extract_json
from utils import remove_inner_thoughts, calculate_bleu_rouge
from self_models import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
random.seed(42)

logger = None


# Set up command line argument parser
parser = argparse.ArgumentParser(
    description='Evaluate role-playing language models via given-circumstance acting (GCA)'
)

# 思考方式
parser.add_argument(
    '--thinking_pattern',
    type=str,
    default='none-first',
)

# Input/output paths
parser.add_argument(
    '--test_file',
    type=str,
    default='data/test/test_set.json',
    help='Path to the test dataset'
)
parser.add_argument(
    '--book_data',
    type=str,
    default='data/final',
    help='Path to the folder containing complete curated data of each book, used when retrieval augmentation is enabled.'
)

# Model configuration
parser.add_argument(
    '--actor_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for role-playing'
)
parser.add_argument(
    '--judge_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for LLM judging'
)
parser.add_argument(
    '--env_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for environment response'
)
parser.add_argument(
    '--nsp_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for next-speaker prediction, default to gpt-4o-mini, but recommend Coser-70B or self-deployed models for better cost-efficiency.'
)

# Runtime settings
parser.add_argument(
    '--continue_from',
    type=int,
    default=0,
    help='Start GCA from the i-th round. The previous rounds will use the ground truth conversations.'
)
parser.add_argument(
    '--wo_thought',
    default=False,
    action='store_true',
    help='Disable inner thoughts in generation'
)
parser.add_argument(
    '--retrieval',
    type=str,
    default=None,
    choices=[None, 'raw_text', 'expr1', 'expr3', 'conv1', 'expr3_conv1', 'expr10_conv1'],
    help='Target for retrieval, we abandoned using character utterance retrieval in this work, so ignore it.'
)
parser.add_argument(
    '--regenerate',
    action='store_true',
    help='Regenerate the simulation results'
)
parser.add_argument(
    '--reevaluate',
    action='store_true',
    help='Reevaluate the simulation results'
)
parser.add_argument(
    '--nth_exp',
    type=int,
    default=0,
    help='Experiment ID. Results will be reused for same ID. Set to -1 to run 3 experiments.'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=10,
    help='Number of parallel workers (default: 10)'
)

# Parse arguments
args = parser.parse_args()
ENVIRONMENT = 'Environment'
NSP = "NSP"
special_characters = [NSP, ENVIRONMENT]

def get_setting():
    settings = {'thinking_pattetern':args.thinking_pattern,'actor_model':args.actor_model}
    return settings

def extract_characters_from_simulation(simulation):
    """从simulation中提取所有参与对话的角色（不包括NSP和Environment）"""
    characters = set()
    for msg in simulation:
        if msg['role'] not in ['NSP', 'Environment']:
            characters.add(msg['role'])
    return characters

def count_character_speech(simulation):
    """统计每个角色的发言次数"""
    speech_counts = defaultdict(int)
    for msg in simulation:
        if msg['role'] not in ['NSP', 'Environment']:
            speech_counts[msg['role']] += 1
    return speech_counts

def calculate_character_scores_for_case(case_data, char_to_level):
    """
    计算单个case中每个角色的最终得分
    使用参考代码中的评分方法
    """
    # 权重设置
    w_avg = 0.5      # 平均扣分权重
    w_severe = 0.1   # 最严重扣分权重
    w_length = 0.15  # 对话长度权重
    
    # 提取simulation中的角色
    simulation_chars = extract_characters_from_simulation(case_data.get('simulation', []))
    
    # 统计每个角色的发言次数
    speech_counts = count_character_speech(case_data.get('simulation', []))
    
    # 初始化角色扣分记录
    char_deduction_records = {}
    for char in simulation_chars:
        if char in char_to_level:
            char_deduction_records[char] = []
    
    # 从evaluation的critique中提取扣分信息
    if 'critique' in case_data and 'Character Fidelity' in case_data['critique']:
        flaws = case_data['critique']['Character Fidelity'].get('flaws', [])
        
        for flaw in flaws:
            # 尝试从flaw中获取character字段
            char_name = flaw.get('character', '')
            
            # 如果没有character字段，尝试从description中识别
            if not char_name:
                flaw_desc = flaw.get('character', '')
                for char in char_to_level.keys():
                    if char in flaw_desc:
                        char_name = char
                        break
            
            severity = flaw.get('severity', 0)
            
            if char_name in char_deduction_records:
                char_deduction_records[char_name].append(severity)
    
    # 计算每个角色的最终得分
    char_scores = {}
    for char in simulation_chars:
        if char in char_to_level:
            speech_count = speech_counts.get(char, 0)
            deductions = char_deduction_records.get(char, [])
            
            if speech_count > 0:
                # 有发言的角色
                total_deduction = sum(deductions) if deductions else 0
                max_deduction = max(deductions) if deductions else 0
                
                P_avg = w_avg * total_deduction
                P_severe = w_severe * (max_deduction)
                P_length = w_length * speech_count
                
                final_score = 5 - P_avg - P_severe + P_length
                final_score = max(0, min(5, final_score))  # 限制在0-5分之间
            else:
                # 没有发言的角色
                final_score = 5 if not deductions else 0
                P_avg = 0
                P_severe = 0
                P_length = 0
                total_deduction = sum(deductions) if deductions else 0
                max_deduction = max(deductions) if deductions else 0
            
            char_scores[char] = {
                'final_score': final_score,
                'total_deduction': total_deduction,
                'max_deduction': max_deduction,
                'speech_count': speech_count,
                'level': char_to_level[char],
                'P_avg': P_avg,
                'P_severe': P_severe,
                'P_length': P_length
            }
    
    return char_scores

def process_single_circumstance(circumstance, actor_model, env_model, nsp_model, retrieval, thinking_pattern):
    """Process a single circumstance for simulation"""
    # collect scenario metadata and context
    book_title = circumstance['book']
    plot = circumstance['plot']
    i_p = plot['i_p'] 
    conversation = circumstance
    i_c = conversation['i_c']
    character_profiles = circumstance['character_profiles']
    topic = circumstance['topic']

    logger.info(f'==========Book {book_title}==========')

    # Load additional book data if retrieval is enabled
    if retrieval:
        book_database = json.load(open(f'{args.book_data}/{book_title}.json', 'r'))

    # Identify the character lists
    plot_characters = [ c['name'] for c in plot['key_characters']] 
    speaking_characters_w_env = conversation['speaking_characters_w_env']
    if ENVIRONMENT not in speaking_characters_w_env:
        speaking_characters_w_env.append(ENVIRONMENT)
    major_characters = conversation['major_characters']

    character_agents = {}
    involved_character_profiles = {}

    # Build enhanced character profiles combining scenario and plot information
    for character in speaking_characters_w_env:    
        if character == ENVIRONMENT:
            continue
        
        character_profile = character_profiles.get(character, '')
        if character in plot_characters:
            character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
            if 'description' in character_info:
                character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
                
        character_profile = character_profile.strip(' \n')
        if character_profile != '':
            involved_character_profiles[character] = character_profile

    # Create agents for all roles (characters + NSP)
    for character in speaking_characters_w_env + [NSP]:    
        # Configure agent based on role type
        if character == NSP:
            # Next Speaker Predictor agent
            system_prompt = get_nsp_prompt(speaking_characters_w_env, conversation['scenario'])
            character_database = None
        elif character == ENVIRONMENT:
            # Environment description agent
            system_prompt = get_environment_prompt(major_characters, conversation['scenario'])
            character_database = None
        else:
            # Character role-playing agent
            if retrieval and character in book_database['character_datasets']:
                # Set up retrieval database for character context
                character_database = book_database['character_datasets'][character]
                involved_plots = [_['i_p'] for _ in character_database['plots']] + \
                               [_['i_p'] for _ in character_database['conversations']] + \
                               [_['i_p'] for _ in character_database['utterances']]
                involved_plots = sorted(set(involved_plots))
                character_database['detailed_plots'] = [ book_database['plots'][i] for i in involved_plots ] 
            else:
                character_database = None

            # Build character context from profile and plot
            character_profile = involved_character_profiles.get(character, '')
            if character in plot_characters:
                character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
            character_profile = character_profile.strip(' \n')

            # Get character motivation if specified
            # find_motivation = [ c.get('motivation', '') for c in conversation['key_characters'] if c.get('name', '') == character]
            find_motivation = [ c.get('thought', '') for c in conversation['key_characters'] if c.get('name', '') == character]
            motivation = find_motivation[0] if find_motivation else ''

            # Configure prompt based on model type
            add_output_example = False if 'coser' in actor_model.lower() else True
            system_prompt = get_character_prompt(
                book_title, character, character_profile, plot["summary"],
                conversation["scenario"], motivation, thoughtless=args.wo_thought,
                other_character_profiles=involved_character_profiles,
                exclude_plot_summary=True, fixed_template=thinking_pattern,
                add_output_example=add_output_example, add_rag=retrieval
            )

        # Select appropriate model for the agent
        if character not in special_characters:
            character_model = actor_model  # Character role-playing
        elif character == ENVIRONMENT:
            character_model = env_model    # Environment description
        elif character == NSP:
            character_model = nsp_model    # Next speaker prediction
        else:
            raise ValueError(f'Invalid character: {character}')

        # Initialize the agent with its configuration
        character_agent = Agent(
            character_model, character, character_database,
            system_prompt=system_prompt,
            retrieval_target=retrieval if (retrieval and character not in special_characters) else None,
            thinking_pattern=thinking_pattern
        )
        character_agent.update('user', "===Conversation Start===\n\n")
        character_agents[character] = character_agent

    # Begin conversation simulation
    max_rounds = 20
    agent_conversations = []
    current_speaker = speaking_characters_w_env[0]  # Start with first character
    
    # Main conversation loop
    for i_round in range(max_rounds):
        if current_speaker == "<END CHAT>":
            break

        logger.info(f'===Round {i_round}===\n')
        
        # Generate responses for current speaker and get next speaker prediction
        for actor in [current_speaker, "NSP"]:
            current_agent = character_agents[actor]
            from utils import add_speaker_name
            
            # Use ground truth for early rounds if specified
            if args.continue_from > i_round:
                if actor == current_speaker:
                    response = conversation['dialogues'][i_round]['message']
                else:  # NSP
                    response = conversation['dialogues'][i_round+1]['character'] if i_round < len(conversation['dialogues']) - 1 else '<END CHAT>'
            else:
                # print(current_agent.messages)
                response = current_agent.chat()

            if actor == "NSP":
                # Process next speaker prediction
                next_actor = response.split(':')[0].strip() if ':' in response else response

                # Validate and set next speaker
                if "<END CHAT>" in response and i_round >= 5:
                    current_speaker = "<END CHAT>"
                elif next_actor in speaking_characters_w_env and next_actor != current_speaker:
                    current_speaker = next_actor
                else:
                    # Fallback to random selection if prediction is invalid
                    candidates = set(major_characters + [ENVIRONMENT]) - {current_speaker}
                    if not candidates:
                        candidates = set(speaking_characters_w_env) - {current_speaker}
                    candidates = list(candidates)
                    current_speaker = random.choice(candidates)
                
                logger.info(f"Next speaker: {current_speaker} (Raw response: {response})")
                agent_conversations.append({"role": actor, "content": next_actor})
                current_agent.update('assistant', next_actor)
            
            else:
                # Process character/environment response
                response = add_speaker_name(response, actor)
                logger.info(f"{env_model if actor == ENVIRONMENT else actor_model}: {response}\n")
                agent_conversations.append({"role": actor, "content": response})

                # Update conversation history for all agents
                for other_actor, other_agent in character_agents.items():
                    if other_actor == actor:
                        other_agent.update('assistant', response)
                    else:
                        other_agent.update('user', remove_inner_thoughts(response))
                    # logger.info(f'更新后的messages{current_agent.messages}')

    # Store simulation results for this circumstance
    result_item = {
        'book_title': book_title,
        'i_p': i_p,
        'i_c': i_c,
        'circumstance': circumstance,
        'simulation': agent_conversations,
        'involved_character_profiles': involved_character_profiles,
    }
    
    return result_item

def gca_simulation(test_file, actor_model, env_model, nsp_model, retrieval, thinking_pattern, nth_exp=0):
    # Set up caching file for model outputs
    from utils import set_cache_path
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Load test set
    test_dataset = json.load(open(test_file, 'r'))
    results = []

    # Configure output path based on model and retrieval settings
    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}_{thinking_pattern}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'

    logger.info(f'Conducting GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {simulation_path}')

    # Load existing results if available
    existing_results = []
    if os.path.exists(simulation_path) and not args.regenerate:
        existing_results = json.load(open(simulation_path, 'r'))
        logger.info(f'Loaded {len(existing_results)} existing results from {simulation_path}')

    # Filter out already processed circumstances
    circumstances_to_process = []
    for circumstance in test_dataset:
        circumstance_exists = False
        for existing_result in existing_results:
            if existing_result['circumstance']['topic'] == circumstance['topic']:
                circumstance_exists = True
                break
        
        if not circumstance_exists:
            circumstances_to_process.append(circumstance)
            logger.info(f'Will process: {circumstance["topic"]}')

    # Parallel processing of circumstances
    if args.num_workers > 1 and len(circumstances_to_process) > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_circumstance = {
                executor.submit(process_single_circumstance, circumstance, actor_model, env_model, nsp_model, retrieval, thinking_pattern): circumstance 
                for circumstance in circumstances_to_process
            }
            
            # Process completed tasks with progress bar
            progress_bar = tqdm(as_completed(future_to_circumstance), total=len(circumstances_to_process), desc="Simulating scenarios", unit="scenario")
            for future in progress_bar:
                try:
                    result_item = future.result()
                    results.append(result_item)
                    existing_results.append(result_item)
                    
                    # Save simulation results after each circumstance
                    os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
                    with open(simulation_path, 'w') as f:
                        json.dump(existing_results, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f'Saved simulation result for circumstance {result_item["i_c"]} to {simulation_path}')
                except Exception as e:
                    logger.error(f'Error processing circumstance: {e}')
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential processing (original code)
        progress_bar = tqdm(circumstances_to_process, desc="Simulating scenarios", unit="scenario")
        for circumstance in progress_bar:
            result_item = process_single_circumstance(circumstance, actor_model, env_model, nsp_model, retrieval, thinking_pattern)
            results.append(result_item)
            existing_results.append(result_item)
            
            # Save simulation results after each circumstance
            os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
            with open(simulation_path, 'w') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
            logger.info(f'Saved simulation result for circumstance {result_item["i_c"]} to {simulation_path}')

    return results

def evaluate_single_result(result, judge_model, thinking_pattern, dimensions, additional_instructions):
    """Evaluate a single simulation result"""
    book_title, i_p, i_c, circumstance, simulation = result['book_title'], result['i_p'], result['i_c'], result['circumstance'], result['simulation']
    
    # Verify indices match
    assert i_p == circumstance['plot']['i_p']
    assert i_c == circumstance['i_c']

    logger.info(f'Book {book_title}')

    # Filter out NSP messages and clean up simulation/reference for comparison
    simulation = result['simulation']
    simulation = [{"role":m["role"],"content":m["content"]} for m in simulation if m['role'] != NSP] #忽略思考
    reference = circumstance['dialogues']

    # Remove inner thoughts for fair comparison
    simulation = [ m if m['role'] == ENVIRONMENT else 
        {**m, 'content': remove_inner_thoughts(m['content'])} 
        for m in simulation  ]

    reference = [ m if m['character'] == ENVIRONMENT else 
        {**m, 'message': remove_inner_thoughts(m['message'])} 
        for m in reference  ]

    # Convert to readable string format for evaluation
    simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
    reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])
        
    logger.info(f'===Simulation===\n\n**************\n{simulation_str}\n\n**************\n\n===Reference===\n\n**************\n{reference_str}\n\n**************\n\n')

    # Prepare context information for evaluation
    scenario_str =  circumstance['scenario']
    character_profile_str = '\n\n'.join([f"### {character}\n\n{profile.strip('')}" for character, profile in result['involved_character_profiles'].items()])
    major_characters = circumstance['major_characters']

    logger.info(f'{book_title}-{i_p}-{i_c}-{scenario_str}')

    # Count non-environment messages for score adjustment
    actor_rounds = len([m for m in simulation if m['role'] != ENVIRONMENT])
    eval_result = {}

    # Helper function to validate evaluation response format
    def parse_response(response, **kwargs):
        try:
            assert isinstance(response, dict)
            for k, v in response.items():
                assert k in dimensions
                assert 'flaws' in v

                for f in v['flaws']:
                    if f.get('severity', None) is None:
                        f['severity'] = 1

            return response
        except:
            return False

    # Evaluate each dimension using LLM
    for dimension in dimensions:
        from prompts import critic_prompts
        critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])

        character_traits_list = []

        if 'key_characters' in circumstance:
            for character in circumstance['key_characters']:
                character_name = character['name']
                # Get English traits directly (no mapping needed)
                english_traits = character.get('morality_classification', {}).get('key_traits', [])
                
                if english_traits:
                    traits_str = f"{character_name}:[ {', '.join(english_traits)}]"
                    character_traits_list.append(traits_str)

        # Combine all character traits into a string
        character_traits_str = '; '.join(character_traits_list) if character_traits_list else 'No specific traits available'
        critic_prompt = critic_prompt.replace('{major_characters}', ', '.join(major_characters))
        critic_prompt = critic_prompt.replace('{character_traits}', character_traits_str)
        res = get_response_json(post_processing_funcs=[extract_json, parse_response], model=judge_model, messages=[{"role": "system", "content": critic_prompt}, {"role": "user", "content": simulation_str}])
        
        eval_result.update({dimension: res[dimension]})
        
        logger.info(json.dumps(res, ensure_ascii=False, indent=2)) 
        
        # Calculate dimension score with length penalty
        res[dimension]['score'] = max(0, min(100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)]) - 0.3 * actor_rounds) * 5, 100) )

    # Calculate automated metrics
    bleu, rouge_l = calculate_bleu_rouge(reference[args.continue_from:], simulation[args.continue_from:])
    eval_result['bleu'] = bleu
    eval_result['rouge_l'] = rouge_l

    # Get character levels from circumstance
    char_to_level = {}
    if 'key_characters' in circumstance:
        for character in circumstance['key_characters']:
            character_name = character['name']
            level = character.get('morality_classification', {}).get('level', None)
            if level is not None:
                char_to_level[character_name] = level
    
    # Calculate character scores using the new method
    character_scores = calculate_character_scores_for_case(
        {'simulation': result['simulation'], 'critique': eval_result},
        char_to_level
    )

    case_key = f'{book_title}-{i_p}-{i_c}'
    case_result = {
        'simulation': simulation,
        'simulation_str': simulation_str,
        'score': sum([eval_result[dimension]['score'] for dimension in dimensions]) / len(dimensions),
        'critique': eval_result,
        'character_scores': character_scores  # Add character scores to results
    }
    
    scores_dict = {dimension: eval_result[dimension]['score'] for dimension in dimensions}
    scores_dict['bleu'] = bleu
    scores_dict['rouge_l'] = rouge_l
    
    return case_key, case_result, scores_dict, character_scores

def gca_judging(test_file, actor_model, retrieval, judge_model, thinking_pattern, nth_exp=0):
    """
    Evaluates the quality of Given-Circumstance Acting (GCA) simulation results using multiple metrics.
    
    This function loads simulation results and evaluates them against reference dialogues using both automated metrics (BLEU, ROUGE-L) and LLM-based judgments across four dimensions:
    - Storyline Consistency: Measures alignment between the simulated conversation and original dialogue 
    - Anthropomorphism: Evaluates whether RPLAs behave in a human-like manner
    - Character Fidelity: Assesses whether RPLAs faithfully portray their characters
    - Storyline Quality: Evaluates whether the simulated conversation develops naturally

    Args:
        test_file (str): Path to JSON file containing test cases
        actor_model (str): Model name for character role-playing agents
        retrieval (str, optional): Type of retrieval data to enhance role-playing. Defaults to None (no retrieval).
        judge_model (str): Model name for LLM Judges.
        nth_exp (int, optional): Experiment ID.

    Returns:
        tuple: (avg_scores, cases)
            - avg_scores (dict): Average scores across all evaluation metrics
            - cases (dict): Detailed evaluation results for each test case
    """
    from utils import set_cache_path

    # Set up caching file for model outputs
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Configure paths based on model and retrieval settings
    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}_{thinking_pattern}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'
    evaluation_path = simulation_path.replace('/simulation/', '/evaluation/')

    logger.info(f'Evaluating GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {evaluation_path}')
    
    # Return cached evaluation results if available
    if os.path.exists(evaluation_path) and not (args.regenerate or args.reevaluate):
        res = json.load(open(evaluation_path, 'r'))
        dimensions = ['Character Fidelity']

        for dimension in dimensions:
            print(res['scores'][dimension])
        print(res['scores']['bleu'])
        print(res['scores']['rouge_l'])
        
        # Print level scores if available
        if 'level_scores' in res['scores']:
            print("\nAverage scores by character level:")
            for level, score_data in res['scores']['level_scores'].items():
                if isinstance(score_data, dict):
                    print(f"Level {level}: {score_data['average_score']:.3f} (n={score_data['num_characters']})")
                else:
                    print(f"Level {level}: {score_data:.3f}")

        return res['scores'], res['cases']
    
    # Load the simulation results
    simulation_results = json.load(open(simulation_path, 'r'))

    # Define evaluation dimensions
    dimensions = ['Character Fidelity']         
    scores = { d: [] for d in dimensions + ['bleu', 'rouge_l'] }
    cases = {}
    
    # Initialize level score tracking
    level_scores_accumulator = {1: [], 2: [], 3: [], 4: []}

    # Add special instructions for partial evaluation if needed
    additional_instructions = ''
    if args.continue_from > 0:
        additional_instructions = f'Please note that the first {args.continue_from} messages in the simulated conversation are the same as the reference. Focus your evaluation only on the content after these messages.'

    # Parallel evaluation of results
    if args.num_workers > 1 and len(simulation_results) > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all evaluation tasks
            future_to_result = {
                executor.submit(evaluate_single_result, result, judge_model, thinking_pattern, dimensions, additional_instructions): result 
                for result in simulation_results
            }
            
            # Process completed evaluations with progress bar
            progress_bar = tqdm(as_completed(future_to_result), total=len(simulation_results), desc="Evaluating simulations", unit="simulation")
            for future in progress_bar:
                try:
                    case_key, case_result, scores_dict, character_scores = future.result()
                    cases[case_key] = case_result
                    
                    # Accumulate scores
                    for dimension in dimensions:
                        scores[dimension].append(scores_dict[dimension])
                    scores['bleu'].append(scores_dict['bleu'])
                    scores['rouge_l'].append(scores_dict['rouge_l'])
                    
                    # Accumulate character level scores
                    for char_name, char_data in character_scores.items():
                        if 'level' in char_data and 'final_score' in char_data:
                            level = char_data['level']
                            if level in level_scores_accumulator:
                                level_scores_accumulator[level].append({
                                    'character': char_name,
                                    'score': char_data['final_score'],
                                    'case': case_key,
                                    'details': char_data
                                })
                    
                except Exception as e:
                    logger.error(f'Error evaluating result: {e}')
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential evaluation (original code)
        progress_bar = tqdm(simulation_results, desc="Evaluating simulations", unit="simulation")
        for result in progress_bar:
            case_key, case_result, scores_dict, character_scores = evaluate_single_result(result, judge_model, thinking_pattern, dimensions, additional_instructions)
            cases[case_key] = case_result
            
            # Accumulate scores
            for dimension in dimensions:
                scores[dimension].append(scores_dict[dimension])
            scores['bleu'].append(scores_dict['bleu'])
            scores['rouge_l'].append(scores_dict['rouge_l'])
            
            # Accumulate character level scores
            for char_name, char_data in character_scores.items():
                if 'level' in char_data and 'final_score' in char_data:
                    level = char_data['level']
                    if level in level_scores_accumulator:
                        level_scores_accumulator[level].append({
                            'character': char_name,
                            'score': char_data['final_score'],
                            'case': case_key,
                            'details': char_data
                        })

    # Calculate average scores across all dimensions
    avg_scores = {dimension: sum(scores[dimension]) / max(1, len(scores[dimension])) for dimension in dimensions}
    avg_scores['avg'] = sum(avg_scores.values()) / len(avg_scores)
    avg_scores.update({metric: sum(scores[metric]) / max(1, len(scores[metric])) for metric in ['bleu', 'rouge_l']})
    
    # Calculate average scores by level with detailed statistics
    level_avg_scores = {}
    for level in range(1, 5):
        if level in level_scores_accumulator and level_scores_accumulator[level]:
            level_data = level_scores_accumulator[level]
            scores_list = [item['score'] for item in level_data]
            level_avg_scores[level] = {
                'average_score': sum(scores_list) / len(scores_list),
                'num_characters': len(level_data),
                'min_score': min(scores_list),
                'max_score': max(scores_list)
            }
        else:
            level_avg_scores[level] = {
                'average_score': 0.0,
                'num_characters': 0,
                'min_score': 0.0,
                'max_score': 0.0
            }
    
    avg_scores['level_scores'] = level_avg_scores

    logger.info(f'{actor_setting}: Average score of {len(simulation_results)} samples: \n{avg_scores["avg"]} {avg_scores} on {test_file}')
    
    # Log level scores with detailed information
    logger.info("\n" + "="*60)
    logger.info("Average scores by character level (using new scoring method):")
    logger.info("Score Formula: 5 - 0.5*total_deduction - 0.5*(max_deduction/5) + 0.15*speech_count")
    logger.info("="*60)
    for level, score_data in level_avg_scores.items():
        logger.info(f"Level {level}:")
        logger.info(f"  Average: {score_data['average_score']:.3f}")
        logger.info(f"  Count: {score_data['num_characters']}")
        logger.info(f"  Range: [{score_data['min_score']:.3f}, {score_data['max_score']:.3f}]")

    # Save evaluation results
    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    with open(evaluation_path, 'w') as f:
        json.dump({'scores': avg_scores, 'cases': cases}, f, ensure_ascii=False, indent=2)
    return avg_scores, cases

if __name__ == "__main__":

    if args.nth_exp >= 0:
        nth_exps = [args.nth_exp]
    else:
        repeat_times = 3
        nth_exps = range(repeat_times)

    # Run experiments for each repeat
    for nth_exp in nth_exps:
        # Configure experiment name and logging
        exp_name = 'eval' 
        if args.continue_from > 0: exp_name += f'-continue_from={args.continue_from}'    
        if nth_exp > 0: exp_name += f'-repeat={nth_exp}'
        
        # 定义目标日志目录
        log_dir = "./logs"

        # 确保目录存在（如果不存在则创建）
        os.makedirs(log_dir, exist_ok=True)

        # 构建完整的日志文件路径
        log_filename = f"main-{exp_name}-{args.actor_model}-{args.thinking_pattern}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        # 设置logger
        logger = setup_logger(__name__, log_filepath)

        # Initialize result storage
        all_cases = {} 
        all_scores = {} 

        from concurrent.futures import ProcessPoolExecutor

        def generate(exp_args):
            """Run simulation for given experiment args"""
            actor_model, args, nth_exp = exp_args
        
            results = gca_simulation(
                args.test_file,
                actor_model, 
                args.env_model,
                args.nsp_model,
                args.retrieval,
                args.thinking_pattern,
                nth_exp
            )

            return results

        def evaluate(exp_args):
            """Run evaluation for given experiment args"""
            actor_model, args, nth_exp = exp_args

            scores, cases = gca_judging(
                args.test_file,
                actor_model,
                args.retrieval,
                args.judge_model,
                args.thinking_pattern,
                nth_exp,
            )

            return scores, cases
        
        # List of actor models to evaluate
        actor_models = [args.actor_model] # you can modify the list to expand to multiple models

        # Create experiment args for each actor model
        exp_args = [(actor_model, args, nth_exp) for actor_model in actor_models]

        # Parallel execution path when multiple workers available
        if args.num_workers > 1 and len(exp_args) > 1:
            # First run all generate tasks simultaneously
            generate_futures = []
            with ProcessPoolExecutor(max_workers=args.num_workers) as generate_executor:
                for exp_arg in exp_args:
                    future = generate_executor.submit(generate, exp_arg)
                    generate_futures.append((future, exp_arg))
            
            # As generate tasks complete, run evaluate tasks in parallel
            with ProcessPoolExecutor(max_workers=args.num_workers) as evaluate_executor:
                evaluate_futures = []
                
                # Process completed generate tasks and submit evaluates
                for generate_future, exp_arg in generate_futures:
                    generate_future.result()  # Wait for generate to complete
                    future = evaluate_executor.submit(evaluate, exp_arg)
                    evaluate_futures.append((future, exp_arg))
                
                # Process evaluate results as they complete
                for evaluate_future, exp_arg in evaluate_futures:
                    scores, cases = evaluate_future.result()

                    actor_model = exp_arg[0]
                    # Create identifier for this model run
                    actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'

                    all_scores[actor_setting] = scores
                    all_cases[actor_setting] = cases

        # Sequential execution path
        else:
            for exp_arg in exp_args:
                generate(exp_arg)
                scores, cases = evaluate(exp_arg)

                actor_model = exp_arg[0]
                # Create identifier for this model run
                actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'

                all_scores[actor_setting] = scores
                all_cases[actor_setting] = cases
                
        # Log final results
        logger.info(f'Evaluation results:\n{json.dumps(all_scores, ensure_ascii=False, indent=2)}')
        for i in all_scores:
            print(i)
            logger.info(i)