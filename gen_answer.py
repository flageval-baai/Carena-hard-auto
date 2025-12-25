import json
import os
import time
import concurrent.futures
import requests
import shortuuid
import tiktoken
from tqdm import tqdm

def load_questions(input_file):
    """
    加载问题数据
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        if 'turns' in item and len(item['turns']) > 0:
            # 找到用户的问题内容
            user_content = None
            for turn in item['turns']:
                if turn.get('role') == 'user':
                    user_content = turn.get('content', '')
                    break
            
            if user_content:
                questions.append({
                    "uid": str(item.get("question_id", len(questions))),
                    "prompt": user_content,
                    "metadata": {
                        "question_id": item.get("question_id"),
                        "category": item.get("category"),
                        "cluster": item.get("cluster"),
                        "language": item.get("language"),
                        "turns": item.get("turns"),
                        "original_data": item.get("original_data", {})
                    }
                })
    
    print(f"加载了 {len(questions)} 个问题")
    return questions

def load_existing_answers(answer_file):
    if not os.path.exists(answer_file):
        return {}
    
    existing = {}
    try:
        with open(answer_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ans = json.loads(line.strip())
                    uid = ans.get('uid')
                    if uid:
                        existing[uid] = ans
        print(f"找到 {len(existing)} 个已存在的答案")
    except Exception as e:
        print(f"读取已存在答案时出错: {e}")
    
    return existing

def call_openrouter_api(messages, model_name, api_config):
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config.get("temperature", 0.0),
        "max_tokens": api_config.get("max_tokens", 4096)
    }
    
    if "reasoning" in api_config and api_config["reasoning"]:
        payload["reasoning"] = api_config["reasoning"]
    
    max_retry = api_config.get("max_retry", 5)
    retry_sleep = api_config.get("retry_sleep", 10)
    
    for attempt in range(max_retry):
        try:
            response = requests.post(
                f"{api_config['api_base']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(30, 120)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait_time = retry_sleep * (2 ** attempt)
                print(f"API速率限制 (尝试 {attempt+1}/{max_retry}): 等待 {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"API调用失败 (尝试 {attempt+1}/{max_retry}): {response.status_code} - {response.text[:200]}")
                if attempt < max_retry - 1:
                    time.sleep(retry_sleep)
                else:
                    return api_config.get("error_output", "$ERROR$")
                    
        except requests.exceptions.Timeout as e:
            print(f"API请求超时 (尝试 {attempt+1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(retry_sleep * 2)
            else:
                return api_config.get("error_output", "$ERROR$")
        except requests.exceptions.ConnectionError as e:
            print(f"API连接错误 (尝试 {attempt+1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(retry_sleep * 2)
            else:
                return api_config.get("error_output", "$ERROR$")
        except Exception as e:
            print(f"API调用异常 (尝试 {attempt+1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(retry_sleep)
            else:
                return api_config.get("error_output", "$ERROR$")
    
    return api_config.get("error_output", "$ERROR$")

def create_answer_counter():
    import threading
    counter = {"value": 0}
    lock = threading.Lock()
    
    def get_next():
        with lock:
            counter["value"] += 1
            return counter["value"]
    
    return get_next

def get_answer(question, answer_file, model_name, api_config, answer_count, display_model_name=None, max_question_retry=3):
    if display_model_name is None:
        display_model_name = model_name
    
    question_retry_count = 0
    
    while question_retry_count < max_question_retry:
        try:
            time.sleep(0.5)
            
            messages = []
            if "sys_prompt" in api_config:
                messages.append({"role": "system", "content": api_config["sys_prompt"]})
            messages.append({"role": "user", "content": question["prompt"]})
            
            output = call_openrouter_api(messages, model_name, api_config)
            
            if output == api_config.get("error_output", "$ERROR$"):
                question_retry_count += 1
                if question_retry_count < max_question_retry:
                    wait_time = 5 * (2 ** question_retry_count)  # 指数退避
                    print(f"问题 {question['uid']} 处理失败，{wait_time}s后重试 ({question_retry_count}/{max_question_retry})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"✗ 问题 {question['uid']} 经过 {max_question_retry} 次重试仍然失败")
                    return False
            
            messages.append({"role": "assistant", "content": output})
            
            current_count = answer_count()
            
            ans = {
                "uid": question["uid"],
                "ans_id": shortuuid.uuid(),
                "answer_sequence": current_count,
                "model": display_model_name,
                "messages": messages,
                "tstamp": time.time(),
                "metadata": question.get("metadata", {})
            }
            
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o")
                token_len = len(encoding.encode(output, disallowed_special=()))
                ans["metadata"]["token_len"] = token_len
            except:
                ans["metadata"]["token_len"] = len(output.split())
            
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(answer_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(ans, ensure_ascii=False) + "\n")
            
            print(f"✓ 完成问题 {question['uid']} (#{current_count})")
            return True
            
        except Exception as e:
            question_retry_count += 1
            if question_retry_count < max_question_retry:
                wait_time = 5 * (2 ** question_retry_count)
                print(f"问题 {question['uid']} 发生异常: {e}，{wait_time}s后重试 ({question_retry_count}/{max_question_retry})")
                time.sleep(wait_time)
            else:
                print(f"问题 {question['uid']} 发生异常，经过 {max_question_retry} 次重试仍然失败: {e}")
                return False
    
    return False

def test_api_connection(api_config, model_name):
    print("正在测试API连接...")
    test_messages = [{"role": "user", "content": "你好，请简单回复一句话"}]
    
    result = call_openrouter_api(test_messages, model_name, api_config)
    if result != api_config.get("error_output", "$ERROR$"):
        print(f"✓ API连接测试成功，响应: {result[:50]}...")
        return True
    else:
        print("✗ API连接测试失败")
        return False

def process_questions_batch(questions_to_process, answer_file, model_name, api_config, answer_counter, display_model_name=None, max_workers=3, max_question_retry=3):
    successful_questions = []
    failed_questions = []
    
    if not questions_to_process:
        return successful_questions, failed_questions
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_question = {}
        for question in questions_to_process:
            future = executor.submit(
                get_answer,
                question,
                answer_file,
                model_name,
                api_config,
                answer_counter,
                display_model_name,
                max_question_retry
            )
            future_to_question[future] = question
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_question), 
            total=len(future_to_question),
            desc="处理问题"
        ):
            question = future_to_question[future]
            try:
                success = future.result()
                if success:
                    successful_questions.append(question)
                else:
                    failed_questions.append(question)
            except Exception as e:
                print(f"任务执行异常 - 问题 {question['uid']}: {e}")
                failed_questions.append(question)
    
    return successful_questions, failed_questions

def retry_failed_questions(failed_questions, answer_file, model_name, api_config, answer_counter, display_model_name=None, max_batch_retry=2, max_question_retry=3):

    current_failed = failed_questions.copy()
    batch_retry_count = 0
    
    while current_failed and batch_retry_count < max_batch_retry:
        batch_retry_count += 1
        print(f"\n=== 批次重试 {batch_retry_count}/{max_batch_retry}: 重试 {len(current_failed)} 个失败的问题 ===")
        
        retry_workers = max(1, len(current_failed) // 10)
        
        successful, still_failed = process_questions_batch(
            current_failed, 
            answer_file, 
            model_name, 
            api_config, 
            answer_counter,
            display_model_name,
            max_workers=retry_workers,
            max_question_retry=max_question_retry
        )
        
        print(f"批次重试 {batch_retry_count} 完成: 成功 {len(successful)} 个, 仍失败 {len(still_failed)} 个")
        
        current_failed = still_failed
        
        if current_failed and batch_retry_count < max_batch_retry:
            wait_time = 30 * batch_retry_count 
            print(f"等待 {wait_time}s 后进行下一轮重试...")
            time.sleep(wait_time)
    
    return current_failed

def save_failed_questions(failed_questions, filename="failed_questions.json"):
    if failed_questions:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(failed_questions, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(failed_questions)} 个失败的问题到 {filename}")

def main():
    config = {
        "input_file": "",
        "model_name": "google/gemini-2.5-pro",
        "display_model_name": "gemini-2.5-pro-think",
        "parallel": 3,
        "temperature": 0.0,
        "max_tokens": 4096,
        "api_base": "",
        "api_key": "",
        "max_retry": 5,
        "retry_sleep": 10,
        "error_output": "$ERROR$",
        "max_batch_retry": 2,
        "max_question_retry": 3,
        "reasoning": {"effort": "high"}
    }
    
    api_config = {
        "api_base": config["api_base"],
        "api_key": config["api_key"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "max_retry": config["max_retry"],
        "retry_sleep": config["retry_sleep"],
        "error_output": config["error_output"],
        "reasoning": config.get("reasoning")
    }
    
    answer_dir = "answer_dir"
    os.makedirs(answer_dir, exist_ok=True)
    
    questions = load_questions(config["input_file"])
    if not questions:
        print("没有找到有效的问题")
        return
    
    if not test_api_connection(api_config, config["model_name"]):
        print("API连接失败，请检查网络连接和API密钥")
        return
    
    model_safe_name = config["display_model_name"].replace("/", "_")
    answer_file = os.path.join(answer_dir, f"{model_safe_name}.jsonl")
    print(f"输出到: {answer_file}")
    
    existing_answers = load_existing_answers(answer_file)
    
    questions_to_process = []
    skipped_count = 0
    
    for question in questions:
        if question["uid"] in existing_answers:
            skipped_count += 1
            continue
        questions_to_process.append(question)
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个已存在的答案")
    
    print(f"需要处理 {len(questions_to_process)} 个问题")
    
    if not questions_to_process:
        print("所有问题都已处理完成！")
        return
    
    answer_counter = create_answer_counter()
    
    print(f"\n=== 开始处理 {len(questions_to_process)} 个问题 ===")
    successful_questions, failed_questions = process_questions_batch(
        questions_to_process, 
        answer_file, 
        config["model_name"], 
        api_config, 
        answer_counter,
        display_model_name=config.get("display_model_name"),
        max_workers=config["parallel"],
        max_question_retry=config["max_question_retry"]
    )
    
    print(f"\n第一轮处理完成: 成功 {len(successful_questions)} 个, 失败 {len(failed_questions)} 个")
    
    final_failed = []
    if failed_questions:
        print(f"\n开始重试失败的问题...")
        final_failed = retry_failed_questions(
            failed_questions, 
            answer_file, 
            config["model_name"], 
            api_config, 
            answer_counter,
            display_model_name=config.get("display_model_name"),
            max_batch_retry=config["max_batch_retry"],
            max_question_retry=config["max_question_retry"]
        )
        
        if final_failed:
            print(f"\n仍有 {len(final_failed)} 个问题最终失败")
            save_failed_questions(final_failed, f"failed_questions_{model_safe_name}.json")
            
            failed_uids = [q["uid"] for q in final_failed]
            print(f"最终失败的问题ID: {failed_uids}")
        else:
            print(f"\n🎉 所有问题都已成功处理！")
    else:
        print(f"\n🎉 所有问题在第一轮就成功处理了！")
    
    total_processed = len(questions_to_process)
    final_failed_count = len(final_failed)
    success_count = total_processed - final_failed_count
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n=== 最终统计 ===")
    print(f"总问题数: {len(questions)}")
    print(f"跳过已存在: {skipped_count}")
    print(f"需要处理: {total_processed}")
    print(f"成功处理: {success_count}")
    print(f"最终失败: {final_failed_count}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"结果保存在: {answer_file}")

if __name__ == "__main__":
    main()