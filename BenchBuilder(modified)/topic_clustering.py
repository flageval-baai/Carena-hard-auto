import json
import argparse
import os


import torch
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

import openai
import tiktoken
from bertopic.representation import OpenAI

import time
import random

def run(args):
    print("正在加载中文embedding模型...")
    try:
        embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        print("模型加载完成！")
    except Exception as e:
        print(f"❌ Embedding模型加载失败: {e}")
        print("🔧 尝试使用备用模型...")
        try:
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 备用模型加载成功")
        except Exception as e2:
            print(f"❌ 备用模型也加载失败: {e2}")
            raise
    
    embeddings_file = f"{args.output_dir}/embeddings.npy"
    post_process_file = f"{args.output_dir}/post_process_convs.json"
    
    if args.embedding_file is not None:
        embeddings = np.load(args.embedding_file)
        if args.post_process_conv is not None:
            all_convs = json.load(open(args.post_process_conv))
        else:
            raise ValueError("Please provide post process conv file")

        convs = []
        for row in all_convs:
            convs.append(row["post_process_conv"])
            
    elif os.path.exists(embeddings_file) and os.path.exists(post_process_file):
        
        try:
            embeddings = np.load(embeddings_file)
            all_convs_new = json.load(open(post_process_file, 'r', encoding='utf-8'))
            convs = [row["post_process_conv"] for row in all_convs_new]
            
        
            if len(convs) != embeddings.shape[0]:
                print(f"数据长度不一致: 对话{len(convs)} vs embeddings{embeddings.shape[0]}")
                min_len = min(len(convs), embeddings.shape[0])
                convs = convs[:min_len]
                embeddings = embeddings[:min_len]
                print(f"已调整为一致长度: {min_len}")
            
        except Exception as e:
            print(f"恢复数据失败: {e}")
            print("将重新处理原始数据...")
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
            if os.path.exists(post_process_file):
                os.remove(post_process_file)
            embeddings = None
            convs = None
            
    else:
        embeddings = None
        convs = None
    
    if embeddings is None or convs is None:
        if not args.conv_file:
            raise ValueError("没有找到可恢复的数据，请提供 --conv-file 参数")
            
        all_convs = json.load(open(args.conv_file))

        if args.first_n is not None:
            all_convs = all_convs[:args.first_n]

        all_convs_new = []
        convs = []
        for row in all_convs:
            try:
                contents_data = json.loads(row["contents"])
                
                conv = ""
                for item in contents_data:
                    if item.get("type") == "text" and "content" in item:
                        conv += f"{item['content']}\n"
                
                conv = conv.replace("<|endoftext|>", "<| endoftext |>")
                conv = conv.strip()
                
                if len(conv) <= 32:
                    continue
                    
                conv_truncated = conv[:10000]
                convs.append(conv_truncated)
                
                row_copy = row.copy()
                row_copy["post_process_conv"] = conv_truncated
                all_convs_new.append(row_copy)
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"跳过无效记录 ID {row.get('id', 'unknown')}: {e}")
                continue

        print(f"成功处理 {len(convs)} 条对话")

        print("保存处理后的对话数据...")
        try:
            with open(f"{args.output_dir}/post_process_convs.json", "w", encoding="utf-8") as f:
                json.dump(all_convs_new, f, indent=4, ensure_ascii=False)
            print("对话数据保存成功")
        except Exception as e:
            print(f"对话数据保存失败: {e}")

        print("📄 开始生成embeddings...")
        batch_size = 32
        embeddings = []
        
        start_batch = 0
        temp_embeddings_file = f"{args.output_dir}/embeddings_temp.npy"
        
        if os.path.exists(temp_embeddings_file):
            try:
                temp_embeddings = np.load(temp_embeddings_file)
                start_batch = temp_embeddings.shape[0] // batch_size
                embeddings = [temp_embeddings[i:i+batch_size] for i in range(0, temp_embeddings.shape[0], batch_size)]
                print(f"发现临时embeddings，从第 {start_batch+1} 批次恢复...")
            except:
                print("临时embeddings文件损坏，重新开始...")
                start_batch = 0
                embeddings = []
        
        for i in tqdm(range(start_batch * batch_size, len(convs), batch_size), desc="生成embeddings"):
            convs_batch = convs[i : i + batch_size]
            try:
                batch_embeddings = embedding_model.encode(
                    convs_batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    current_embeddings = np.vstack(embeddings)
                    np.save(temp_embeddings_file, current_embeddings)
                    print(f"临时保存embeddings进度: {current_embeddings.shape[0]}/{len(convs)}")
                
            except Exception as e:
                print(f"Embedding生成错误 (batch {i//batch_size + 1}): {e}")
                if embeddings:
                    current_embeddings = np.vstack(embeddings)
                    np.save(temp_embeddings_file, current_embeddings)
                    print(f"已保存当前进度: {current_embeddings.shape[0]} embeddings")
                raise
        
        embeddings = np.vstack(embeddings)
        
        np.save(f"{args.output_dir}/embeddings.npy", embeddings)
        if os.path.exists(temp_embeddings_file):
            os.remove(temp_embeddings_file)
        print(f"✅ Embeddings已保存，形状: {embeddings.shape}")

    print(f"📊 总对话数: {len(convs)}")
        
    print("配置表示模型...")
    representation_model = {}
    
    try:
        keybert_model = KeyBERTInspired()
        representation_model["KeyBERT"] = keybert_model
        print("KeyBERT配置成功")
    except Exception as e:
        print(f"KeyBERT配置失败: {e}")
    
    if hasattr(args, 'use_openai') and args.use_openai:
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            
           YOUR_API_KEY = ""  # 请替换为您的OpenAI API密钥
            prompt = """
            我有一个主题包含以下文档：
            [DOCUMENTS]
            这个主题由以下关键词描述：[KEYWORDS]

            基于以上信息，请提取一个简短但高度描述性的主题标签，最多5个词。请确保格式如下：
            主题：<主题标签>
            """
            
            openai_model = OpenAI(
                client, 
                model="openai/gpt-4o-mini",
                exponential_backoff=True, 
                chat=True, 
                prompt=prompt, 
                nr_docs=20,
                doc_length=200,
                tokenizer=tokenizer,
            )
            representation_model["OpenAI"] = openai_model
            print("OpenAI模型配置成功")
            
        except Exception as e:
            print(f"OpenAI配置失败: {e}")
            print("将仅使用KeyBERT进行主题表示...")
    
    if embedding_model is None:
        print("重新初始化embedding模型...")
        try:
            embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        except:
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    if not representation_model:
        print("使用默认KeyBERT表示模型...")
        representation_model = KeyBERTInspired()

    print("创建BERTopic模型...")
    try:
        topic_model = BERTopic(
            verbose=True,
            embedding_model=embedding_model,
            representation_model=representation_model,
            min_topic_size=args.min_topic_size,
        )
        print("BERTopic模型创建成功")
    except Exception as e:
        print(f"BERTopic创建失败: {e}")
        print("尝试使用简化配置...")
        topic_model = BERTopic(
            verbose=True,
            embedding_model=embedding_model,
            min_topic_size=args.min_topic_size,
        )
        print("简化BERTopic模型创建成功")
    
    print("📄 开始训练BERTopic模型...")
    try:
        topics, _ = topic_model.fit_transform(convs, embeddings)
        print(f"发现 {len(topic_model.get_topic_info())} 个主题")
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("尝试使用更简单的配置重新训练...")
        
        simple_topic_model = BERTopic(
            verbose=True,
            embedding_model=embedding_model,
            min_topic_size=max(args.min_topic_size, 5),
        )
        topics, _ = simple_topic_model.fit_transform(convs, embeddings)
        topic_model = simple_topic_model
        print(f"简化模型训练成功，发现 {len(topic_model.get_topic_info())} 个主题")

    print("处理离群值...")
    try:
        new_topics = topic_model.reduce_outliers(convs, topics)
        print("离群值处理完成")
    except Exception as e:
        print(f"离群值处理失败: {e}")
        print("跳过离群值处理，使用原始主题分配...")
        new_topics = topics
    
    print("保存主题分配结果...")
    try:
        with open(f"{args.output_dir}/conv_topics.json", "w", encoding="utf-8") as f:
            json.dump(new_topics, f, default=str, ensure_ascii=False)
        print("主题分配结果保存成功")
    except Exception as e:
        print(f"主题分配结果保存失败: {e}")

    print("保存模型...")
    try:
        topic_model.save(
            f"{args.output_dir}/model_dir", 
            serialization="pickle", 
            save_ctfidf=True, 
            save_embedding_model=False
        )
        print("模型保存成功")
    except Exception as e:
        print(f"模型保存失败: {e}")
        print("尝试保存模型基本信息...")
        try:
            topic_info = topic_model.get_topic_info()
            topic_info.to_pickle(f"{args.output_dir}/topic_info.pkl")
            print("主题信息保存成功")
        except Exception as e2:
            print(f"主题信息保存也失败: {e2}")
    
    print("保存主题信息...")
    try:
        df = topic_model.get_topic_info()
        df.to_csv(f"{args.output_dir}/topics.csv", index=False, encoding="utf-8")
        
        topics_dict = {}
        for topic_id in df['Topic'].values:
            if topic_id != -1:
                try:
                    topic_words = topic_model.get_topic(topic_id)
                    topics_dict[str(topic_id)] = topic_words
                except:
                    continue
        
        with open(f"{args.output_dir}/topic_words.json", "w", encoding="utf-8") as f:
            json.dump(topics_dict, f, ensure_ascii=False, indent=2)
            
        print("主题信息保存成功")
        print(f"主题总数: {len(df)}")
        print(f"有效主题数: {len([t for t in df['Topic'].values if t != -1])}")
        
    except Exception as e:
        print(f"主题信息保存失败: {e}")
    
    print(f"处理完成！结果保存在 {args.output_dir} 目录")
    print("\n生成的文件列表:")
    for file in os.listdir(args.output_dir):
        file_path = os.path.join(args.output_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1024 / 1024  
            print(f"  📄 {file} ({size:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用BERTopic进行主题聚类分析")
    parser.add_argument("--conv-file", type=str, required=False, help="输入的JSON对话文件路径")
    parser.add_argument("--min-topic-size", type=int, default=32, help="最小主题大小")
    parser.add_argument("--embedding-file", type=str, default=None, help="预计算的embedding文件路径")
    parser.add_argument("--post-process-conv", type=str, default=None, help="预处理的对话文件路径")
    parser.add_argument("--output-dir", type=str, default="topic_model_dir", help="输出目录")
    parser.add_argument("--first-n", type=int, default=None, help="只处理前N条记录")
    parser.add_argument("--use-openai", action="store_true", help="是否使用OpenAI进行主题标签生成")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n用户中断程序")
        print("已保存的文件可用于后续恢复")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n故障排除提示:")
        print("1. 检查输入文件格式是否正确")
        print("2. 确保有足够的内存和GPU资源")
        print("3. 检查网络连接（如果使用OpenAI API）")
        print("4. 查看上面的错误信息进行具体诊断")