import os
import openai
from dotenv import load_dotenv
import time

# 1. 打开保险箱，拿 DeepSeek 的 Key
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") # 确保 .env 里写的是这个名字

if not api_key:
    raise ValueError("未找到 API Key，请检查 .env 配置！")

# 2. 连接 DeepSeek 大脑
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1" # DeepSeek 官方接口
)

def auto_annotate_bio(text_chunk):
    """把普通汉字变成 BIO 格式"""
    system_prompt = """
    你是一个极其严谨的医学 NLP 数据标注专家。你的任务是将输入的医患对话文本，转换为严格的 BIO 序列标注格式。
    
    【实体定义】
    - Symptom (症状)：如“发烧”、“肚子痛”。
    - Disease (疾病)：如“疝气”、“鞘膜积液”。
    
    【输出格式要求】
    1. 每个汉字、标点符号占一行，字和标签之间用一个空格隔开。
    2. 标签集限定为：O, B-Symptom, I-Symptom, B-Disease, I-Disease。
    3. 普通汉字和标点全部标为 O。
    4. 绝对不要输出任何 markdown 格式 (```) 或解释性废话，只输出纯文本。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请标注以下文本：\n{text_chunk}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"标注失败: {e}")
        return None

def process_dataset(input_file, output_file, max_lines=5):
    """流水线：读取 txt -> 扔给大模型 -> 存入新文件"""
    print(f"开始处理: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # 为了测试，我们先只读前几行
        lines = [next(fin).strip() for _ in range(max_lines)]
        
        for i, line in enumerate(lines):
            if not line or "http" in line or "id=" in line:
                continue 
                
            print(f"正在让大模型标注第 {i+1} 行...")
            bio_result = auto_annotate_bio(line)
            
            if bio_result:
                bio_result = bio_result.replace("```text", "").replace("```", "").strip()
                fout.write(bio_result + "\n\n") 
                
            time.sleep(1) # 停顿1秒，防止被 API 封锁
            
    print(f"处理完成！快去看看生成的 {output_file} 吧！")

# ================= 运行测试 =================
if __name__ == "__main__":
    # 注意：这里的路径要对应你截图里的文件名
    input_path = "Python\\智能问诊与医疗资源调度系统\\dataset\\2020.txt" 
    output_path = "Python\\智能问诊与医疗资源调度系统\\train_bio.txt"
    
    # 先用 10 行数据做个小测试
    process_dataset(input_path, output_path, max_lines=3000)