from collections import Counter

# 注意：确保这里填的是你最新生成的那个大数据集文件的路径！
DATA_FILE = "D:\\code\\C++\\Python\\智能问诊与医疗资源调度系统\\train_bio.txt" 

print(f"正在给 {DATA_FILE} 做全身体检...\n")

try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        labels = []
        sentence_count = 0
        
        for line in f:
            line = line.strip()
            if not line:
                sentence_count += 1
                continue
                
            parts = line.split(" ")
            if len(parts) >= 2:
                # 把每行后面的那个标签提取出来
                labels.append(parts[-1]) 

    print(f"📊 体检报告出炉：")
    print(f"1. 你的教材里总共有: {sentence_count} 句话。")
    print(f"2. 标签的种类和数量分布如下:")
    
    # 统计每个标签出现了多少次
    counter = Counter(labels)
    for label, count in counter.most_common():
        print(f"   - {label} : {count} 次")

except FileNotFoundError:
    print(f"❌ 找不到文件 {DATA_FILE}，请检查路径！")